"""Fine-tune WavLM Base+ on the FSC22 paper-compatible augmented protocol.

Run from the FSC22_Research project root:

    python src\\train_wavlm_paper_protocol.py

The first run downloads ``microsoft/wavlm-base-plus`` (internet required until
the model is loaded). Training then uses the local waveform cache and can
continue without internet. The configuration is designed for a 4-GB RTX 3050:
batch size 2, gradient accumulation, AMP, and only the top four WavLM encoder
blocks unfrozen.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import WavLMForSequenceClassification


MODEL_NAME = "microsoft/wavlm-base-plus"
SEED = 42
NUM_CLASSES = 27
EPOCHS = 20
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8
BACKBONE_LEARNING_RATE = 5e-6
HEAD_LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 6
UNFROZEN_ENCODER_BLOCKS = 4
LABEL_SMOOTHING = 0.05

ROOT = Path.cwd()
MANIFEST_FILE = ROOT / "outputs" / "wavlm_augmented_split_seed42.csv"
MODEL_PATH = ROOT / "models" / "fsc22_wavlm_paper_protocol_seed42.pt"
RESULT_FOLDER = ROOT / "outputs" / "wavlm_paper_protocol"
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
RESULT_FOLDER.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class WavLMDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        class_to_index: dict[int, int],
        training: bool,
    ) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.class_to_index = class_to_index
        self.training = training

    def __len__(self) -> int:
        return len(self.dataframe)

    @staticmethod
    def augment(waveform: torch.Tensor) -> torch.Tensor:
        if random.random() < 0.35:
            waveform = waveform * random.uniform(0.75, 1.25)

        if random.random() < 0.30:
            signal_rms = waveform.square().mean().sqrt().clamp_min(1e-4)
            noise_scale = signal_rms * random.uniform(0.002, 0.010)
            waveform = waveform + noise_scale * torch.randn_like(waveform)

        if random.random() < 0.20:
            maximum_shift = min(1600, waveform.numel() // 20)
            shift = random.randint(-maximum_shift, maximum_shift)
            waveform = torch.roll(waveform, shifts=shift)

        return waveform.clamp(-1.0, 1.0)

    @staticmethod
    def normalize(waveform: torch.Tensor) -> torch.Tensor:
        waveform = waveform - waveform.mean()
        variance = waveform.square().mean()
        return waveform / torch.sqrt(variance + 1e-7)

    def __getitem__(self, index: int):
        row = self.dataframe.iloc[index]
        audio_path = ROOT / str(row["Audio Path"])
        waveform = torch.from_numpy(np.load(audio_path).astype(np.float32))
        if self.training:
            waveform = self.augment(waveform)
        waveform = self.normalize(waveform)
        label = self.class_to_index[int(row["Class ID"])]
        return waveform, torch.tensor(label, dtype=torch.long)


def apply_mixup(
    waveforms: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if random.random() >= 0.30:
        return waveforms, labels, labels, 1.0

    weight = float(np.random.beta(alpha, alpha))
    permutation = torch.randperm(waveforms.size(0), device=waveforms.device)
    mixed = weight * waveforms + (1.0 - weight) * waveforms[permutation]
    return mixed, labels, labels[permutation], weight


def load_pretrained_model(
    id_to_label: dict[int, str],
    label_to_id: dict[str, int],
) -> WavLMForSequenceClassification:
    arguments = {
        "num_labels": NUM_CLASSES,
        "id2label": id_to_label,
        "label2id": label_to_id,
        "ignore_mismatched_sizes": True,
    }
    try:
        print("Loading WavLM Base+ from the local model cache...")
        return WavLMForSequenceClassification.from_pretrained(
            MODEL_NAME,
            local_files_only=True,
            **arguments,
        )
    except OSError:
        print("WavLM Base+ is not cached. Downloading it now...")
        print("Keep Wi-Fi connected until 'WAVLM LOADED SUCCESSFULLY' appears.")
        model = WavLMForSequenceClassification.from_pretrained(
            MODEL_NAME,
            local_files_only=False,
            **arguments,
        )
        print("WAVLM LOADED SUCCESSFULLY")
        return model


def configure_trainable_parameters(
    model: WavLMForSequenceClassification,
) -> tuple[list[nn.Parameter], list[nn.Parameter], int, int]:
    for parameter in model.parameters():
        parameter.requires_grad = False

    if not hasattr(model.wavlm.encoder, "layers"):
        raise AttributeError("Could not locate WavLM encoder layers.")

    backbone_parameters: list[nn.Parameter] = []
    for block in model.wavlm.encoder.layers[-UNFROZEN_ENCODER_BLOCKS:]:
        for parameter in block.parameters():
            parameter.requires_grad = True
            backbone_parameters.append(parameter)

    # The final encoder normalization influences the pooled representation.
    if hasattr(model.wavlm.encoder, "layer_norm"):
        for parameter in model.wavlm.encoder.layer_norm.parameters():
            parameter.requires_grad = True
            backbone_parameters.append(parameter)

    head_parameters: list[nn.Parameter] = []
    for module_name in ("projector", "classifier"):
        module = getattr(model, module_name, None)
        if module is None:
            raise AttributeError(f"WavLM classification head lacks {module_name}.")
        for parameter in module.parameters():
            parameter.requires_grad = True
            head_parameters.append(parameter)

    # Some WavLM configurations learn a weighted sum of hidden layers.
    if hasattr(model, "layer_weights") and model.layer_weights is not None:
        model.layer_weights.requires_grad = True
        head_parameters.append(model.layer_weights)

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return (
        backbone_parameters,
        head_parameters,
        total_parameters,
        trainable_parameters,
    )


def metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_precision": float(
            precision_score(labels, predictions, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(labels, predictions, average="macro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
    }


@torch.inference_mode()
def evaluate(
    model: WavLMForSequenceClassification,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    amp_enabled: bool,
) -> dict[str, object]:
    model.eval()
    total_loss = 0.0
    labels_parts: list[np.ndarray] = []
    logits_parts: list[np.ndarray] = []

    for waveforms, labels in loader:
        waveforms = waveforms.to(device, non_blocking=True)
        labels_device = labels.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            logits = model(input_values=waveforms).logits
            loss = loss_function(logits, labels_device)

        total_loss += loss.item() * labels.size(0)
        labels_parts.append(labels.numpy())
        logits_parts.append(logits.float().cpu().numpy())

    all_labels = np.concatenate(labels_parts)
    all_logits = np.concatenate(logits_parts)
    predictions = all_logits.argmax(axis=1)
    result: dict[str, object] = metrics(all_labels, predictions)
    result.update(
        {
            "loss": total_loss / len(loader.dataset),
            "labels": all_labels,
            "logits": all_logits,
            "predictions": predictions,
        }
    )
    return result


def save_figures(
    history: pd.DataFrame,
    labels: np.ndarray,
    predictions: np.ndarray,
    class_names: list[str],
) -> None:
    sns.set_theme(style="whitegrid")
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(history["epoch"], history["train_loss"], label="Training")
    axes[0].plot(history["epoch"], history["validation_loss"], label="Validation")
    axes[0].set_title("WavLM Fine-Tuning Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].legend()

    axes[1].plot(
        history["epoch"], history["validation_accuracy"] * 100, label="Accuracy"
    )
    axes[1].plot(
        history["epoch"], history["validation_f1"] * 100, label="Macro-F1"
    )
    axes[1].axhline(92.59, color="black", linestyle="--", label="Base paper")
    axes[1].axhline(95.00, color="green", linestyle=":", label="95%")
    axes[1].axhline(97.00, color="blue", linestyle=":", label="97% target")
    axes[1].set_title("WavLM Validation Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score (%)")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(RESULT_FOLDER / "wavlm_training_curves.png", dpi=300)
    plt.close(figure)

    matrix = confusion_matrix(labels, predictions)
    figure = plt.figure(figsize=(18, 15))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="GnBu",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False,
    )
    plt.title("FSC22 WavLM Confusion Matrix")
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    figure.savefig(RESULT_FOLDER / "wavlm_confusion_matrix.png", dpi=300)
    plt.close(figure)


def main() -> None:
    set_seed(SEED)
    torch.set_float32_matmul_precision("high")
    if not MANIFEST_FILE.exists():
        raise FileNotFoundError(f"WavLM manifest not found: {MANIFEST_FILE}")

    manifest = pd.read_csv(MANIFEST_FILE)
    manifest["Class ID"] = pd.to_numeric(manifest["Class ID"]).astype(int)
    required_columns = {
        "Audio Path",
        "Original Dataset File Name",
        "Variant",
        "Class ID",
        "Class Name",
        "Split",
    }
    missing_columns = required_columns.difference(manifest.columns)
    if missing_columns:
        raise KeyError(f"Manifest is missing: {sorted(missing_columns)}")

    missing_audio = [
        path
        for path in manifest["Audio Path"].astype(str)
        if not (ROOT / path).exists()
    ]
    if missing_audio:
        raise FileNotFoundError(f"Missing {len(missing_audio)} waveform arrays.")

    class_table = (
        manifest[["Class ID", "Class Name"]]
        .drop_duplicates()
        .sort_values("Class ID")
    )
    class_ids = class_table["Class ID"].tolist()
    class_names = class_table["Class Name"].astype(str).tolist()
    class_to_index = {
        class_id: index for index, class_id in enumerate(class_ids)
    }
    id_to_label = {index: name for index, name in enumerate(class_names)}
    label_to_id = {name: index for index, name in id_to_label.items()}

    train_dataframe = manifest[manifest["Split"] == "train"].copy()
    validation_dataframe = manifest[manifest["Split"] == "test"].copy()
    train_dataset = WavLMDataset(train_dataframe, class_to_index, training=True)
    validation_dataset = WavLMDataset(
        validation_dataframe, class_to_index, training=False
    )

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if not use_cuda:
        raise RuntimeError("CUDA GPU is required for this WavLM training script.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    model = load_pretrained_model(id_to_label, label_to_id)
    (
        backbone_parameters,
        head_parameters,
        total_parameters,
        trainable_parameters,
    ) = configure_trainable_parameters(model)
    model.to(device)

    loss_function = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": BACKBONE_LEARNING_RATE},
            {"params": head_parameters, "lr": HEAD_LEARNING_RATE},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=2e-7
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    torch.cuda.reset_peak_memory_stats()

    print("\nFSC22 WAVLM PAPER-PROTOCOL TRAINING")
    print("-----------------------------------")
    print("GPU:", torch.cuda.get_device_name(0))
    print("Training samples:", len(train_dataset))
    print("Validation samples:", len(validation_dataset))
    print("Unfrozen encoder blocks:", UNFROZEN_ENCODER_BLOCKS)
    print("Total parameters:", f"{total_parameters:,}")
    print("Trainable parameters:", f"{trainable_parameters:,}")
    print("Batch size:", BATCH_SIZE)
    print("Effective batch size:", BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)
    print("Maximum epochs:", EPOCHS)

    history_rows: list[dict[str, float | int]] = []
    best_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_training_loss = 0.0

        for step, (waveforms, labels) in enumerate(train_loader, start=1):
            waveforms = waveforms.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            waveforms, labels_a, labels_b, mixing_weight = apply_mixup(
                waveforms, labels
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=True,
            ):
                logits = model(input_values=waveforms).logits
                loss = (
                    mixing_weight * loss_function(logits, labels_a)
                    + (1.0 - mixing_weight) * loss_function(logits, labels_b)
                )
                scaled_loss = loss / GRADIENT_ACCUMULATION_STEPS

            scaler.scale(scaled_loss).backward()
            total_training_loss += loss.item() * labels.size(0)

            if (
                step % GRADIENT_ACCUMULATION_STEPS == 0
                or step == len(train_loader)
            ):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad), 1.0
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        training_loss = total_training_loss / len(train_dataset)
        validation = evaluate(
            model,
            validation_loader,
            loss_function,
            device,
            amp_enabled=True,
        )
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": training_loss,
                "validation_loss": float(validation["loss"]),
                "validation_accuracy": float(validation["accuracy"]),
                "validation_precision": float(validation["macro_precision"]),
                "validation_recall": float(validation["macro_recall"]),
                "validation_f1": float(validation["macro_f1"]),
                "backbone_learning_rate": optimizer.param_groups[0]["lr"],
                "head_learning_rate": optimizer.param_groups[1]["lr"],
            }
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train loss {training_loss:.4f} | "
            f"Val loss {float(validation['loss']):.4f} | "
            f"Acc {100 * float(validation['accuracy']):.2f}% | "
            f"F1 {100 * float(validation['macro_f1']):.2f}% | "
            f"LR {optimizer.param_groups[0]['lr']:.2e}"
        )

        if float(validation["macro_f1"]) > best_f1 + 1e-4:
            best_f1 = float(validation["macro_f1"])
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            epochs_without_improvement += 1

        scheduler.step()
        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping after epoch {epoch}.")
            break

    history = pd.DataFrame(history_rows)
    history.to_csv(RESULT_FOLDER / "wavlm_history.csv", index=False)

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device, weights_only=True)
    )
    final_result = evaluate(
        model,
        validation_loader,
        loss_function,
        device,
        amp_enabled=False,
    )
    labels = np.asarray(final_result["labels"])
    logits = np.asarray(final_result["logits"])
    predictions = np.asarray(final_result["predictions"])
    np.save(RESULT_FOLDER / "wavlm_test_labels.npy", labels)
    np.save(RESULT_FOLDER / "wavlm_test_logits.npy", logits)

    report = classification_report(
        labels,
        predictions,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(
        RESULT_FOLDER / "wavlm_classification_report.csv"
    )

    prediction_table = validation_dataframe[
        [
            "Original Dataset File Name",
            "Variant",
            "Class ID",
            "Class Name",
        ]
    ].reset_index(drop=True)
    prediction_table["true_index"] = labels
    prediction_table["wavlm_prediction"] = [
        class_names[index] for index in predictions
    ]
    prediction_table["correct"] = predictions == labels
    prediction_table.to_csv(RESULT_FOLDER / "wavlm_predictions.csv", index=False)

    save_figures(history, labels, predictions, class_names)
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2)
    summary = {
        "model": "WavLM Base+ paper-protocol classifier",
        "pretrained_checkpoint": MODEL_NAME,
        "seed": SEED,
        "best_epoch": best_epoch,
        "unfrozen_encoder_blocks": UNFROZEN_ENCODER_BLOCKS,
        "accuracy": float(final_result["accuracy"]),
        "macro_precision": float(final_result["macro_precision"]),
        "macro_recall": float(final_result["macro_recall"]),
        "macro_f1": float(final_result["macro_f1"]),
        "correct_predictions": int((predictions == labels).sum()),
        "test_samples": int(len(labels)),
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "peak_vram_mb": peak_vram_mb,
        "base_paper_accuracy": 0.9259,
        "protocol_warning": (
            "Paper-compatible augmentation-overlap protocol; not clean "
            "unseen-recording generalization."
        ),
    }
    with open(
        RESULT_FOLDER / "wavlm_metrics.json", "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=4)

    print("\nBEST WAVLM PAPER-PROTOCOL RESULTS")
    print("---------------------------------")
    print("Best epoch:", best_epoch)
    print(f"Accuracy:        {100 * float(final_result['accuracy']):.2f}%")
    print(f"Macro precision: {100 * float(final_result['macro_precision']):.2f}%")
    print(f"Macro recall:    {100 * float(final_result['macro_recall']):.2f}%")
    print(f"Macro F1-score:  {100 * float(final_result['macro_f1']):.2f}%")
    print(f"Correct predictions: {int((predictions == labels).sum())} / {len(labels)}")
    print(f"Peak GPU memory: {peak_vram_mb:.1f} MB")
    print("Model saved to:", MODEL_PATH)
    print("Results saved to:", RESULT_FOLDER)
    print("\nWAVLM TRAINING: PASSED")
    print(
        "Protocol note: report this only as a paper-compatible augmented-"
        "overlap result; the clean AST result remains separate."
    )


if __name__ == "__main__":
    main()
