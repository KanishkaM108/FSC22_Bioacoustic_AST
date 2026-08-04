"""Fine-tune a pretrained Audio Spectrogram Transformer on FSC22."""

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
from transformers import ASTForAudioClassification


MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
SEED = 42
NUM_CLASSES = 27
EPOCHS = 25
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 7
UNFROZEN_TRANSFORMER_BLOCKS = 2

ROOT = Path.cwd()
SPLIT_FILE = ROOT / "outputs" / "paper_split_seed42.csv"
FEATURE_FOLDER = ROOT / "outputs" / "ast_cache"
MODEL_FOLDER = ROOT / "models"
RESULT_FOLDER = ROOT / "outputs" / "ast"

MODEL_FOLDER.mkdir(parents=True, exist_ok=True)
RESULT_FOLDER.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FSC22ASTDataset(Dataset):
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
    def _spec_augment(features: torch.Tensor) -> torch.Tensor:
        # AST feature shape: [time frames, mel bins]
        if random.random() < 0.50:
            width = random.randint(12, 40)
            start = random.randint(0, features.shape[0] - width)
            features[start : start + width, :] = 0.0

        if random.random() < 0.50:
            width = random.randint(4, 12)
            start = random.randint(0, features.shape[1] - width)
            features[:, start : start + width] = 0.0

        return features

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.dataframe.iloc[index]
        feature_name = Path(str(row["Dataset File Name"])).stem + ".npy"
        feature_path = FEATURE_FOLDER / feature_name

        features = np.load(feature_path).astype(np.float32)
        features = torch.from_numpy(features)

        if self.training:
            features = self._spec_augment(features)

        class_id = int(row["Class ID"])
        label = torch.tensor(self.class_to_index[class_id], dtype=torch.long)
        return features, label


def freeze_for_memory_safe_finetuning(
    model: ASTForAudioClassification,
) -> tuple[int, int]:
    for parameter in model.parameters():
        parameter.requires_grad = False

    for parameter in model.classifier.parameters():
        parameter.requires_grad = True

    ast_backbone = model.audio_spectrogram_transformer

    # Transformers 5.x exposes blocks as ``ASTModel.layers``. Older releases
    # used ``ASTModel.encoder.layer``. Supporting both keeps this script
    # compatible across environments.
    if hasattr(ast_backbone, "layers"):
        transformer_blocks = ast_backbone.layers
    elif hasattr(ast_backbone, "encoder") and hasattr(
        ast_backbone.encoder, "layer"
    ):
        transformer_blocks = ast_backbone.encoder.layer
    else:
        raise AttributeError(
            "Could not locate the AST transformer blocks in this "
            "Transformers version."
        )

    for transformer_block in transformer_blocks[-UNFROZEN_TRANSFORMER_BLOCKS:]:
        for parameter in transformer_block.parameters():
            parameter.requires_grad = True

    for parameter in ast_backbone.layernorm.parameters():
        parameter.requires_grad = True

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return total_parameters, trainable_parameters


def apply_mixup(
    features: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if random.random() >= 0.50:
        return features, labels, labels, 1.0

    mixing_weight = float(np.random.beta(alpha, alpha))
    permutation = torch.randperm(features.size(0), device=features.device)
    mixed_features = (
        mixing_weight * features
        + (1.0 - mixing_weight) * features[permutation]
    )
    return mixed_features, labels, labels[permutation], mixing_weight


@torch.inference_mode()
def evaluate(
    model: ASTForAudioClassification,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    amp_enabled: bool,
) -> dict[str, object]:
    model.eval()
    total_loss = 0.0
    true_labels: list[int] = []
    predictions: list[int] = []

    for features, labels in loader:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            logits = model(input_values=features).logits
            loss = loss_function(logits, labels)

        total_loss += loss.item() * labels.size(0)
        predicted_labels = logits.argmax(dim=1)
        true_labels.extend(labels.cpu().tolist())
        predictions.extend(predicted_labels.cpu().tolist())

    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy_score(true_labels, predictions),
        "precision": precision_score(
            true_labels, predictions, average="macro", zero_division=0
        ),
        "recall": recall_score(
            true_labels, predictions, average="macro", zero_division=0
        ),
        "f1": f1_score(
            true_labels, predictions, average="macro", zero_division=0
        ),
        "true_labels": true_labels,
        "predictions": predictions,
    }


def save_results(
    history: pd.DataFrame,
    true_labels: list[int],
    predictions: list[int],
    class_names: list[str],
) -> None:
    sns.set_theme(style="whitegrid")

    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(history["epoch"], history["train_loss"], label="Training")
    axes[0].plot(history["epoch"], history["validation_loss"], label="Validation")
    axes[0].set_title("AST Fine-Tuning Loss")
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
    axes[1].axhline(95.00, color="green", linestyle=":", label="Target")
    axes[1].set_title("AST Validation Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score (%)")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(RESULT_FOLDER / "ast_training_curves.png", dpi=300)
    plt.close(figure)

    matrix = confusion_matrix(true_labels, predictions)
    figure = plt.figure(figsize=(18, 15))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Purples",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False,
    )
    plt.title("FSC22 Audio Spectrogram Transformer Confusion Matrix")
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    figure.savefig(RESULT_FOLDER / "ast_confusion_matrix.png", dpi=300)
    plt.close(figure)


def main() -> None:
    set_seed(SEED)
    torch.set_float32_matmul_precision("high")

    if not SPLIT_FILE.exists():
        raise FileNotFoundError(f"Split file not found: {SPLIT_FILE}")
    if len(list(FEATURE_FOLDER.glob("*.npy"))) != 2025:
        raise FileNotFoundError("Expected 2025 cached AST feature files.")

    metadata = pd.read_csv(SPLIT_FILE)
    metadata["Class ID"] = pd.to_numeric(metadata["Class ID"]).astype(int)

    class_table = (
        metadata[["Class ID", "Class Name"]]
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

    train_dataframe = metadata[metadata["Split"] == "train"].copy()
    validation_dataframe = metadata[metadata["Split"] == "test"].copy()

    train_dataset = FSC22ASTDataset(
        train_dataframe, class_to_index, training=True
    )
    validation_dataset = FSC22ASTDataset(
        validation_dataframe, class_to_index, training=False
    )

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    amp_enabled = use_cuda

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=use_cuda,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=use_cuda,
    )

    model = ASTForAudioClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_CLASSES,
        id2label=id_to_label,
        label2id=label_to_id,
        ignore_mismatched_sizes=True,
        local_files_only=True,
    )
    total_parameters, trainable_parameters = freeze_for_memory_safe_finetuning(
        model
    )
    model.to(device)

    loss_function = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=2e-7,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    model_path = MODEL_FOLDER / "fsc22_ast_seed42.pt"
    history_rows: list[dict[str, float | int]] = []
    best_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    if use_cuda:
        torch.cuda.reset_peak_memory_stats()

    print("\nFSC22 AST FINE-TUNING")
    print("---------------------")
    print("Device:", device)
    if use_cuda:
        print("GPU:", torch.cuda.get_device_name(0))
    print("Total parameters:", f"{total_parameters:,}")
    print("Trainable parameters:", f"{trainable_parameters:,}")
    print("Training samples:", len(train_dataset))
    print("Validation samples:", len(validation_dataset))
    print("Maximum epochs:", EPOCHS)
    print(
        "Effective batch size:",
        BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
    )
    print()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_training_loss = 0.0

        for step, (features, labels) in enumerate(train_loader, start=1):
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            features, labels_a, labels_b, mixing_weight = apply_mixup(
                features, labels
            )

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                logits = model(input_values=features).logits
                loss = (
                    mixing_weight * loss_function(logits, labels_a)
                    + (1.0 - mixing_weight) * loss_function(logits, labels_b)
                )
                scaled_loss = loss / GRADIENT_ACCUMULATION_STEPS

            scaler.scale(scaled_loss).backward()
            total_training_loss += loss.item() * labels.size(0)

            should_step = (
                step % GRADIENT_ACCUMULATION_STEPS == 0
                or step == len(train_loader)
            )
            if should_step:
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
            amp_enabled,
        )
        current_lr = optimizer.param_groups[0]["lr"]

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": training_loss,
                "validation_loss": float(validation["loss"]),
                "validation_accuracy": float(validation["accuracy"]),
                "validation_precision": float(validation["precision"]),
                "validation_recall": float(validation["recall"]),
                "validation_f1": float(validation["f1"]),
                "learning_rate": current_lr,
            }
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train loss {training_loss:.4f} | "
            f"Val loss {float(validation['loss']):.4f} | "
            f"Acc {100 * float(validation['accuracy']):.2f}% | "
            f"F1 {100 * float(validation['f1']):.2f}% | "
            f"LR {current_lr:.2e}"
        )

        if float(validation["f1"]) > best_f1 + 1e-4:
            best_f1 = float(validation["f1"])
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), model_path)
        else:
            epochs_without_improvement += 1

        scheduler.step()

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping after epoch {epoch}.")
            break

    history = pd.DataFrame(history_rows)
    history.to_csv(RESULT_FOLDER / "ast_history.csv", index=False)

    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    final_metrics = evaluate(
        model,
        validation_loader,
        loss_function,
        device,
        amp_enabled,
    )

    report = classification_report(
        final_metrics["true_labels"],
        final_metrics["predictions"],
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(
        RESULT_FOLDER / "ast_classification_report.csv"
    )

    peak_vram_mb = (
        torch.cuda.max_memory_allocated() / (1024**2) if use_cuda else 0.0
    )
    result = {
        "model": "Audio Spectrogram Transformer",
        "pretrained_checkpoint": MODEL_NAME,
        "seed": SEED,
        "best_epoch": best_epoch,
        "accuracy": float(final_metrics["accuracy"]),
        "macro_precision": float(final_metrics["precision"]),
        "macro_recall": float(final_metrics["recall"]),
        "macro_f1": float(final_metrics["f1"]),
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "peak_vram_mb": peak_vram_mb,
        "base_paper_accuracy": 0.9259,
    }
    with open(RESULT_FOLDER / "ast_metrics.json", "w", encoding="utf-8") as file:
        json.dump(result, file, indent=4)

    save_results(
        history,
        final_metrics["true_labels"],
        final_metrics["predictions"],
        class_names,
    )

    print("\nBEST AST RESULTS")
    print("----------------")
    print("Best epoch:", best_epoch)
    print(f"Accuracy:        {100 * float(final_metrics['accuracy']):.2f}%")
    print(f"Macro precision: {100 * float(final_metrics['precision']):.2f}%")
    print(f"Macro recall:    {100 * float(final_metrics['recall']):.2f}%")
    print(f"Macro F1-score:  {100 * float(final_metrics['f1']):.2f}%")
    print(f"Peak GPU memory: {peak_vram_mb:.1f} MB")
    print("Model saved to:", model_path)
    print("Results saved to:", RESULT_FOLDER)
    print("\nAST TRAINING: PASSED")


if __name__ == "__main__":
    main()
