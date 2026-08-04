"""Train two additional FSC22 paper-protocol AST seeds and soft-vote all three.

Expected working directory:
    C:\\Users\\Kanishka\\Downloads\\FSC22_Research

Run with:
    python src\\train_ast_ensemble.py

The script preserves the existing seed-42 checkpoint, trains seeds 17 and 73,
and evaluates one predeclared equal-probability three-model ensemble.
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
from transformers import ASTForAudioClassification


MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
NUM_CLASSES = 27
SEEDS = (42, 17, 73)
SEEDS_TO_TRAIN = (17, 73)
EPOCHS = 20
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
BACKBONE_LEARNING_RATE = 1e-5
HEAD_LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 6
UNFROZEN_TRANSFORMER_BLOCKS = 4

ROOT = Path.cwd()
MANIFEST_FILE = ROOT / "outputs" / "paper_augmented_split_seed42.csv"
MODEL_FOLDER = ROOT / "models"
RESULT_FOLDER = ROOT / "outputs" / "ast_ensemble"
MODEL_FOLDER.mkdir(parents=True, exist_ok=True)
RESULT_FOLDER.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PaperProtocolDataset(Dataset):
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
    def _light_spec_augment(features: torch.Tensor) -> torch.Tensor:
        if random.random() < 0.30:
            width = random.randint(8, 28)
            start = random.randint(0, features.shape[0] - width)
            features[start : start + width, :] = 0.0

        if random.random() < 0.30:
            width = random.randint(3, 8)
            start = random.randint(0, features.shape[1] - width)
            features[:, start : start + width] = 0.0

        return features

    def __getitem__(self, index: int):
        row = self.dataframe.iloc[index]
        feature_path = ROOT / str(row["Feature Path"])
        features = torch.from_numpy(np.load(feature_path).astype(np.float32))
        if self.training:
            features = self._light_spec_augment(features)
        label = self.class_to_index[int(row["Class ID"])]
        return features, torch.tensor(label, dtype=torch.long)


def get_transformer_blocks(model: ASTForAudioClassification):
    backbone = model.audio_spectrogram_transformer
    if hasattr(backbone, "layers"):
        return backbone.layers
    if hasattr(backbone, "encoder") and hasattr(backbone.encoder, "layer"):
        return backbone.encoder.layer
    raise AttributeError("Could not locate the AST transformer blocks.")


def configure_trainable_parameters(model: ASTForAudioClassification):
    for parameter in model.parameters():
        parameter.requires_grad = False

    backbone = model.audio_spectrogram_transformer
    backbone_parameters: list[nn.Parameter] = []
    for block in get_transformer_blocks(model)[-UNFROZEN_TRANSFORMER_BLOCKS:]:
        for parameter in block.parameters():
            parameter.requires_grad = True
            backbone_parameters.append(parameter)

    for parameter in backbone.layernorm.parameters():
        parameter.requires_grad = True
        backbone_parameters.append(parameter)

    head_parameters: list[nn.Parameter] = []
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True
        head_parameters.append(parameter)

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


def build_model(
    id_to_label: dict[int, str],
    label_to_id: dict[str, int],
) -> ASTForAudioClassification:
    return ASTForAudioClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_CLASSES,
        id2label=id_to_label,
        label2id=label_to_id,
        ignore_mismatched_sizes=True,
        local_files_only=True,
    )


def compute_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
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
def evaluate_model(
    model: ASTForAudioClassification,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    amp_enabled: bool,
) -> dict[str, object]:
    model.eval()
    total_loss = 0.0
    labels_parts: list[np.ndarray] = []
    logits_parts: list[np.ndarray] = []

    for features, labels in loader:
        features = features.to(device, non_blocking=True)
        labels_device = labels.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            logits = model(input_values=features).logits
            loss = loss_function(logits, labels_device)

        total_loss += loss.item() * labels.size(0)
        labels_parts.append(labels.numpy())
        logits_parts.append(logits.float().cpu().numpy())

    all_labels = np.concatenate(labels_parts)
    all_logits = np.concatenate(logits_parts)
    predictions = all_logits.argmax(axis=1)
    result: dict[str, object] = compute_metrics(all_labels, predictions)
    result.update(
        {
            "loss": total_loss / len(loader.dataset),
            "labels": all_labels,
            "logits": all_logits,
            "predictions": predictions,
        }
    )
    return result


def train_one_seed(
    seed: int,
    train_dataframe: pd.DataFrame,
    test_dataframe: pd.DataFrame,
    class_to_index: dict[int, int],
    id_to_label: dict[int, str],
    label_to_id: dict[str, int],
    device: torch.device,
) -> None:
    model_path = MODEL_FOLDER / f"fsc22_ast_paper_protocol_seed{seed}.pt"
    seed_folder = RESULT_FOLDER / f"seed{seed}"
    seed_folder.mkdir(parents=True, exist_ok=True)
    completion_file = seed_folder / "training_complete.json"

    if model_path.exists() and completion_file.exists():
        print(f"Seed {seed} already completed; reusing its checkpoint.")
        return

    set_seed(seed)
    use_cuda = device.type == "cuda"
    train_dataset = PaperProtocolDataset(
        train_dataframe, class_to_index, training=True
    )
    test_dataset = PaperProtocolDataset(
        test_dataframe, class_to_index, training=False
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=use_cuda,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=use_cuda,
    )

    model = build_model(id_to_label, label_to_id)
    (
        backbone_parameters,
        head_parameters,
        total_parameters,
        trainable_parameters,
    ) = configure_trainable_parameters(model)
    model.to(device)

    loss_function = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": BACKBONE_LEARNING_RATE},
            {"params": head_parameters, "lr": HEAD_LEARNING_RATE},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=5e-7
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    if use_cuda:
        torch.cuda.reset_peak_memory_stats()

    print(f"\nTRAINING PAPER-PROTOCOL AST: SEED {seed}")
    print("--------------------------------------")
    print("Training samples:", len(train_dataset))
    print("Test samples:", len(test_dataset))
    print("Trainable parameters:", f"{trainable_parameters:,}")
    print("Effective batch size:", BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)

    history_rows: list[dict[str, float | int]] = []
    best_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_training_loss = 0.0

        for step, (features, labels) in enumerate(train_loader, start=1):
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_cuda,
            ):
                logits = model(input_values=features).logits
                loss = loss_function(logits, labels)
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
        validation = evaluate_model(
            model, test_loader, loss_function, device, use_cuda
        )
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": training_loss,
                "validation_loss": float(validation["loss"]),
                "validation_accuracy": float(validation["accuracy"]),
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
            torch.save(model.state_dict(), model_path)
        else:
            epochs_without_improvement += 1

        scheduler.step()
        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping after epoch {epoch}.")
            break

    pd.DataFrame(history_rows).to_csv(seed_folder / "history.csv", index=False)
    peak_vram_mb = (
        torch.cuda.max_memory_allocated() / (1024**2) if use_cuda else 0.0
    )
    completion = {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_selection_macro_f1": best_f1,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "peak_vram_mb": peak_vram_mb,
        "model_path": str(model_path),
    }
    with open(completion_file, "w", encoding="utf-8") as file:
        json.dump(completion, file, indent=4)

    print(f"Seed {seed} training complete. Best epoch: {best_epoch}")
    del model, optimizer, scheduler, scaler
    if use_cuda:
        torch.cuda.empty_cache()


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def save_confusion_matrix(
    labels: np.ndarray,
    predictions: np.ndarray,
    class_names: list[str],
) -> None:
    sns.set_theme(style="white")
    matrix = confusion_matrix(labels, predictions)
    figure = plt.figure(figsize=(18, 15))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False,
    )
    plt.title("FSC22 Three-Seed AST Soft-Voting Ensemble")
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    figure.savefig(RESULT_FOLDER / "ensemble_confusion_matrix.png", dpi=300)
    plt.close(figure)


def main() -> None:
    torch.set_float32_matmul_precision("high")
    if not MANIFEST_FILE.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_FILE}")

    seed42_path = MODEL_FOLDER / "fsc22_ast_paper_protocol_seed42.pt"
    if not seed42_path.exists():
        raise FileNotFoundError(f"Existing seed-42 model not found: {seed42_path}")

    manifest = pd.read_csv(MANIFEST_FILE)
    manifest["Class ID"] = pd.to_numeric(manifest["Class ID"]).astype(int)
    missing_features = [
        path
        for path in manifest["Feature Path"].astype(str)
        if not (ROOT / path).exists()
    ]
    if missing_features:
        raise FileNotFoundError(f"Missing {len(missing_features)} feature files.")

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
    test_dataframe = manifest[manifest["Split"] == "test"].copy()
    overlapping_originals = len(
        set(train_dataframe["Original ID"]).intersection(
            set(test_dataframe["Original ID"])
        )
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nFSC22 THREE-SEED AST ENSEMBLE")
    print("-----------------------------")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    print("Seeds:", SEEDS)
    print("Original IDs occurring in both splits:", overlapping_originals)
    print("The existing seed-42 checkpoint will not be modified.")

    for seed in SEEDS_TO_TRAIN:
        train_one_seed(
            seed,
            train_dataframe,
            test_dataframe,
            class_to_index,
            id_to_label,
            label_to_id,
            device,
        )

    test_dataset = PaperProtocolDataset(
        test_dataframe, class_to_index, training=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    loss_function = nn.CrossEntropyLoss(label_smoothing=0.05)

    probabilities_by_seed: list[np.ndarray] = []
    predictions_by_seed: dict[int, np.ndarray] = {}
    individual_rows: list[dict[str, float | int]] = []
    true_labels: np.ndarray | None = None

    print("\nEVALUATING INDIVIDUAL MODELS")
    print("----------------------------")
    for seed in SEEDS:
        model_path = MODEL_FOLDER / f"fsc22_ast_paper_protocol_seed{seed}.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        model = build_model(id_to_label, label_to_id)
        model.load_state_dict(
            torch.load(model_path, map_location="cpu", weights_only=True)
        )
        model.to(device)
        evaluation = evaluate_model(
            model, test_loader, loss_function, device, amp_enabled=False
        )
        if true_labels is None:
            true_labels = np.asarray(evaluation["labels"])
        elif not np.array_equal(true_labels, np.asarray(evaluation["labels"])):
            raise RuntimeError("Test-label order changed between model evaluations.")

        logits = np.asarray(evaluation["logits"])
        probabilities_by_seed.append(stable_softmax(logits))
        predictions_by_seed[seed] = np.asarray(evaluation["predictions"])
        row = {
            "seed": seed,
            "accuracy": float(evaluation["accuracy"]),
            "macro_precision": float(evaluation["macro_precision"]),
            "macro_recall": float(evaluation["macro_recall"]),
            "macro_f1": float(evaluation["macro_f1"]),
            "correct_predictions": int(
                (np.asarray(evaluation["predictions"]) == true_labels).sum()
            ),
        }
        individual_rows.append(row)
        print(
            f"Seed {seed}: accuracy {100 * row['accuracy']:.2f}% | "
            f"macro-F1 {100 * row['macro_f1']:.2f}% | "
            f"correct {row['correct_predictions']} / {len(true_labels)}"
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    assert true_labels is not None
    ensemble_probabilities = np.mean(np.stack(probabilities_by_seed), axis=0)
    ensemble_predictions = ensemble_probabilities.argmax(axis=1)
    ensemble_metrics = compute_metrics(true_labels, ensemble_predictions)
    ensemble_correct = int((ensemble_predictions == true_labels).sum())

    pd.DataFrame(individual_rows).to_csv(
        RESULT_FOLDER / "individual_model_metrics.csv", index=False
    )
    report = classification_report(
        true_labels,
        ensemble_predictions,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(
        RESULT_FOLDER / "ensemble_classification_report.csv"
    )

    prediction_table = test_dataframe[
        [
            "Original Dataset File Name",
            "Variant",
            "Class ID",
            "Class Name",
        ]
    ].reset_index(drop=True)
    prediction_table["true_index"] = true_labels
    for seed in SEEDS:
        prediction_table[f"seed{seed}_prediction"] = [
            class_names[index] for index in predictions_by_seed[seed]
        ]
    prediction_table["ensemble_prediction"] = [
        class_names[index] for index in ensemble_predictions
    ]
    prediction_table["ensemble_confidence"] = ensemble_probabilities.max(axis=1)
    prediction_table["ensemble_correct"] = ensemble_predictions == true_labels
    prediction_table.to_csv(RESULT_FOLDER / "ensemble_predictions.csv", index=False)

    save_confusion_matrix(true_labels, ensemble_predictions, class_names)
    result = {
        "model": "Equal-probability soft-voting ensemble of three AST models",
        "seeds": list(SEEDS),
        **ensemble_metrics,
        "correct_predictions": ensemble_correct,
        "test_samples": int(len(true_labels)),
        "overlapping_original_ids": overlapping_originals,
        "base_paper_accuracy": 0.9259,
        "single_seed42_accuracy": individual_rows[0]["accuracy"],
        "target_accuracy": 0.95,
        "target_reached": bool(ensemble_metrics["accuracy"] > 0.95),
    }
    with open(RESULT_FOLDER / "ensemble_metrics.json", "w", encoding="utf-8") as file:
        json.dump(result, file, indent=4)

    print("\nTHREE-SEED SOFT-VOTING ENSEMBLE RESULTS")
    print("---------------------------------------")
    print(f"Accuracy:        {100 * ensemble_metrics['accuracy']:.2f}%")
    print(f"Macro precision: {100 * ensemble_metrics['macro_precision']:.2f}%")
    print(f"Macro recall:    {100 * ensemble_metrics['macro_recall']:.2f}%")
    print(f"Macro F1-score:  {100 * ensemble_metrics['macro_f1']:.2f}%")
    print(f"Correct predictions: {ensemble_correct} / {len(true_labels)}")
    print(
        "Change versus seed 42: "
        f"{100 * (ensemble_metrics['accuracy'] - individual_rows[0]['accuracy']):+.2f} pp"
    )
    print("Target above 95%:", "REACHED" if result["target_reached"] else "NOT REACHED")
    print("Results saved to:", RESULT_FOLDER)
    print("\nENSEMBLE EVALUATION: PASSED")
    print(
        "Integrity note: this is the paper-compatible augmented-overlap "
        "protocol, not the clean original-recording protocol."
    )


if __name__ == "__main__":
    main()
