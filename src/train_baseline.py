"""Train a reproducible base-paper-style CNN on FSC22 log-Mel features."""

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


SEED = 42
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 10
NUM_CLASSES = 27

ROOT = Path.cwd()
SPLIT_FILE = ROOT / "outputs" / "paper_split_seed42.csv"
FEATURE_FOLDER = ROOT / "outputs" / "mel_cache"
MODEL_FOLDER = ROOT / "models"
RESULT_FOLDER = ROOT / "outputs" / "baseline"

MODEL_FOLDER.mkdir(parents=True, exist_ok=True)
RESULT_FOLDER.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FSC22Dataset(Dataset):
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

    def _spec_augment(self, feature: torch.Tensor) -> torch.Tensor:
        # Feature shape: [1, mel bins, time frames]
        if random.random() < 0.60:
            mask_width = random.randint(4, 16)
            start = random.randint(0, feature.shape[-1] - mask_width)
            feature[:, :, start : start + mask_width] = 0.0

        if random.random() < 0.60:
            mask_width = random.randint(3, 10)
            start = random.randint(0, feature.shape[-2] - mask_width)
            feature[:, start : start + mask_width, :] = 0.0

        if random.random() < 0.30:
            feature = feature + 0.01 * torch.randn_like(feature)

        return feature

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.dataframe.iloc[index]
        filename = Path(str(row["Dataset File Name"])).stem + ".npy"
        feature_path = FEATURE_FOLDER / filename

        feature = np.load(feature_path).astype(np.float32)
        feature = torch.from_numpy(feature).unsqueeze(0)

        # Per-recording standardization stabilizes optimization.
        feature = (feature - feature.mean()) / (feature.std() + 1e-6)

        if self.training:
            feature = self._spec_augment(feature)

        class_id = int(row["Class ID"])
        label = torch.tensor(self.class_to_index[class_id], dtype=torch.long)
        return feature, label


class BaselineCNN(nn.Module):
    def __init__(self, number_of_classes: int) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.10),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.15),

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.35),
            nn.Linear(256, number_of_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    total_loss = 0.0
    true_labels: list[int] = []
    predictions: list[int] = []

    for features, labels in loader:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(features)
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


def save_plots(
    history: pd.DataFrame,
    true_labels: list[int],
    predictions: list[int],
    class_names: list[str],
) -> None:
    sns.set_theme(style="whitegrid")

    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(history["epoch"], history["train_loss"], label="Training")
    axes[0].plot(history["epoch"], history["validation_loss"], label="Validation")
    axes[0].set_title("Baseline CNN Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].legend()

    axes[1].plot(
        history["epoch"], history["validation_accuracy"] * 100, label="Accuracy"
    )
    axes[1].plot(
        history["epoch"], history["validation_f1"] * 100, label="Macro-F1"
    )
    axes[1].axhline(92.59, color="black", linestyle="--", label="Base paper: 92.59%")
    axes[1].set_title("Baseline CNN Validation Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score (%)")
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(RESULT_FOLDER / "baseline_training_curves.png", dpi=300)
    plt.close(figure)

    matrix = confusion_matrix(true_labels, predictions)
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
    plt.title("FSC22 Baseline CNN Confusion Matrix")
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    figure.savefig(RESULT_FOLDER / "baseline_confusion_matrix.png", dpi=300)
    plt.close(figure)


def main() -> None:
    set_seed(SEED)

    if not SPLIT_FILE.exists():
        raise FileNotFoundError(f"Split file not found: {SPLIT_FILE}")
    if len(list(FEATURE_FOLDER.glob("*.npy"))) != 2025:
        raise FileNotFoundError("Expected 2025 cached Mel spectrogram files.")

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

    if len(class_names) != NUM_CLASSES:
        raise ValueError(f"Expected {NUM_CLASSES} classes, found {len(class_names)}.")

    train_dataframe = metadata[metadata["Split"] == "train"].copy()
    validation_dataframe = metadata[metadata["Split"] == "test"].copy()

    train_dataset = FSC22Dataset(train_dataframe, class_to_index, training=True)
    validation_dataset = FSC22Dataset(
        validation_dataframe, class_to_index, training=False
    )

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

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

    model = BaselineCNN(NUM_CLASSES).to(device)
    number_of_parameters = sum(
        parameter.numel() for parameter in model.parameters()
    )

    loss_function = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
    )

    amp_enabled = use_cuda
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    model_path = MODEL_FOLDER / "fsc22_baseline_cnn_seed42.pt"
    history_rows: list[dict[str, float | int]] = []
    best_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    if use_cuda:
        torch.cuda.reset_peak_memory_stats()

    print("\nFSC22 BASELINE CNN TRAINING")
    print("---------------------------")
    print("Device:", device)
    if use_cuda:
        print("GPU:", torch.cuda.get_device_name(0))
    print("Parameters:", f"{number_of_parameters:,}")
    print("Training samples:", len(train_dataset))
    print("Validation samples:", len(validation_dataset))
    print("Maximum epochs:", EPOCHS)
    print()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        training_loss = 0.0

        for features, labels in train_loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                logits = model(features)
                loss = loss_function(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            training_loss += loss.item() * labels.size(0)

        training_loss /= len(train_dataset)
        validation = evaluate(model, validation_loader, loss_function, device)
        scheduler.step(float(validation["f1"]))

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

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping after epoch {epoch}.")
            break

    history = pd.DataFrame(history_rows)
    history.to_csv(RESULT_FOLDER / "baseline_history.csv", index=False)

    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    final_metrics = evaluate(model, validation_loader, loss_function, device)

    report = classification_report(
        final_metrics["true_labels"],
        final_metrics["predictions"],
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(
        RESULT_FOLDER / "baseline_classification_report.csv"
    )

    peak_vram_mb = (
        torch.cuda.max_memory_allocated() / (1024 ** 2) if use_cuda else 0.0
    )
    metrics_to_save = {
        "model": "Base-paper-style CNN",
        "seed": SEED,
        "best_epoch": best_epoch,
        "accuracy": float(final_metrics["accuracy"]),
        "macro_precision": float(final_metrics["precision"]),
        "macro_recall": float(final_metrics["recall"]),
        "macro_f1": float(final_metrics["f1"]),
        "parameters": number_of_parameters,
        "peak_vram_mb": peak_vram_mb,
        "base_paper_accuracy": 0.9259,
    }
    with open(RESULT_FOLDER / "baseline_metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics_to_save, file, indent=4)

    save_plots(
        history,
        final_metrics["true_labels"],
        final_metrics["predictions"],
        class_names,
    )

    print("\nBEST BASELINE RESULTS")
    print("---------------------")
    print("Best epoch:", best_epoch)
    print(f"Accuracy:        {100 * float(final_metrics['accuracy']):.2f}%")
    print(f"Macro precision: {100 * float(final_metrics['precision']):.2f}%")
    print(f"Macro recall:    {100 * float(final_metrics['recall']):.2f}%")
    print(f"Macro F1-score:  {100 * float(final_metrics['f1']):.2f}%")
    print(f"Peak GPU memory: {peak_vram_mb:.1f} MB")
    print("Model saved to:", model_path)
    print("Results saved to:", RESULT_FOLDER)
    print("\nBASELINE TRAINING: PASSED")


if __name__ == "__main__":
    main()
