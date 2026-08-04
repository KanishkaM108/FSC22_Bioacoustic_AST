"""Train AST using FSC22's augmentation-before-split paper protocol."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import ASTForAudioClassification

import train_ast as common


SEED = 42
NUM_CLASSES = 27
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
MODEL_PATH = ROOT / "models" / "fsc22_ast_paper_protocol_seed42.pt"
RESULT_FOLDER = ROOT / "outputs" / "paper_protocol_ast"
RESULT_FOLDER.mkdir(parents=True, exist_ok=True)

# Direct the shared plotting function to this experiment's output folder.
common.RESULT_FOLDER = RESULT_FOLDER


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
        features = torch.from_numpy(
            np.load(feature_path).astype(np.float32)
        )

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


def configure_trainable_parameters(
    model: ASTForAudioClassification,
):
    for parameter in model.parameters():
        parameter.requires_grad = False

    backbone = model.audio_spectrogram_transformer
    transformer_blocks = get_transformer_blocks(model)
    backbone_parameters: list[nn.Parameter] = []

    for block in transformer_blocks[-UNFROZEN_TRANSFORMER_BLOCKS:]:
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


def main() -> None:
    set_seed(SEED)
    torch.set_float32_matmul_precision("high")

    if not MANIFEST_FILE.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_FILE}")

    manifest = pd.read_csv(MANIFEST_FILE)
    manifest["Class ID"] = pd.to_numeric(manifest["Class ID"]).astype(int)

    missing_features = [
        feature_path
        for feature_path in manifest["Feature Path"].astype(str)
        if not (ROOT / feature_path).exists()
    ]
    if missing_features:
        raise FileNotFoundError(
            f"Missing {len(missing_features)} augmented feature files."
        )

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

    train_originals = set(train_dataframe["Original ID"])
    validation_originals = set(validation_dataframe["Original ID"])
    overlapping_originals = len(train_originals.intersection(validation_originals))

    train_dataset = PaperProtocolDataset(
        train_dataframe, class_to_index, training=True
    )
    validation_dataset = PaperProtocolDataset(
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

    # Start fresh from the AudioSet checkpoint. Earlier FSC22 weights are not
    # loaded, preventing contamination from our clean-split experiment.
    model = ASTForAudioClassification.from_pretrained(
        common.MODEL_NAME,
        num_labels=NUM_CLASSES,
        id2label=id_to_label,
        label2id=label_to_id,
        ignore_mismatched_sizes=True,
        local_files_only=True,
    )
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
            {
                "params": backbone_parameters,
                "lr": BACKBONE_LEARNING_RATE,
            },
            {
                "params": head_parameters,
                "lr": HEAD_LEARNING_RATE,
            },
        ],
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=5e-7,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    if use_cuda:
        torch.cuda.reset_peak_memory_stats()

    print("\nFSC22 PAPER-PROTOCOL AST TRAINING")
    print("---------------------------------")
    print("Device:", device)
    if use_cuda:
        print("GPU:", torch.cuda.get_device_name(0))
    print("Training samples:", len(train_dataset))
    print("Validation samples:", len(validation_dataset))
    print("Original IDs in both splits:", overlapping_originals)
    print("Unfrozen transformer blocks:", UNFROZEN_TRANSFORMER_BLOCKS)
    print("Total parameters:", f"{total_parameters:,}")
    print("Trainable parameters:", f"{trainable_parameters:,}")
    print("Effective batch size:", BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)
    print("Maximum epochs:", EPOCHS)
    print()

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
                enabled=amp_enabled,
            ):
                logits = model(input_values=features).logits
                loss = loss_function(logits, labels)
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
        validation = common.evaluate(
            model,
            validation_loader,
            loss_function,
            device,
            amp_enabled,
        )

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": training_loss,
                "validation_loss": float(validation["loss"]),
                "validation_accuracy": float(validation["accuracy"]),
                "validation_precision": float(validation["precision"]),
                "validation_recall": float(validation["recall"]),
                "validation_f1": float(validation["f1"]),
                "backbone_learning_rate": optimizer.param_groups[0]["lr"],
                "head_learning_rate": optimizer.param_groups[1]["lr"],
            }
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train loss {training_loss:.4f} | "
            f"Val loss {float(validation['loss']):.4f} | "
            f"Acc {100 * float(validation['accuracy']):.2f}% | "
            f"F1 {100 * float(validation['f1']):.2f}% | "
            f"LR {optimizer.param_groups[0]['lr']:.2e}"
        )

        if float(validation["f1"]) > best_f1 + 1e-4:
            best_f1 = float(validation["f1"])
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
    history.to_csv(
        RESULT_FOLDER / "paper_protocol_ast_history.csv",
        index=False,
    )

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device, weights_only=True)
    )
    # Final metrics use float32 inference for stable, batch-independent results.
    final_metrics = common.evaluate(
        model,
        validation_loader,
        loss_function,
        device,
        False,
    )

    report = classification_report(
        final_metrics["true_labels"],
        final_metrics["predictions"],
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(
        RESULT_FOLDER / "paper_protocol_ast_classification_report.csv"
    )

    peak_vram_mb = (
        torch.cuda.max_memory_allocated() / (1024**2) if use_cuda else 0.0
    )
    metrics = {
        "model": "AST using augmentation-before-split protocol",
        "seed": SEED,
        "best_epoch": best_epoch,
        "accuracy": float(final_metrics["accuracy"]),
        "macro_precision": float(final_metrics["precision"]),
        "macro_recall": float(final_metrics["recall"]),
        "macro_f1": float(final_metrics["f1"]),
        "overlapping_original_ids": overlapping_originals,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "peak_vram_mb": peak_vram_mb,
        "base_paper_accuracy": 0.9259,
        "clean_protocol_ast_accuracy": 0.9012,
    }
    with open(
        RESULT_FOLDER / "paper_protocol_ast_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics, file, indent=4)

    common.save_results(
        history,
        final_metrics["true_labels"],
        final_metrics["predictions"],
        class_names,
    )

    print("\nBEST PAPER-PROTOCOL AST RESULTS")
    print("-------------------------------")
    print("Best epoch:", best_epoch)
    print(f"Accuracy:        {100 * float(final_metrics['accuracy']):.2f}%")
    print(f"Macro precision: {100 * float(final_metrics['precision']):.2f}%")
    print(f"Macro recall:    {100 * float(final_metrics['recall']):.2f}%")
    print(f"Macro F1-score:  {100 * float(final_metrics['f1']):.2f}%")
    print(f"Peak GPU memory: {peak_vram_mb:.1f} MB")
    print("Model saved to:", MODEL_PATH)
    print("Results saved to:", RESULT_FOLDER)
    print("\nPAPER-PROTOCOL AST TRAINING: PASSED")


if __name__ == "__main__":
    main()
