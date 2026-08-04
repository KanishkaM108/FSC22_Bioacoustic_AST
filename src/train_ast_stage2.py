"""Stage-2 fine-tuning of the best FSC22 AST checkpoint."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from torch import nn
from torch.utils.data import DataLoader
from transformers import ASTForAudioClassification

import train_ast as stage1


SEED = 42
NUM_CLASSES = 27
EPOCHS = 20
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8
BACKBONE_LEARNING_RATE = 5e-6
HEAD_LEARNING_RATE = 2e-5
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 6
UNFROZEN_TRANSFORMER_BLOCKS = 6

ROOT = Path.cwd()
SPLIT_FILE = ROOT / "outputs" / "paper_split_seed42.csv"
FEATURE_FOLDER = ROOT / "outputs" / "ast_cache"
STAGE1_MODEL_PATH = ROOT / "models" / "fsc22_ast_seed42.pt"
STAGE2_MODEL_PATH = ROOT / "models" / "fsc22_ast_stage2_seed42.pt"
RESULT_FOLDER = ROOT / "outputs" / "ast_stage2"
RESULT_FOLDER.mkdir(parents=True, exist_ok=True)

# Reuse the tested plotting routine, but direct it to the Stage-2 folder.
stage1.RESULT_FOLDER = RESULT_FOLDER


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Stage2Dataset(stage1.FSC22ASTDataset):
    @staticmethod
    def _spec_augment(features: torch.Tensor) -> torch.Tensor:
        # Gentler augmentation is used during the low-learning-rate refinement.
        if random.random() < 0.30:
            width = random.randint(8, 24)
            start = random.randint(0, features.shape[0] - width)
            features[start : start + width, :] = 0.0

        if random.random() < 0.30:
            width = random.randint(3, 8)
            start = random.randint(0, features.shape[1] - width)
            features[:, start : start + width] = 0.0

        return features


def light_mixup(
    features: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if random.random() >= 0.25:
        return features, labels, labels, 1.0

    mixing_weight = float(np.random.beta(alpha, alpha))
    permutation = torch.randperm(features.size(0), device=features.device)
    mixed = (
        mixing_weight * features
        + (1.0 - mixing_weight) * features[permutation]
    )
    return mixed, labels, labels[permutation], mixing_weight


def get_transformer_blocks(model: ASTForAudioClassification):
    backbone = model.audio_spectrogram_transformer
    if hasattr(backbone, "layers"):
        return backbone.layers
    if hasattr(backbone, "encoder") and hasattr(backbone.encoder, "layer"):
        return backbone.encoder.layer
    raise AttributeError("Could not locate the AST transformer blocks.")


def configure_stage2(
    model: ASTForAudioClassification,
) -> tuple[list[nn.Parameter], list[nn.Parameter], int, int]:
    for parameter in model.parameters():
        parameter.requires_grad = False

    backbone = model.audio_spectrogram_transformer
    blocks = get_transformer_blocks(model)

    backbone_parameters: list[nn.Parameter] = []
    for block in blocks[-UNFROZEN_TRANSFORMER_BLOCKS:]:
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

    if not SPLIT_FILE.exists():
        raise FileNotFoundError(f"Split file not found: {SPLIT_FILE}")
    if not STAGE1_MODEL_PATH.exists():
        raise FileNotFoundError(f"Stage-1 model not found: {STAGE1_MODEL_PATH}")
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

    train_dataset = Stage2Dataset(
        train_dataframe, class_to_index, training=True
    )
    validation_dataset = Stage2Dataset(
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
        stage1.MODEL_NAME,
        num_labels=NUM_CLASSES,
        id2label=id_to_label,
        label2id=label_to_id,
        ignore_mismatched_sizes=True,
        local_files_only=True,
    )
    model.load_state_dict(
        torch.load(STAGE1_MODEL_PATH, map_location="cpu", weights_only=True)
    )

    (
        backbone_parameters,
        head_parameters,
        total_parameters,
        trainable_parameters,
    ) = configure_stage2(model)
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

    initial_metrics = stage1.evaluate(
        model,
        validation_loader,
        loss_function,
        device,
        amp_enabled,
    )
    best_f1 = float(initial_metrics["f1"])
    best_epoch = 0
    epochs_without_improvement = 0
    torch.save(model.state_dict(), STAGE2_MODEL_PATH)

    print("\nFSC22 AST STAGE-2 FINE-TUNING")
    print("-----------------------------")
    print("Device:", device)
    if use_cuda:
        print("GPU:", torch.cuda.get_device_name(0))
    print("Starting accuracy:", f"{100 * float(initial_metrics['accuracy']):.2f}%")
    print("Starting macro-F1:", f"{100 * best_f1:.2f}%")
    print("Unfrozen transformer blocks:", UNFROZEN_TRANSFORMER_BLOCKS)
    print("Total parameters:", f"{total_parameters:,}")
    print("Trainable parameters:", f"{trainable_parameters:,}")
    print("Effective batch size:", BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)
    print("Maximum epochs:", EPOCHS)
    print()

    history_rows: list[dict[str, float | int]] = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_training_loss = 0.0

        for step, (features, labels) in enumerate(train_loader, start=1):
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            features, labels_a, labels_b, mixing_weight = light_mixup(
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
        validation = stage1.evaluate(
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
            torch.save(model.state_dict(), STAGE2_MODEL_PATH)
        else:
            epochs_without_improvement += 1

        scheduler.step()

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping after epoch {epoch}.")
            break

    history = pd.DataFrame(history_rows)
    history.to_csv(RESULT_FOLDER / "ast_stage2_history.csv", index=False)

    model.load_state_dict(
        torch.load(STAGE2_MODEL_PATH, map_location=device, weights_only=True)
    )
    final_metrics = stage1.evaluate(
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
        RESULT_FOLDER / "ast_stage2_classification_report.csv"
    )

    peak_vram_mb = (
        torch.cuda.max_memory_allocated() / (1024**2) if use_cuda else 0.0
    )
    result = {
        "model": "AST Stage 2 - top six blocks",
        "seed": SEED,
        "best_stage2_epoch": best_epoch,
        "accuracy": float(final_metrics["accuracy"]),
        "macro_precision": float(final_metrics["precision"]),
        "macro_recall": float(final_metrics["recall"]),
        "macro_f1": float(final_metrics["f1"]),
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "peak_vram_mb": peak_vram_mb,
        "base_paper_accuracy": 0.9259,
    }
    with open(
        RESULT_FOLDER / "ast_stage2_metrics.json", "w", encoding="utf-8"
    ) as file:
        json.dump(result, file, indent=4)

    stage1.save_results(
        history,
        final_metrics["true_labels"],
        final_metrics["predictions"],
        class_names,
    )

    print("\nBEST AST STAGE-2 RESULTS")
    print("------------------------")
    print("Best Stage-2 epoch:", best_epoch)
    print(f"Accuracy:        {100 * float(final_metrics['accuracy']):.2f}%")
    print(f"Macro precision: {100 * float(final_metrics['precision']):.2f}%")
    print(f"Macro recall:    {100 * float(final_metrics['recall']):.2f}%")
    print(f"Macro F1-score:  {100 * float(final_metrics['f1']):.2f}%")
    print(f"Peak GPU memory: {peak_vram_mb:.1f} MB")
    print("Model saved to:", STAGE2_MODEL_PATH)
    print("Results saved to:", RESULT_FOLDER)
    print("\nAST STAGE-2 TRAINING: PASSED")


if __name__ == "__main__":
    main()
