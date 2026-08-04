"""Train a source-consistency-regularized AST on clean FSC22 source groups.

The script is sized for a 4 GB RTX 3050 Laptop GPU. It uses:

* six AST blocks are fine-tuned with differential learning rates;
* each minibatch contains two independently augmented views of each source;
* focal-smoothed cross entropy emphasizes hard examples without test-derived
  class weights;
* logit and embedding consistency losses encourage pitch-invariant decisions.

The existing checkpoints are never overwritten.  Train three diverse models::

    python src\train_ast_v2_source_consistent.py --seed 101
    python src\train_ast_v2_source_consistent.py --seed 202
    python src\train_ast_v2_source_consistent.py --seed 303

The manifest must be created by ``prepare_clean_grouped_protocol.py``. The
trainer reads only ``train`` and ``validation`` rows; it never loads the locked
``test`` partition.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
from sklearn.metrics import classification_report
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import ASTForAudioClassification

import train_ast_ensemble as ensemble


ROOT = Path.cwd()
DEFAULT_MANIFEST_FILE = ROOT / "outputs" / "clean_grouped_manifest_seed42.csv"
DEFAULT_TAG = "clean_ast_v1"
MODEL_FOLDER = ROOT / "models"
MODEL_FOLDER.mkdir(parents=True, exist_ok=True)

DEFAULT_EPOCHS = 30
DEFAULT_UNFROZEN_BLOCKS = 6
BATCH_SIZE = 4
SOURCES_PER_BATCH = 2
VIEWS_PER_SOURCE = 2
GRADIENT_ACCUMULATION_STEPS = 4
BACKBONE_LEARNING_RATE = 7.5e-6
HEAD_LEARNING_RATE = 7.5e-5
WEIGHT_DECAY = 2e-4
WARMUP_FRACTION = 0.08
EARLY_STOPPING_PATIENCE = 8
LABEL_SMOOTHING = 0.04
FOCAL_GAMMA = 1.25
LOGIT_CONSISTENCY_WEIGHT = 0.20
EMBEDDING_CONSISTENCY_WEIGHT = 0.05


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Source-consistency-regularized FSC22 AST fine-tuning"
    )
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_FILE,
        help="Leakage-free manifest created by prepare_clean_grouped_protocol.py",
    )
    parser.add_argument(
        "--tag",
        default=DEFAULT_TAG,
        help="Safe checkpoint/output identifier",
    )
    parser.add_argument(
        "--unfrozen-blocks",
        type=int,
        default=DEFAULT_UNFROZEN_BLOCKS,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing v2 checkpoint for this seed",
    )
    return parser.parse_args()


def validate_tag(tag: str) -> str:
    safe = "".join(
        character
        for character in tag
        if character.isalnum() or character in "-_"
    )
    if not safe or safe != tag:
        raise ValueError("--tag may contain only letters, numbers, '-' and '_'.")
    return safe


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SourceAwareFeatureDataset(Dataset):
    """Return cached AST features with independent stochastic views."""

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
    def _shift_time(features: torch.Tensor, shift: int) -> torch.Tensor:
        if shift == 0:
            return features
        shifted = torch.roll(features, shifts=shift, dims=0)
        if shift > 0:
            shifted[:shift, :] = 0.0
        else:
            shifted[shift:, :] = 0.0
        return shifted

    @classmethod
    def _augment(cls, features: torch.Tensor) -> torch.Tensor:
        output = features.clone()

        if random.random() < 0.40:
            output = cls._shift_time(output, random.randint(-16, 16))

        if random.random() < 0.55:
            width = random.randint(8, min(36, output.shape[0]))
            start = random.randint(0, output.shape[0] - width)
            output[start : start + width, :] = 0.0

        if random.random() < 0.45:
            width = random.randint(2, min(10, output.shape[1]))
            start = random.randint(0, output.shape[1] - width)
            output[:, start : start + width] = 0.0

        if random.random() < 0.20:
            output = output * random.uniform(0.97, 1.03)

        if random.random() < 0.20:
            output = output + torch.randn_like(output) * 0.01

        return output

    def __getitem__(self, index: int):
        row = self.dataframe.iloc[index]
        features = torch.from_numpy(
            np.load(ROOT / str(row["Feature Path"])).astype(np.float32)
        )
        if self.training:
            features = self._augment(features)
        label = self.class_to_index[int(row["Class ID"])]
        return (
            features,
            torch.tensor(label, dtype=torch.long),
            str(row["Original ID"]),
        )


class SourcePairBatchSampler(Sampler[list[int]]):
    """Build batches as two views from each of two original recordings."""

    def __init__(self, dataframe: pd.DataFrame, seed: int) -> None:
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, original_id in enumerate(dataframe["Original ID"].astype(str)):
            grouped[original_id].append(index)
        self.grouped = dict(grouped)
        self.source_ids = sorted(self.grouped)
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.source_ids) // SOURCES_PER_BATCH

    def __iter__(self) -> Iterator[list[int]]:
        generator = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        source_order = generator.permutation(self.source_ids)

        usable = len(source_order) - (len(source_order) % SOURCES_PER_BATCH)
        for start in range(0, usable, SOURCES_PER_BATCH):
            batch: list[int] = []
            for source_id in source_order[start : start + SOURCES_PER_BATCH]:
                candidates = self.grouped[str(source_id)]
                if len(candidates) >= VIEWS_PER_SOURCE:
                    selected = generator.choice(
                        candidates,
                        size=VIEWS_PER_SOURCE,
                        replace=False,
                    ).tolist()
                else:
                    # Two stochastic augmentations of the only available row.
                    selected = [candidates[0]] * VIEWS_PER_SOURCE
                batch.extend(int(index) for index in selected)
            generator.shuffle(batch)
            yield batch


def get_transformer_blocks(model: ASTForAudioClassification):
    backbone = model.audio_spectrogram_transformer
    if hasattr(backbone, "layers"):
        return backbone.layers
    if hasattr(backbone, "encoder") and hasattr(backbone.encoder, "layer"):
        return backbone.encoder.layer
    raise AttributeError("Could not locate the AST transformer blocks.")


def configure_trainable_parameters(
    model: ASTForAudioClassification,
    unfrozen_blocks: int,
) -> tuple[list[nn.Parameter], list[nn.Parameter], int, int]:
    transformer_blocks = get_transformer_blocks(model)
    if unfrozen_blocks < 1 or unfrozen_blocks > len(transformer_blocks):
        raise ValueError(
            f"--unfrozen-blocks must be between 1 and {len(transformer_blocks)}."
        )

    for parameter in model.parameters():
        parameter.requires_grad = False

    backbone = model.audio_spectrogram_transformer
    backbone_parameters: list[nn.Parameter] = []
    for block in transformer_blocks[-unfrozen_blocks:]:
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


def focal_smoothed_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    per_example_ce = functional.cross_entropy(
        logits,
        labels,
        reduction="none",
        label_smoothing=LABEL_SMOOTHING,
    )
    correct_probability = functional.softmax(logits, dim=1).gather(
        1, labels[:, None]
    ).squeeze(1)
    focal_weight = (1.0 - correct_probability).pow(FOCAL_GAMMA)
    focal_term = (focal_weight * per_example_ce).mean()
    return 0.50 * per_example_ce.mean() + 0.50 * focal_term


def source_consistency_loss(
    logits: torch.Tensor,
    embeddings: torch.Tensor,
    source_ids: list[str] | tuple[str, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Average consistency over all within-batch views of the same source."""
    probability = functional.softmax(logits.float(), dim=1)
    normalized_embedding = functional.normalize(embeddings.float(), dim=1)
    js_terms: list[torch.Tensor] = []
    cosine_terms: list[torch.Tensor] = []

    source_array = np.asarray(source_ids, dtype=str)
    for source_id in np.unique(source_array):
        indices = np.flatnonzero(source_array == source_id)
        if len(indices) < 2:
            continue
        first = int(indices[0])
        for second_value in indices[1:]:
            second = int(second_value)
            mean_probability = 0.5 * (
                probability[first] + probability[second]
            )
            log_mean = mean_probability.clamp_min(1e-7).log()
            js = 0.5 * (
                functional.kl_div(
                    log_mean,
                    probability[first],
                    reduction="sum",
                )
                + functional.kl_div(
                    log_mean,
                    probability[second],
                    reduction="sum",
                )
            )
            cosine = 1.0 - (
                normalized_embedding[first] * normalized_embedding[second]
            ).sum()
            js_terms.append(js)
            cosine_terms.append(cosine)

    if not js_terms:
        zero = logits.sum() * 0.0
        return zero, zero
    return torch.stack(js_terms).mean(), torch.stack(cosine_terms).mean()


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = max(1, int(total_steps * WARMUP_FRACTION))

    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return max(1e-3, step / warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def evaluate_with_tta(
    model: ASTForAudioClassification,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    labels_parts: list[np.ndarray] = []
    logits_parts: list[np.ndarray] = []
    source_id_parts: list[str] = []
    total_loss = 0.0
    with torch.inference_mode():
        for features, labels, source_ids in loader:
            features = features.to(device, non_blocking=True)
            labels_device = labels.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = model(input_values=features).logits
                loss = functional.cross_entropy(logits, labels_device)
            total_loss += loss.item() * labels.size(0)
            labels_parts.append(labels.numpy())
            logits_parts.append(logits.float().cpu().numpy())
            source_id_parts.extend(str(value) for value in source_ids)

    row_labels = np.concatenate(labels_parts)
    row_logits = np.concatenate(logits_parts)
    row_probabilities = ensemble.stable_softmax(row_logits)
    source_ids = np.asarray(source_id_parts, dtype=str)

    labels: list[int] = []
    probabilities: list[np.ndarray] = []
    unique_source_ids: list[str] = []
    for source_id in dict.fromkeys(source_ids.tolist()):
        indices = np.flatnonzero(source_ids == source_id)
        source_labels = np.unique(row_labels[indices])
        if len(source_labels) != 1:
            raise RuntimeError(f"Conflicting labels for Original ID {source_id}.")
        labels.append(int(source_labels[0]))
        probabilities.append(row_probabilities[indices].mean(axis=0))
        unique_source_ids.append(source_id)

    all_labels = np.asarray(labels, dtype=np.int64)
    all_probabilities = np.stack(probabilities).astype(np.float32)
    predictions = all_probabilities.argmax(axis=1)
    metrics = ensemble.compute_metrics(all_labels, predictions)
    return {
        **metrics,
        "loss": total_loss / len(loader.dataset),
        "labels": all_labels,
        "probabilities": all_probabilities,
        "predictions": predictions,
        "original_ids": np.asarray(unique_source_ids, dtype=str),
    }


def main() -> None:
    arguments = parse_arguments()
    if arguments.epochs < 1:
        raise ValueError("--epochs must be positive.")
    set_seed(arguments.seed)
    torch.set_float32_matmul_precision("high")
    tag = validate_tag(arguments.tag)

    manifest_file = arguments.manifest
    if not manifest_file.is_absolute():
        manifest_file = ROOT / manifest_file
    if not manifest_file.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_file}")
    manifest = pd.read_csv(manifest_file)
    manifest["Class ID"] = pd.to_numeric(manifest["Class ID"]).astype(int)
    required_columns = {
        "Feature Path",
        "Original ID",
        "Split",
        "Class ID",
        "Class Name",
        "Source Group ID",
    }
    missing_columns = required_columns.difference(manifest.columns)
    if missing_columns:
        raise KeyError(f"Manifest is missing columns: {sorted(missing_columns)}")
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

    training_dataframe = (
        manifest[manifest["Split"] == "train"].copy().reset_index(drop=True)
    )
    development_dataframe = (
        manifest[manifest["Split"] == "validation"].copy().reset_index(drop=True)
    )
    if training_dataframe.empty or development_dataframe.empty:
        raise RuntimeError("Manifest must contain train and validation rows.")
    training_groups = set(training_dataframe["Source Group ID"].astype(str))
    development_groups = set(
        development_dataframe["Source Group ID"].astype(str)
    )
    if training_groups & development_groups:
        raise RuntimeError("Source Group ID leakage between train and validation.")
    training_dataset = SourceAwareFeatureDataset(
        training_dataframe,
        class_to_index,
        training=True,
    )
    development_dataset = SourceAwareFeatureDataset(
        development_dataframe,
        class_to_index,
        training=False,
    )
    training_sampler = SourcePairBatchSampler(training_dataframe, arguments.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training_loader = DataLoader(
        training_dataset,
        batch_sampler=training_sampler,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    development_loader = DataLoader(
        development_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = ASTForAudioClassification.from_pretrained(
        ensemble.MODEL_NAME,
        num_labels=ensemble.NUM_CLASSES,
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
    ) = configure_trainable_parameters(model, arguments.unfrozen_blocks)
    model.to(device)

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": BACKBONE_LEARNING_RATE},
            {"params": head_parameters, "lr": HEAD_LEARNING_RATE},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    optimizer_steps_per_epoch = math.ceil(
        len(training_loader) / GRADIENT_ACCUMULATION_STEPS
    )
    scheduler = build_scheduler(
        optimizer,
        max(1, optimizer_steps_per_epoch * arguments.epochs),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    model_path = MODEL_FOLDER / f"fsc22_{tag}_seed{arguments.seed}.pt"
    result_folder = ROOT / "outputs" / tag / f"seed{arguments.seed}"
    result_folder.mkdir(parents=True, exist_ok=True)
    completion_path = result_folder / "training_complete.json"
    if (model_path.exists() or completion_path.exists()) and not arguments.force:
        raise FileExistsError(
            f"Seed {arguments.seed} already exists. Use --force only if you "
            "intend to replace its v2 checkpoint."
        )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    print("\nFSC22 CLEAN SOURCE-GROUPED AST TRAINING")
    print("-----------------------------------------")
    print("Seed:", arguments.seed)
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    print("Training rows available:", len(training_dataset))
    print("Source-paired rows sampled per epoch:", len(training_loader) * BATCH_SIZE)
    print("Validation feature rows:", len(development_dataset))
    print("Validation original clips:", development_dataframe["Original ID"].nunique())
    print("Train/validation source overlap: 0")
    print("Unfrozen transformer blocks:", arguments.unfrozen_blocks)
    print("Trainable parameters:", f"{trainable_parameters:,}")
    print("Effective batch size:", BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)
    print("Maximum epochs:", arguments.epochs)
    print(
        "Protocol note: checkpoint selection uses validation source groups only; "
        "the locked test split is never loaded by this trainer."
    )

    history_rows: list[dict[str, float | int]] = []
    best_macro_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, arguments.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_classification = 0.0
        total_logit_consistency = 0.0
        total_embedding_consistency = 0.0
        rows_seen = 0

        for step, (features, labels, source_ids) in enumerate(
            training_loader,
            start=1,
        ):
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                backbone_output = model.audio_spectrogram_transformer(
                    input_values=features
                )
                embeddings = backbone_output.pooler_output
                logits = model.classifier(embeddings)
                classification = focal_smoothed_loss(logits, labels)
                logit_consistency, embedding_consistency = source_consistency_loss(
                    logits,
                    embeddings,
                    source_ids,
                )
                loss = (
                    classification
                    + LOGIT_CONSISTENCY_WEIGHT * logit_consistency
                    + EMBEDDING_CONSISTENCY_WEIGHT * embedding_consistency
                )
                scaled_loss = loss / GRADIENT_ACCUMULATION_STEPS

            scaler.scale(scaled_loss).backward()
            batch_rows = labels.size(0)
            rows_seen += batch_rows
            total_loss += loss.item() * batch_rows
            total_classification += classification.item() * batch_rows
            total_logit_consistency += logit_consistency.item() * batch_rows
            total_embedding_consistency += embedding_consistency.item() * batch_rows

            should_step = (
                step % GRADIENT_ACCUMULATION_STEPS == 0
                or step == len(training_loader)
            )
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    (parameter for parameter in model.parameters() if parameter.requires_grad),
                    max_norm=1.0,
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

        development = evaluate_with_tta(
            model,
            development_loader,
            device,
        )
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / rows_seen,
                "train_classification_loss": total_classification / rows_seen,
                "train_logit_consistency_loss": total_logit_consistency / rows_seen,
                "train_embedding_consistency_loss": (
                    total_embedding_consistency / rows_seen
                ),
                "development_loss": float(development["loss"]),
                "development_accuracy": float(development["accuracy"]),
                "development_macro_precision": float(
                    development["macro_precision"]
                ),
                "development_macro_recall": float(development["macro_recall"]),
                "development_macro_f1": float(development["macro_f1"]),
                "backbone_learning_rate": optimizer.param_groups[0]["lr"],
                "head_learning_rate": optimizer.param_groups[1]["lr"],
            }
        )
        print(
            f"Epoch {epoch:02d}/{arguments.epochs} | "
            f"Train {total_loss / rows_seen:.4f} | "
            f"Dev loss {float(development['loss']):.4f} | "
            f"Acc {100 * float(development['accuracy']):.2f}% | "
            f"F1 {100 * float(development['macro_f1']):.2f}% | "
            f"LR {optimizer.param_groups[0]['lr']:.2e}"
        )

        if float(development["macro_f1"]) > best_macro_f1 + 1e-4:
            best_macro_f1 = float(development["macro_f1"])
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), model_path)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping after epoch {epoch}.")
            break

    history = pd.DataFrame(history_rows)
    history.to_csv(result_folder / "history.csv", index=False)

    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    final = evaluate_with_tta(model, development_loader, device)
    report = classification_report(
        final["labels"],
        final["predictions"],
        labels=np.arange(len(class_names)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(
        result_folder / "classification_report.csv"
    )
    np.savez_compressed(
        result_folder / "validation_probabilities_labels.npz",
        probabilities=np.asarray(final["probabilities"], dtype=np.float32),
        labels=np.asarray(final["labels"], dtype=np.int64),
        original_ids=np.asarray(final["original_ids"], dtype=str),
    )

    peak_vram_mb = (
        torch.cuda.max_memory_allocated() / (1024**2)
        if device.type == "cuda"
        else 0.0
    )
    completion = {
        "model": "Clean source-grouped source-consistency-regularized AST",
        "seed": arguments.seed,
        "best_epoch": best_epoch,
        "accuracy": float(final["accuracy"]),
        "macro_precision": float(final["macro_precision"]),
        "macro_recall": float(final["macro_recall"]),
        "macro_f1": float(final["macro_f1"]),
        "unfrozen_transformer_blocks": arguments.unfrozen_blocks,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "peak_vram_mb": peak_vram_mb,
        "model_path": str(model_path),
        "manifest": str(manifest_file),
        "protocol_note": (
            "Metrics are validation-only. Source groups are disjoint from "
            "training, and the locked test split was not loaded."
        ),
    }
    with open(completion_path, "w", encoding="utf-8") as file:
        json.dump(completion, file, indent=4)

    print("\nBEST CLEAN AST VALIDATION RESULTS")
    print("---------------------------------")
    print("Best epoch:", best_epoch)
    print(f"Accuracy:        {100 * float(final['accuracy']):.2f}%")
    print(f"Macro precision: {100 * float(final['macro_precision']):.2f}%")
    print(f"Macro recall:    {100 * float(final['macro_recall']):.2f}%")
    print(f"Macro F1-score:  {100 * float(final['macro_f1']):.2f}%")
    print(f"Peak GPU memory: {peak_vram_mb:.1f} MB")
    print("Model saved to:", model_path)
    print("Results saved to:", result_folder)
    print("\nCLEAN AST TRAINING: PASSED")


if __name__ == "__main__":
    main()
