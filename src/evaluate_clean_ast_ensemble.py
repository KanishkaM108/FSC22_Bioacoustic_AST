"""Evaluate a predeclared AST ensemble on the locked, source-disjoint test set.

This script performs no fitting, calibration, threshold selection, class
weighting, or ensemble-weight selection.  It averages three model probability
vectors and then averages the three test-time-augmentation views belonging to
each unseen clip.  Test labels are accessed only after predictions are fixed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

import train_ast_ensemble as ensemble
import train_ast_v2_source_consistent as clean_train


ROOT = Path.cwd()
DEFAULT_MANIFEST = ROOT / "outputs" / "clean_grouped_manifest_seed42.csv"
DEFAULT_SEEDS = (101, 202, 303)
DEFAULT_CHECKPOINT_TEMPLATE = "models/fsc22_clean_ast_v1_seed{seed}.pt"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Locked FSC22 clean-test evaluation with AST TTA ensemble"
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--checkpoint-template",
        default=DEFAULT_CHECKPOINT_TEMPLATE,
    )
    parser.add_argument("--tag", default="clean_ast_v1")
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


@torch.inference_mode()
def infer_probabilities(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    probability_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    source_ids: list[str] = []
    for features, labels, original_ids in loader:
        features = features.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            logits = model(input_values=features).logits
        probability_parts.append(
            ensemble.stable_softmax(logits.float().cpu().numpy())
        )
        label_parts.append(labels.numpy())
        source_ids.extend(str(value) for value in original_ids)
    return (
        np.concatenate(probability_parts),
        np.concatenate(label_parts),
        np.asarray(source_ids, dtype=str),
    )


def aggregate_tta(
    row_probabilities: np.ndarray,
    row_labels: np.ndarray,
    original_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probabilities: list[np.ndarray] = []
    labels: list[int] = []
    ordered_ids: list[str] = []
    for original_id in dict.fromkeys(original_ids.tolist()):
        indices = np.flatnonzero(original_ids == original_id)
        unique_labels = np.unique(row_labels[indices])
        if len(unique_labels) != 1:
            raise RuntimeError(f"Conflicting labels for Original ID {original_id}.")
        probabilities.append(row_probabilities[indices].mean(axis=0))
        labels.append(int(unique_labels[0]))
        ordered_ids.append(original_id)
    return (
        np.stack(probabilities).astype(np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(ordered_ids, dtype=str),
    )


def save_confusion_matrix(
    labels: np.ndarray,
    predictions: np.ndarray,
    class_names: list[str],
    output: Path,
) -> None:
    matrix = confusion_matrix(
        labels,
        predictions,
        labels=np.arange(len(class_names)),
    )
    sns.set_theme(style="white")
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
    plt.title("FSC22 source-disjoint AST ensemble confusion matrix")
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    arguments = parse_arguments()
    tag = clean_train.validate_tag(arguments.tag)
    if "{seed}" not in arguments.checkpoint_template:
        raise ValueError("--checkpoint-template must contain {seed}.")
    manifest_file = arguments.manifest
    if not manifest_file.is_absolute():
        manifest_file = ROOT / manifest_file
    if not manifest_file.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_file}")

    manifest = pd.read_csv(manifest_file)
    required = {
        "Feature Path",
        "Original ID",
        "Source Group ID",
        "Original Dataset File Name",
        "Source File Name",
        "Variant",
        "Split",
        "Class ID",
        "Class Name",
    }
    missing = required.difference(manifest.columns)
    if missing:
        raise KeyError(f"Manifest is missing columns: {sorted(missing)}")
    manifest["Class ID"] = pd.to_numeric(manifest["Class ID"]).astype(int)

    split_groups = {
        split: set(
            manifest.loc[manifest["Split"] == split, "Source Group ID"].astype(str)
        )
        for split in ("train", "validation", "test")
    }
    overlap = (
        (split_groups["train"] & split_groups["test"])
        | (split_groups["validation"] & split_groups["test"])
    )
    if overlap:
        raise RuntimeError(
            f"Locked test has {len(overlap)} source groups present outside test."
        )

    test_dataframe = (
        manifest[manifest["Split"] == "test"].copy().reset_index(drop=True)
    )
    if test_dataframe.empty:
        raise RuntimeError("Manifest contains no locked test rows.")
    missing_features = [
        value
        for value in test_dataframe["Feature Path"].astype(str)
        if not (ROOT / value).exists()
    ]
    if missing_features:
        raise FileNotFoundError(
            f"Locked test is missing {len(missing_features)} feature files."
        )

    class_table = (
        manifest[["Class ID", "Class Name"]]
        .drop_duplicates()
        .sort_values("Class ID")
    )
    class_ids = class_table["Class ID"].tolist()
    class_names = class_table["Class Name"].astype(str).tolist()
    if len(class_names) != ensemble.NUM_CLASSES:
        raise RuntimeError(
            f"Expected {ensemble.NUM_CLASSES} classes, found {len(class_names)}."
        )
    class_to_index = {
        class_id: index for index, class_id in enumerate(class_ids)
    }
    id_to_label = {index: name for index, name in enumerate(class_names)}
    label_to_id = {name: index for index, name in id_to_label.items()}

    dataset = clean_train.SourceAwareFeatureDataset(
        test_dataframe,
        class_to_index,
        training=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(
        dataset,
        batch_size=arguments.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    print("\nFSC22 LOCKED SOURCE-DISJOINT TEST")
    print("---------------------------------")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    print("Seeds:", tuple(arguments.seeds))
    print("Test feature rows:", len(test_dataframe))
    print("Test original clips:", test_dataframe["Original ID"].nunique())
    print("Test source groups:", len(split_groups["test"]))
    print("Test source overlap: 0")

    seed_probabilities: list[np.ndarray] = []
    labels_reference: np.ndarray | None = None
    ids_reference: np.ndarray | None = None
    per_seed_metrics: dict[str, dict[str, float]] = {}
    for seed in arguments.seeds:
        checkpoint = ROOT / arguments.checkpoint_template.format(seed=seed)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        print(f"Inference for seed {seed}...")
        model = ensemble.build_model(id_to_label, label_to_id)
        model.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=True)
        )
        model.to(device)
        row_probability, row_labels, row_ids = infer_probabilities(
            model, loader, device
        )
        probability, labels, ordered_ids = aggregate_tta(
            row_probability, row_labels, row_ids
        )
        if labels_reference is None:
            labels_reference = labels
            ids_reference = ordered_ids
        elif not (
            np.array_equal(labels_reference, labels)
            and np.array_equal(ids_reference, ordered_ids)
        ):
            raise RuntimeError("Test order changed between seed evaluations.")
        seed_probabilities.append(probability)
        seed_predictions = probability.argmax(axis=1)
        per_seed_metrics[str(seed)] = ensemble.compute_metrics(
            labels, seed_predictions
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    assert labels_reference is not None and ids_reference is not None
    final_probabilities = np.stack(seed_probabilities).mean(axis=0)
    predictions = final_probabilities.argmax(axis=1)
    metrics = ensemble.compute_metrics(labels_reference, predictions)

    result_folder = ROOT / "outputs" / f"{tag}_locked_test"
    result_folder.mkdir(parents=True, exist_ok=True)
    report = classification_report(
        labels_reference,
        predictions,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(
        result_folder / "classification_report.csv"
    )

    clip_table = (
        test_dataframe.sort_values(["Original ID", "Variant"])
        .drop_duplicates("Original ID")
        .set_index("Original ID")
        .loc[ids_reference]
        .reset_index()
    )
    output_table = clip_table[
        [
            "Original ID",
            "Source Group ID",
            "Original Dataset File Name",
            "Source File Name",
            "Class ID",
            "Class Name",
        ]
    ].copy()
    output_table["prediction"] = [class_names[index] for index in predictions]
    output_table["correct"] = predictions == labels_reference
    output_table["tta_views"] = 3
    for class_index, class_name in enumerate(class_names):
        safe_name = "".join(
            character.lower() if character.isalnum() else "_"
            for character in class_name
        ).strip("_")
        output_table[f"probability_{safe_name}"] = final_probabilities[
            :, class_index
        ]
    output_table.to_csv(result_folder / "locked_test_predictions.csv", index=False)

    save_confusion_matrix(
        labels_reference,
        predictions,
        class_names,
        result_folder / "confusion_matrix.png",
    )
    np.savez_compressed(
        result_folder / "locked_test_probabilities_labels.npz",
        probabilities=final_probabilities.astype(np.float32),
        labels=labels_reference,
        predictions=predictions,
        original_ids=ids_reference,
    )

    summary = {
        "model": "Three-seed AST ensemble with unseen-clip TTA",
        "evaluation_type": "locked source-disjoint test",
        "manifest": str(manifest_file),
        "seeds": list(arguments.seeds),
        "checkpoint_template": arguments.checkpoint_template,
        "ensemble_rule": "equal probability mean fixed before test inference",
        "tta_rule": "equal probability mean over original and pitch +/-2 views",
        "test_original_clips": int(len(labels_reference)),
        "test_source_groups": int(len(split_groups["test"])),
        "test_source_overlap": 0,
        "correct_predictions": int((predictions == labels_reference).sum()),
        "per_seed_metrics": per_seed_metrics,
        **{key: float(value) for key, value in metrics.items()},
        "integrity_note": (
            "Test groups are absent from training and validation. No test label "
            "was used for checkpoint, calibration, threshold, or ensemble selection."
        ),
    }
    (result_folder / "metrics.json").write_text(
        json.dumps(summary, indent=4), encoding="utf-8"
    )

    print("\nLOCKED CLEAN-TEST RESULTS")
    print("-------------------------")
    print(f"Accuracy:        {100 * metrics['accuracy']:.2f}%")
    print(f"Macro precision: {100 * metrics['macro_precision']:.2f}%")
    print(f"Macro recall:    {100 * metrics['macro_recall']:.2f}%")
    print(f"Macro F1-score:  {100 * metrics['macro_f1']:.2f}%")
    print(
        "Correct predictions:",
        int((predictions == labels_reference).sum()),
        "/",
        len(labels_reference),
    )
    print("Results saved to:", result_folder)
    print("\nLOCKED CLEAN-TEST EVALUATION: PASSED")


if __name__ == "__main__":
    main()
