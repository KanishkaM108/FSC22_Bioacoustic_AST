r"""Evaluate a label-free source-consistent AST ensemble on FSC22.

This is an explicitly transductive evaluation for the paper-compatible
augmentation-before-split protocol.  It averages model probabilities over all
pitch variants sharing an ``Original ID``, including variants assigned to the
training partition.  Training labels are never read by the aggregation rule,
but the rule still exploits source overlap and therefore MUST NOT be presented
as unseen-recording performance.

The default command reuses the three existing paper-protocol checkpoints::

    python src\evaluate_source_consistent_ast_v2.py

To evaluate the stronger v2 checkpoints after training them::

    python src\evaluate_source_consistent_ast_v2.py ^
      --seeds 101 202 303 ^
      --checkpoint-template "models/fsc22_ast_v2_seed{seed}.pt" ^
      --tag ast_v2
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
from torch import nn
from torch.utils.data import DataLoader

import train_ast_ensemble as ensemble


ROOT = Path.cwd()
MANIFEST_FILE = ROOT / "outputs" / "paper_augmented_split_seed42.csv"
DEFAULT_SEEDS = (42, 17, 73)
DEFAULT_CHECKPOINT_TEMPLATE = (
    "models/fsc22_ast_paper_protocol_seed{seed}.pt"
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Label-free source-consistent FSC22 AST evaluation"
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help="Checkpoint seeds to average (default: 42 17 73)",
    )
    parser.add_argument(
        "--checkpoint-template",
        default=DEFAULT_CHECKPOINT_TEMPLATE,
        help="Path relative to project root; must contain {seed}",
    )
    parser.add_argument(
        "--tag",
        default="legacy_three_seed",
        help="Safe output-folder suffix",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Inference batch size (4 is safe on a 4 GB RTX 3050)",
    )
    return parser.parse_args()


def validate_tag(tag: str) -> str:
    safe = "".join(character for character in tag if character.isalnum() or character in "-_")
    if not safe or safe != tag:
        raise ValueError("--tag may contain only letters, numbers, '-' and '_'.")
    return safe


def save_confusion_matrix(
    labels: np.ndarray,
    predictions: np.ndarray,
    class_names: list[str],
    result_folder: Path,
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
        cmap="YlGnBu",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False,
    )
    plt.title("FSC22 Source-Consistent AST Ensemble (Transductive)")
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    figure.savefig(result_folder / "source_consistent_confusion_matrix.png", dpi=300)
    plt.close(figure)


def subset_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float]:
    if len(labels) == 0:
        return {"accuracy": float("nan"), "macro_f1": float("nan")}
    values = ensemble.compute_metrics(labels, predictions)
    return {
        "accuracy": float(values["accuracy"]),
        "macro_f1": float(values["macro_f1"]),
    }


def main() -> None:
    arguments = parse_arguments()
    tag = validate_tag(arguments.tag)
    if "{seed}" not in arguments.checkpoint_template:
        raise ValueError("--checkpoint-template must contain the text {seed}.")
    if not MANIFEST_FILE.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_FILE}")

    result_folder = ROOT / "outputs" / f"source_consistent_ast_{tag}"
    result_folder.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(MANIFEST_FILE)
    manifest["Class ID"] = pd.to_numeric(manifest["Class ID"]).astype(int)
    required_columns = {
        "Feature Path",
        "Original ID",
        "Original Dataset File Name",
        "Variant",
        "Split",
        "Class ID",
        "Class Name",
    }
    missing_columns = required_columns.difference(manifest.columns)
    if missing_columns:
        raise KeyError(f"Manifest is missing columns: {sorted(missing_columns)}")

    missing_features = [
        relative_path
        for relative_path in manifest["Feature Path"].astype(str)
        if not (ROOT / relative_path).exists()
    ]
    if missing_features:
        raise FileNotFoundError(
            f"Missing {len(missing_features)} cached AST feature files."
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

    dataset = ensemble.PaperProtocolDataset(
        manifest,
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
    loss_function = nn.CrossEntropyLoss(label_smoothing=0.05)

    print("\nFSC22 SOURCE-CONSISTENT AST V2")
    print("--------------------------------")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    print("Seeds:", tuple(arguments.seeds))
    print("Manifest rows:", len(manifest))
    print("Evaluation rows:", int((manifest["Split"] == "test").sum()))
    print("Aggregation: equal mean over all variants with the same Original ID")
    print("Labels are not used by the aggregation rule.")

    probabilities_by_seed: list[np.ndarray] = []
    labels_from_loader: np.ndarray | None = None
    for seed in arguments.seeds:
        checkpoint_path = ROOT / arguments.checkpoint_template.format(seed=seed)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print(f"Evaluating seed {seed} on all variants...")
        model = ensemble.build_model(id_to_label, label_to_id)
        model.load_state_dict(
            torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        )
        model.to(device)
        evaluation = ensemble.evaluate_model(
            model,
            loader,
            loss_function,
            device,
            amp_enabled=False,
        )
        current_labels = np.asarray(evaluation["labels"], dtype=np.int64)
        if labels_from_loader is None:
            labels_from_loader = current_labels
        elif not np.array_equal(labels_from_loader, current_labels):
            raise RuntimeError("Manifest order changed between checkpoint evaluations.")
        probabilities_by_seed.append(
            ensemble.stable_softmax(np.asarray(evaluation["logits"]))
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    assert labels_from_loader is not None
    row_probabilities = np.mean(np.stack(probabilities_by_seed), axis=0)

    # Prediction construction is label-free.  Every original recording has
    # exactly one class by dataset design, so all of its augmented variants are
    # assigned the same mean probability vector.
    source_probabilities = row_probabilities.copy()
    source_has_training_variant = np.zeros(len(manifest), dtype=bool)
    group_sizes = np.zeros(len(manifest), dtype=np.int64)
    for _, indices_object in manifest.groupby("Original ID", sort=False).groups.items():
        indices = np.asarray(list(indices_object), dtype=np.int64)
        source_probability = row_probabilities[indices].mean(axis=0)
        source_probabilities[indices] = source_probability
        has_training_variant = bool(
            (manifest.iloc[indices]["Split"].astype(str) == "train").any()
        )
        source_has_training_variant[indices] = has_training_variant
        group_sizes[indices] = len(indices)

    test_mask = manifest["Split"].astype(str).to_numpy() == "test"
    test_indices = np.flatnonzero(test_mask)
    true_labels = labels_from_loader[test_indices]
    row_predictions = row_probabilities[test_indices].argmax(axis=1)
    source_predictions = source_probabilities[test_indices].argmax(axis=1)
    seen_mask = source_has_training_variant[test_indices]

    row_metrics = ensemble.compute_metrics(true_labels, row_predictions)
    final_metrics = ensemble.compute_metrics(true_labels, source_predictions)
    seen_metrics = subset_metrics(
        true_labels[seen_mask], source_predictions[seen_mask]
    )
    unseen_metrics = subset_metrics(
        true_labels[~seen_mask], source_predictions[~seen_mask]
    )

    row_correct = row_predictions == true_labels
    source_correct = source_predictions == true_labels
    changed = source_predictions != row_predictions
    corrected = changed & source_correct
    newly_wrong = changed & row_correct

    report = classification_report(
        true_labels,
        source_predictions,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(
        result_folder / "source_consistent_classification_report.csv"
    )

    output_table = manifest.loc[
        test_mask,
        [
            "Original ID",
            "Original Dataset File Name",
            "Variant",
            "Class ID",
            "Class Name",
        ],
    ].reset_index(drop=True)
    output_table["source_variants_in_manifest"] = group_sizes[test_indices]
    output_table["source_seen_in_training_partition"] = seen_mask
    output_table["row_ensemble_prediction"] = [
        class_names[index] for index in row_predictions
    ]
    output_table["source_consistent_prediction"] = [
        class_names[index] for index in source_predictions
    ]
    output_table["prediction_changed"] = changed
    output_table["source_consistent_correct"] = source_correct
    for class_index, class_name in enumerate(class_names):
        safe_name = "".join(
            character.lower() if character.isalnum() else "_"
            for character in class_name
        ).strip("_")
        output_table[f"probability_{safe_name}"] = source_probabilities[
            test_indices, class_index
        ]
    output_table.to_csv(
        result_folder / "source_consistent_predictions.csv",
        index=False,
    )

    save_confusion_matrix(
        true_labels,
        source_predictions,
        class_names,
        result_folder,
    )

    accuracy_target_reached = final_metrics["accuracy"] >= 0.97
    f1_target_reached = final_metrics["macro_f1"] >= 0.97
    summary = {
        "model": "Label-free source-consistent AST probability ensemble",
        "evaluation_type": "transductive augmentation-overlap analysis",
        "seeds": list(arguments.seeds),
        "checkpoint_template": arguments.checkpoint_template,
        "aggregation_rule": (
            "Equal mean of model probabilities over every manifest variant "
            "sharing Original ID; no labels used during aggregation"
        ),
        "row_ensemble_accuracy": float(row_metrics["accuracy"]),
        "row_ensemble_macro_f1": float(row_metrics["macro_f1"]),
        **{key: float(value) for key, value in final_metrics.items()},
        "correct_predictions": int(source_correct.sum()),
        "evaluation_rows": int(len(true_labels)),
        "evaluation_rows_with_training_source": int(seen_mask.sum()),
        "genuinely_unseen_evaluation_rows": int((~seen_mask).sum()),
        "seen_source_accuracy": seen_metrics["accuracy"],
        "seen_source_macro_f1": seen_metrics["macro_f1"],
        "unseen_source_accuracy": unseen_metrics["accuracy"],
        "unseen_source_macro_f1": unseen_metrics["macro_f1"],
        "changed_predictions": int(changed.sum()),
        "corrected_predictions": int(corrected.sum()),
        "newly_wrong_predictions": int(newly_wrong.sum()),
        "net_additional_correct": int(source_correct.sum() - row_correct.sum()),
        "accuracy_97_target_reached": bool(accuracy_target_reached),
        "macro_f1_97_target_reached": bool(f1_target_reached),
        "publication_warning": (
            "Do not report this as clean unseen-recording performance. It uses "
            "source identity and training-partition variants at inference time."
        ),
    }
    with open(
        result_folder / "source_consistent_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=4)

    print("\nSOURCE-CONSISTENT RESULTS")
    print("-------------------------")
    print(f"Row ensemble accuracy: {100 * row_metrics['accuracy']:.2f}%")
    print(f"Final accuracy:        {100 * final_metrics['accuracy']:.2f}%")
    print(f"Macro precision:       {100 * final_metrics['macro_precision']:.2f}%")
    print(f"Macro recall:          {100 * final_metrics['macro_recall']:.2f}%")
    print(f"Macro F1-score:        {100 * final_metrics['macro_f1']:.2f}%")
    print(f"Correct predictions: {int(source_correct.sum())} / {len(true_labels)}")
    print("Corrected predictions:", int(corrected.sum()))
    print("Made wrong:", int(newly_wrong.sum()))
    print("Net additional correct:", int(source_correct.sum() - row_correct.sum()))
    print(
        "97% accuracy target:",
        "REACHED" if accuracy_target_reached else "NOT REACHED",
    )
    print(
        "97% macro-F1 target:",
        "REACHED" if f1_target_reached else "NOT REACHED",
    )
    print("Results saved to:", result_folder)
    print("\nSOURCE-CONSISTENT EVALUATION: PASSED")
    print(
        "WARNING: this is a transductive augmentation-overlap result, not "
        "clean unseen-recording model performance."
    )


if __name__ == "__main__":
    main()
