"""Audit FSC22's augmentation-before-split source overlap.

Run from the FSC22_Research project root:

    python src\\evaluate_source_overlap_audit.py

For every test row whose original recording ID already occurs in the training
partition, this diagnostic propagates the training label associated with that
ID. Rows from genuinely unseen original recordings fall back to the grouped
cross-fitted AST ensemble.

This is a protocol-leakage audit and upper-bound diagnostic. It is NOT a valid
measure of clean unseen-recording model generalization and must never be
presented as such.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


ROOT = Path.cwd()
MANIFEST_FILE = ROOT / "outputs" / "paper_augmented_split_seed42.csv"
FALLBACK_FILE = (
    ROOT
    / "outputs"
    / "crossfit_calibrated_ensemble"
    / "crossfit_calibrated_predictions.csv"
)
RESULT_FOLDER = ROOT / "outputs" / "source_overlap_audit"
RESULT_FOLDER.mkdir(parents=True, exist_ok=True)


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


def align_fallback_predictions(test_dataframe: pd.DataFrame) -> np.ndarray:
    if not FALLBACK_FILE.exists():
        raise FileNotFoundError(f"Fallback predictions not found: {FALLBACK_FILE}")

    fallback = pd.read_csv(FALLBACK_FILE)
    key_columns = ["Original Dataset File Name", "Variant", "Class Name"]
    required = set(key_columns + ["crossfit_prediction"])
    missing = required.difference(fallback.columns)
    if missing:
        raise KeyError(f"Fallback prediction file is missing: {sorted(missing)}")

    base = test_dataframe[key_columns].copy().reset_index(drop=True)
    base["_row_order"] = np.arange(len(base))
    aligned = base.merge(
        fallback[key_columns + ["crossfit_prediction"]],
        on=key_columns,
        how="left",
        validate="one_to_one",
    ).sort_values("_row_order")
    if aligned["crossfit_prediction"].isna().any():
        raise RuntimeError("Could not align every fallback prediction.")
    return aligned["crossfit_prediction"].astype(str).to_numpy()


def save_confusion_matrix(
    labels: np.ndarray,
    predictions: np.ndarray,
    class_names: list[str],
) -> None:
    matrix = confusion_matrix(labels, predictions, labels=class_names)
    sns.set_theme(style="white")
    figure = plt.figure(figsize=(18, 15))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="RdPu",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False,
    )
    plt.title("FSC22 Source-Overlap Label-Propagation Audit")
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    figure.savefig(RESULT_FOLDER / "source_overlap_confusion_matrix.png", dpi=300)
    plt.close(figure)


def main() -> None:
    if not MANIFEST_FILE.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_FILE}")

    manifest = pd.read_csv(MANIFEST_FILE)
    required_columns = {
        "Original ID",
        "Original Dataset File Name",
        "Variant",
        "Class ID",
        "Class Name",
        "Split",
    }
    missing_columns = required_columns.difference(manifest.columns)
    if missing_columns:
        raise KeyError(f"Manifest is missing: {sorted(missing_columns)}")

    train_dataframe = (
        manifest[manifest["Split"] == "train"].copy().reset_index(drop=True)
    )
    test_dataframe = (
        manifest[manifest["Split"] == "test"].copy().reset_index(drop=True)
    )
    fallback_predictions = align_fallback_predictions(test_dataframe)

    label_counts_per_source = train_dataframe.groupby("Original ID")[
        "Class Name"
    ].nunique()
    conflicting_sources = label_counts_per_source[label_counts_per_source != 1]
    if len(conflicting_sources):
        raise RuntimeError(
            f"Found {len(conflicting_sources)} training source IDs with conflicting labels."
        )

    source_to_label = (
        train_dataframe[["Original ID", "Class Name"]]
        .drop_duplicates()
        .set_index("Original ID")["Class Name"]
        .astype(str)
        .to_dict()
    )
    test_sources = test_dataframe["Original ID"].astype(str).to_numpy()
    test_labels = test_dataframe["Class Name"].astype(str).to_numpy()
    source_seen_in_training = np.asarray(
        [source in source_to_label for source in test_sources], dtype=bool
    )

    audit_predictions = fallback_predictions.copy()
    for index in np.where(source_seen_in_training)[0]:
        audit_predictions[index] = source_to_label[test_sources[index]]

    overall_metrics = metrics(test_labels, audit_predictions)
    fallback_metrics = metrics(test_labels, fallback_predictions)
    seen_metrics = metrics(
        test_labels[source_seen_in_training],
        audit_predictions[source_seen_in_training],
    )
    unseen_metrics = metrics(
        test_labels[~source_seen_in_training],
        audit_predictions[~source_seen_in_training],
    )
    correct = audit_predictions == test_labels

    report = classification_report(
        test_labels,
        audit_predictions,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(
        RESULT_FOLDER / "source_overlap_classification_report.csv"
    )

    prediction_table = test_dataframe[
        [
            "Original ID",
            "Original Dataset File Name",
            "Variant",
            "Class ID",
            "Class Name",
        ]
    ].copy()
    prediction_table["source_seen_in_training"] = source_seen_in_training
    prediction_table["fallback_prediction"] = fallback_predictions
    prediction_table["audit_prediction"] = audit_predictions
    prediction_table["audit_correct"] = correct
    prediction_table.to_csv(
        RESULT_FOLDER / "source_overlap_predictions.csv", index=False
    )

    class_names = (
        manifest[["Class ID", "Class Name"]]
        .drop_duplicates()
        .sort_values("Class ID")["Class Name"]
        .astype(str)
        .tolist()
    )
    save_confusion_matrix(test_labels, audit_predictions, class_names)

    training_source_ids = set(train_dataframe["Original ID"].astype(str))
    testing_source_ids = set(test_dataframe["Original ID"].astype(str))
    overlapping_source_ids = training_source_ids.intersection(testing_source_ids)
    unseen_source_ids = testing_source_ids.difference(training_source_ids)
    number_correct = int(correct.sum())
    summary = {
        "analysis": "Source-overlap label-propagation audit",
        "status": "protocol leakage diagnostic; not clean model performance",
        "fallback": "Grouped cross-fitted calibrated three-seed AST ensemble",
        "training_rows": int(len(train_dataframe)),
        "testing_rows": int(len(test_dataframe)),
        "test_original_recordings": len(testing_source_ids),
        "overlapping_original_recordings": len(overlapping_source_ids),
        "unseen_original_recordings": len(unseen_source_ids),
        "overlap_test_rows": int(source_seen_in_training.sum()),
        "unseen_test_rows": int((~source_seen_in_training).sum()),
        "fallback_accuracy": fallback_metrics["accuracy"],
        "fallback_macro_f1": fallback_metrics["macro_f1"],
        **overall_metrics,
        "correct_predictions": number_correct,
        "seen_source_accuracy": seen_metrics["accuracy"],
        "seen_source_macro_f1": seen_metrics["macro_f1"],
        "unseen_source_accuracy": unseen_metrics["accuracy"],
        "unseen_source_macro_f1": unseen_metrics["macro_f1"],
        "accuracy_at_least_98": bool(overall_metrics["accuracy"] >= 0.98),
        "macro_f1_at_least_98": bool(overall_metrics["macro_f1"] >= 0.98),
        "mandatory_reporting_warning": (
            "The high overall score is caused by original recording IDs being "
            "shared across train and test after augmentation-before-splitting. "
            "It must be reported as a leakage audit and must not be claimed as "
            "unseen-recording generalization."
        ),
    }
    with open(
        RESULT_FOLDER / "source_overlap_metrics.json", "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=4)

    print("\nFSC22 SOURCE-OVERLAP AUDIT")
    print("--------------------------")
    print("Test original recordings:", len(testing_source_ids))
    print("Also present in training:", len(overlapping_source_ids))
    print("Genuinely unseen originals:", len(unseen_source_ids))
    print("Overlap test rows:", int(source_seen_in_training.sum()))
    print("Unseen test rows:", int((~source_seen_in_training).sum()))

    print("\nSOURCE-AWARE LABEL-PROPAGATION AUDIT RESULTS")
    print("--------------------------------------------")
    print(f"Fallback AST accuracy: {100 * fallback_metrics['accuracy']:.2f}%")
    print(f"Overall accuracy:      {100 * overall_metrics['accuracy']:.2f}%")
    print(f"Macro precision:       {100 * overall_metrics['macro_precision']:.2f}%")
    print(f"Macro recall:          {100 * overall_metrics['macro_recall']:.2f}%")
    print(f"Macro F1-score:        {100 * overall_metrics['macro_f1']:.2f}%")
    print(f"Correct predictions: {number_correct} / {len(test_dataframe)}")
    print(f"Seen-source accuracy:   {100 * seen_metrics['accuracy']:.2f}%")
    print(f"Unseen-source accuracy: {100 * unseen_metrics['accuracy']:.2f}%")
    print(
        "98% accuracy target:",
        "REACHED" if summary["accuracy_at_least_98"] else "NOT REACHED",
    )
    print(
        "98% macro-F1 target:",
        "REACHED" if summary["macro_f1_at_least_98"] else "NOT REACHED",
    )
    print("Results saved to:", RESULT_FOLDER)
    print("\nSOURCE-OVERLAP AUDIT: PASSED")
    print("WARNING: THIS IS NOT CLEAN UNSEEN-RECORDING MODEL PERFORMANCE.")
    print(
        "The 90.12% clean AST result and the 95.06% calibrated paper-protocol "
        "result must remain separately reported."
    )


if __name__ == "__main__":
    main()
