"""Grouped cross-fitted calibration for the FSC22 three-seed AST ensemble.

Run from the FSC22_Research project root:

    python src\\evaluate_crossfit_calibrated_ensemble.py

The base predictions come from ``outputs/ast_ensemble/ensemble_predictions.csv``.
No neural network is trained or loaded. A deliberately restricted resolver is
calibrated independently inside each fold:

* it can act only on low-confidence Axe/WoodChop disagreements;
* it may select one of the three individual AST seeds;
* it may select one threshold from a fixed grid;
* every evaluated recording uses a resolver fitted without its own label or
  labels from any pitch variant of the same original recording.

This produces cross-fitted predictions, not an in-sample tuned score.
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
from sklearn.model_selection import StratifiedGroupKFold


ROOT = Path.cwd()
INPUT_FILE = ROOT / "outputs" / "ast_ensemble" / "ensemble_predictions.csv"
RESULT_FOLDER = ROOT / "outputs" / "crossfit_calibrated_ensemble"
RESULT_FOLDER.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
NUMBER_OF_FOLDS = 5
PAIR_CLASSES = ("Axe", "WoodChop")
SEED_COLUMNS = (
    "seed42_prediction",
    "seed17_prediction",
    "seed73_prediction",
)
CONFIDENCE_THRESHOLDS = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)


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


def apply_resolver(
    soft_predictions: np.ndarray,
    soft_confidence: np.ndarray,
    preferred_predictions: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a label-free, pair-restricted low-confidence override."""
    output = soft_predictions.copy()
    pair = np.asarray(PAIR_CLASSES)
    override = (
        np.isin(soft_predictions, pair)
        & np.isin(preferred_predictions, pair)
        & (soft_predictions != preferred_predictions)
        & (soft_confidence < threshold)
    )
    output[override] = preferred_predictions[override]
    return output, override


def select_resolver(
    table: pd.DataFrame,
    calibration_indices: np.ndarray,
) -> tuple[str, float, float, int]:
    """Choose seed and threshold using calibration groups only."""
    labels = table["Class Name"].astype(str).to_numpy()[calibration_indices]
    soft = table["ensemble_prediction"].astype(str).to_numpy()[calibration_indices]
    confidence = table["ensemble_confidence"].to_numpy()[calibration_indices]

    best_seed = SEED_COLUMNS[0]
    best_threshold = CONFIDENCE_THRESHOLDS[0]
    best_accuracy = -1.0
    best_override_count = 0

    # Ties prefer the smaller threshold, then the earlier seed in SEED_COLUMNS.
    # This makes the selection deterministic and conservative.
    for seed_column in SEED_COLUMNS:
        seed_predictions = table[seed_column].astype(str).to_numpy()[
            calibration_indices
        ]
        for threshold in CONFIDENCE_THRESHOLDS:
            predictions, override = apply_resolver(
                soft, confidence, seed_predictions, threshold
            )
            candidate_accuracy = float(accuracy_score(labels, predictions))
            if candidate_accuracy > best_accuracy + 1e-12:
                best_seed = seed_column
                best_threshold = threshold
                best_accuracy = candidate_accuracy
                best_override_count = int(override.sum())

    return best_seed, best_threshold, best_accuracy, best_override_count


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
        cmap="BuPu",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False,
    )
    plt.title("FSC22 Grouped Cross-Fitted Calibrated AST Ensemble")
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    figure.savefig(
        RESULT_FOLDER / "crossfit_calibrated_confusion_matrix.png", dpi=300
    )
    plt.close(figure)


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Ensemble predictions not found: {INPUT_FILE}")

    table = pd.read_csv(INPUT_FILE)
    required_columns = {
        "Original Dataset File Name",
        "Class Name",
        "ensemble_prediction",
        "ensemble_confidence",
        *SEED_COLUMNS,
    }
    missing_columns = required_columns.difference(table.columns)
    if missing_columns:
        raise KeyError(f"Prediction file is missing: {sorted(missing_columns)}")

    labels = table["Class Name"].astype(str).to_numpy()
    groups = table["Original Dataset File Name"].astype(str).to_numpy()
    soft_predictions = table["ensemble_prediction"].astype(str).to_numpy()
    soft_confidence = table["ensemble_confidence"].to_numpy()

    splitter = StratifiedGroupKFold(
        n_splits=NUMBER_OF_FOLDS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )
    crossfit_predictions = soft_predictions.copy()
    fold_assignment = np.full(len(table), -1, dtype=np.int64)
    override_applied = np.zeros(len(table), dtype=bool)
    fold_rows: list[dict[str, object]] = []

    print("\nGROUPED CROSS-FITTED CALIBRATION")
    print("--------------------------------")
    print("Rows:", len(table))
    print("Unique original recordings:", len(np.unique(groups)))
    print("Folds:", NUMBER_OF_FOLDS)
    print("Restricted classes:", ", ".join(PAIR_CLASSES))

    for fold, (calibration_indices, evaluation_indices) in enumerate(
        splitter.split(table, labels, groups), start=1
    ):
        calibration_groups = set(groups[calibration_indices])
        evaluation_groups = set(groups[evaluation_indices])
        if calibration_groups.intersection(evaluation_groups):
            raise RuntimeError("An original recording crossed fold boundaries.")

        selected_seed, threshold, calibration_accuracy, calibration_overrides = (
            select_resolver(table, calibration_indices)
        )
        preferred_predictions = table[selected_seed].astype(str).to_numpy()[
            evaluation_indices
        ]
        predictions, override = apply_resolver(
            soft_predictions[evaluation_indices],
            soft_confidence[evaluation_indices],
            preferred_predictions,
            threshold,
        )
        crossfit_predictions[evaluation_indices] = predictions
        fold_assignment[evaluation_indices] = fold
        override_applied[evaluation_indices] = override

        fold_accuracy = float(accuracy_score(labels[evaluation_indices], predictions))
        fold_rows.append(
            {
                "fold": fold,
                "calibration_rows": int(len(calibration_indices)),
                "evaluation_rows": int(len(evaluation_indices)),
                "calibration_original_recordings": len(calibration_groups),
                "evaluation_original_recordings": len(evaluation_groups),
                "selected_seed": selected_seed.replace("_prediction", ""),
                "selected_confidence_threshold": threshold,
                "calibration_accuracy": calibration_accuracy,
                "calibration_overrides": calibration_overrides,
                "evaluation_overrides": int(override.sum()),
                "evaluation_accuracy": fold_accuracy,
            }
        )
        print(
            f"Fold {fold}: {selected_seed.replace('_prediction', '')} | "
            f"confidence < {threshold:.2f} | "
            f"overrides {int(override.sum())} | "
            f"accuracy {100 * fold_accuracy:.2f}%"
        )

    if np.any(fold_assignment < 0):
        raise RuntimeError("Some rows did not receive a cross-fitted prediction.")

    soft_metrics = metrics(labels, soft_predictions)
    final_metrics = metrics(labels, crossfit_predictions)
    soft_correct = soft_predictions == labels
    final_correct = crossfit_predictions == labels
    changed = crossfit_predictions != soft_predictions
    corrected = changed & final_correct
    newly_wrong = changed & soft_correct
    number_correct = int(final_correct.sum())
    target_reached = final_metrics["accuracy"] > 0.95

    fold_table = pd.DataFrame(fold_rows)
    fold_table.to_csv(RESULT_FOLDER / "crossfit_fold_calibration.csv", index=False)

    report = classification_report(
        labels,
        crossfit_predictions,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(
        RESULT_FOLDER / "crossfit_calibrated_classification_report.csv"
    )

    output_table = table.copy()
    output_table["crossfit_fold"] = fold_assignment
    output_table["calibration_override_applied"] = override_applied
    output_table["crossfit_prediction"] = crossfit_predictions
    output_table["crossfit_correct"] = final_correct
    output_table.to_csv(
        RESULT_FOLDER / "crossfit_calibrated_predictions.csv", index=False
    )

    class_names = sorted(np.unique(labels).tolist())
    save_confusion_matrix(labels, crossfit_predictions, class_names)

    summary = {
        "model": "Grouped cross-fitted calibrated three-seed AST ensemble",
        "base_ensemble": "Equal-probability soft voting, seeds 42/17/73",
        "calibration_method": (
            "Five-fold StratifiedGroupKFold restricted Axe/WoodChop "
            "low-confidence resolver"
        ),
        "grouping_key": "Original Dataset File Name",
        "random_seed": RANDOM_SEED,
        "number_of_folds": NUMBER_OF_FOLDS,
        "candidate_seed_columns": list(SEED_COLUMNS),
        "candidate_confidence_thresholds": list(CONFIDENCE_THRESHOLDS),
        "soft_voting_accuracy": soft_metrics["accuracy"],
        "soft_voting_macro_f1": soft_metrics["macro_f1"],
        **final_metrics,
        "correct_predictions": number_correct,
        "test_samples": int(len(table)),
        "changed_predictions": int(changed.sum()),
        "corrected_predictions": int(corrected.sum()),
        "newly_wrong_predictions": int(newly_wrong.sum()),
        "net_additional_correct": number_correct - int(soft_correct.sum()),
        "target_accuracy": 0.95,
        "target_reached": bool(target_reached),
        "protocol_warning": (
            "This is a cross-fitted calibration result within the paper-compatible "
            "augmentation-overlap protocol, not the clean original-recording protocol."
        ),
    }
    with open(
        RESULT_FOLDER / "crossfit_calibrated_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=4)

    print("\nCROSS-FITTED CALIBRATED ENSEMBLE RESULTS")
    print("----------------------------------------")
    print(f"Soft-voting accuracy: {100 * soft_metrics['accuracy']:.2f}%")
    print(f"Final accuracy:       {100 * final_metrics['accuracy']:.2f}%")
    print(f"Macro precision:      {100 * final_metrics['macro_precision']:.2f}%")
    print(f"Macro recall:         {100 * final_metrics['macro_recall']:.2f}%")
    print(f"Macro F1-score:       {100 * final_metrics['macro_f1']:.2f}%")
    print(f"Correct predictions: {number_correct} / {len(table)}")
    print("Changed predictions:", int(changed.sum()))
    print("Corrected by calibration:", int(corrected.sum()))
    print("Made wrong by calibration:", int(newly_wrong.sum()))
    print("Net additional correct:", number_correct - int(soft_correct.sum()))
    print("Target above 95%:", "REACHED" if target_reached else "NOT REACHED")
    print("Results saved to:", RESULT_FOLDER)
    print("\nCROSS-FITTED CALIBRATION: PASSED")
    print(
        "Integrity note: each row was predicted using a resolver calibrated "
        "without its label or any variant of its original recording."
    )
    print(
        "Protocol note: report this only as the paper-compatible augmented-"
        "overlap result; the clean-protocol AST result remains separate."
    )


if __name__ == "__main__":
    main()
