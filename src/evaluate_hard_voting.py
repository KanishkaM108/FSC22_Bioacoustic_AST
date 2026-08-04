"""Evaluate fixed hard-majority voting for the three FSC22 AST seeds.

Run from the FSC22_Research project root:

    python src\\evaluate_hard_voting.py

This script reads the predictions already saved by ``train_ast_ensemble.py``.
It performs no training, no model loading, and no checkpoint modification.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
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
INPUT_FILE = ROOT / "outputs" / "ast_ensemble" / "ensemble_predictions.csv"
RESULT_FOLDER = ROOT / "outputs" / "hard_voting"
RESULT_FOLDER.mkdir(parents=True, exist_ok=True)

SEED_COLUMNS = (
    "seed42_prediction",
    "seed17_prediction",
    "seed73_prediction",
)


def hard_vote(row: pd.Series) -> tuple[str, str]:
    """Return majority class and rule used; soft ensemble breaks three-way ties."""
    votes = [str(row[column]) for column in SEED_COLUMNS]
    counts = Counter(votes)
    winner, number_of_votes = counts.most_common(1)[0]
    if number_of_votes >= 2:
        return winner, "hard_majority"
    return str(row["ensemble_prediction"]), "soft_tiebreak"


def metrics(labels: pd.Series, predictions: pd.Series) -> dict[str, float]:
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


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Ensemble predictions not found: {INPUT_FILE}")

    table = pd.read_csv(INPUT_FILE)
    required_columns = {
        "Class Name",
        "ensemble_prediction",
        *SEED_COLUMNS,
    }
    missing_columns = required_columns.difference(table.columns)
    if missing_columns:
        raise KeyError(f"Prediction file is missing: {sorted(missing_columns)}")

    vote_results = table.apply(hard_vote, axis=1)
    table["hard_vote_prediction"] = [result[0] for result in vote_results]
    table["voting_rule"] = [result[1] for result in vote_results]

    labels = table["Class Name"].astype(str)
    soft_predictions = table["ensemble_prediction"].astype(str)
    hard_predictions = table["hard_vote_prediction"].astype(str)
    soft_metrics = metrics(labels, soft_predictions)
    hard_metrics = metrics(labels, hard_predictions)

    soft_correct = soft_predictions == labels
    hard_correct = hard_predictions == labels
    changed = hard_predictions != soft_predictions
    corrected = changed & hard_correct
    newly_wrong = changed & soft_correct
    number_correct = int(hard_correct.sum())
    target_reached = hard_metrics["accuracy"] > 0.95

    report = classification_report(
        labels,
        hard_predictions,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(
        RESULT_FOLDER / "hard_voting_classification_report.csv"
    )

    table["hard_vote_correct"] = hard_correct
    table["prediction_changed_from_soft"] = changed
    table.to_csv(RESULT_FOLDER / "hard_voting_predictions.csv", index=False)

    class_names = sorted(labels.unique().tolist())
    matrix = confusion_matrix(labels, hard_predictions, labels=class_names)
    sns.set_theme(style="white")
    figure = plt.figure(figsize=(18, 15))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Oranges",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False,
    )
    plt.title("FSC22 Three-Seed AST Hard-Majority Voting")
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    figure.savefig(RESULT_FOLDER / "hard_voting_confusion_matrix.png", dpi=300)
    plt.close(figure)

    summary = {
        "model": "Three-seed AST hard-majority voting",
        "seeds": [42, 17, 73],
        "rule": (
            "Class receiving at least two seed votes; equal-probability soft "
            "ensemble used only when all three seeds predict different classes"
        ),
        "soft_voting_accuracy": soft_metrics["accuracy"],
        "soft_voting_macro_f1": soft_metrics["macro_f1"],
        **hard_metrics,
        "correct_predictions": number_correct,
        "test_samples": int(len(table)),
        "majority_rows": int((table["voting_rule"] == "hard_majority").sum()),
        "three_way_tie_rows": int((table["voting_rule"] == "soft_tiebreak").sum()),
        "changed_predictions": int(changed.sum()),
        "corrected_predictions": int(corrected.sum()),
        "newly_wrong_predictions": int(newly_wrong.sum()),
        "net_additional_correct": number_correct - int(soft_correct.sum()),
        "target_accuracy": 0.95,
        "target_reached": bool(target_reached),
    }
    with open(
        RESULT_FOLDER / "hard_voting_metrics.json", "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=4)

    print("\nTHREE-SEED HARD-VOTING RESULTS")
    print("------------------------------")
    print(f"Soft-voting accuracy: {100 * soft_metrics['accuracy']:.2f}%")
    print(f"Final accuracy:       {100 * hard_metrics['accuracy']:.2f}%")
    print(f"Macro precision:      {100 * hard_metrics['macro_precision']:.2f}%")
    print(f"Macro recall:         {100 * hard_metrics['macro_recall']:.2f}%")
    print(f"Macro F1-score:       {100 * hard_metrics['macro_f1']:.2f}%")
    print(f"Correct predictions: {number_correct} / {len(table)}")
    print("Changed predictions:", int(changed.sum()))
    print("Corrected by hard voting:", int(corrected.sum()))
    print("Made wrong by hard voting:", int(newly_wrong.sum()))
    print("Net additional correct:", number_correct - int(soft_correct.sum()))
    print("Target above 95%:", "REACHED" if target_reached else "NOT REACHED")
    print("Results saved to:", RESULT_FOLDER)
    print("\nHARD-VOTING EVALUATION: PASSED")
    print(
        "Integrity note: the voting rule used model predictions only; true "
        "labels were used solely to calculate the final metrics."
    )


if __name__ == "__main__":
    main()
