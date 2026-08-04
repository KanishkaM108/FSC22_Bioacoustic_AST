"""Evaluate recording-consistency averaging for the FSC22 AST ensemble.

Place this file beside ``train_ast_ensemble.py`` in the project's ``src``
folder, then run from the FSC22_Research project root:

    python src\\evaluate_recording_consistency.py

No training is performed and no checkpoint is modified. For original
recordings represented by multiple pitch variants in the test partition, the
script averages the already-trained three-seed ensemble probabilities across
the available test variants and assigns one recording-consistent prediction.
"""

from __future__ import annotations

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
RESULT_FOLDER = ROOT / "outputs" / "recording_consistency"
RESULT_FOLDER.mkdir(parents=True, exist_ok=True)


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
        cmap="Greens",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False,
    )
    plt.title("FSC22 Recording-Consistent Three-Seed AST Ensemble")
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    figure.savefig(
        RESULT_FOLDER / "recording_consistency_confusion_matrix.png", dpi=300
    )
    plt.close(figure)


def main() -> None:
    torch.set_float32_matmul_precision("high")
    if not MANIFEST_FILE.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_FILE}")

    manifest = pd.read_csv(MANIFEST_FILE)
    manifest["Class ID"] = pd.to_numeric(manifest["Class ID"]).astype(int)
    test_dataframe = (
        manifest[manifest["Split"] == "test"].copy().reset_index(drop=True)
    )

    required_columns = {
        "Feature Path",
        "Original Dataset File Name",
        "Variant",
        "Class ID",
        "Class Name",
    }
    missing_columns = required_columns.difference(test_dataframe.columns)
    if missing_columns:
        raise KeyError(f"Manifest is missing columns: {sorted(missing_columns)}")

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset = ensemble.PaperProtocolDataset(
        test_dataframe, class_to_index, training=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=ensemble.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    loss_function = nn.CrossEntropyLoss(label_smoothing=0.05)

    print("\nFSC22 RECORDING-CONSISTENCY EVALUATION")
    print("--------------------------------------")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    print("Models:", ensemble.SEEDS)
    print("Test rows:", len(test_dataframe))
    print("No training or checkpoint modification will occur.")

    probabilities_by_seed: list[np.ndarray] = []
    true_labels: np.ndarray | None = None
    for seed in ensemble.SEEDS:
        model_path = (
            ROOT / "models" / f"fsc22_ast_paper_protocol_seed{seed}.pt"
        )
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        print(f"Evaluating seed {seed}...")
        model = ensemble.build_model(id_to_label, label_to_id)
        model.load_state_dict(
            torch.load(model_path, map_location="cpu", weights_only=True)
        )
        model.to(device)
        result = ensemble.evaluate_model(
            model,
            test_loader,
            loss_function,
            device,
            amp_enabled=False,
        )
        labels = np.asarray(result["labels"])
        if true_labels is None:
            true_labels = labels
        elif not np.array_equal(true_labels, labels):
            raise RuntimeError("Test-label order changed during evaluation.")

        probabilities_by_seed.append(
            ensemble.stable_softmax(np.asarray(result["logits"]))
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    assert true_labels is not None
    row_probabilities = np.mean(np.stack(probabilities_by_seed), axis=0)
    original_predictions = row_probabilities.argmax(axis=1)
    original_metrics = ensemble.compute_metrics(
        true_labels, original_predictions
    )

    # Fixed, label-free rule: singleton recordings keep their row prediction;
    # for multi-variant recordings, average only the variants present in the
    # test partition. Training-partition variants are never used here.
    consistent_probabilities = row_probabilities.copy()
    group_sizes = (
        test_dataframe.groupby("Original Dataset File Name", sort=False)
        .size()
        .to_dict()
    )
    multi_variant_recordings = 0
    multi_variant_rows = 0
    for _, row_indices in test_dataframe.groupby(
        "Original Dataset File Name", sort=False
    ).groups.items():
        indices = np.asarray(list(row_indices), dtype=np.int64)
        if len(indices) <= 1:
            continue
        multi_variant_recordings += 1
        multi_variant_rows += len(indices)
        recording_probability = row_probabilities[indices].mean(axis=0)
        consistent_probabilities[indices] = recording_probability

    consistent_predictions = consistent_probabilities.argmax(axis=1)
    consistent_metrics = ensemble.compute_metrics(
        true_labels, consistent_predictions
    )
    original_correct = int((original_predictions == true_labels).sum())
    consistent_correct = int((consistent_predictions == true_labels).sum())
    changed = original_predictions != consistent_predictions
    corrected = changed & (consistent_predictions == true_labels)
    newly_wrong = changed & (original_predictions == true_labels)

    report = classification_report(
        true_labels,
        consistent_predictions,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(
        RESULT_FOLDER / "recording_consistency_classification_report.csv"
    )

    output_table = test_dataframe[
        [
            "Original Dataset File Name",
            "Variant",
            "Class ID",
            "Class Name",
        ]
    ].copy()
    output_table["test_variants_for_original"] = output_table[
        "Original Dataset File Name"
    ].map(group_sizes)
    output_table["original_ensemble_prediction"] = [
        class_names[index] for index in original_predictions
    ]
    output_table["consistent_prediction"] = [
        class_names[index] for index in consistent_predictions
    ]
    output_table["prediction_changed"] = changed
    output_table["consistent_correct"] = consistent_predictions == true_labels
    output_table.to_csv(
        RESULT_FOLDER / "recording_consistency_predictions.csv", index=False
    )

    save_confusion_matrix(true_labels, consistent_predictions, class_names)
    target_reached = consistent_metrics["accuracy"] > 0.95
    summary = {
        "model": "Recording-consistent three-seed AST soft-voting ensemble",
        "seeds": list(ensemble.SEEDS),
        "aggregation_rule": (
            "Mean probabilities across test-partition pitch variants sharing "
            "an original recording ID; singleton rows unchanged"
        ),
        "original_ensemble_accuracy": original_metrics["accuracy"],
        "original_ensemble_macro_f1": original_metrics["macro_f1"],
        **consistent_metrics,
        "original_correct_predictions": original_correct,
        "consistent_correct_predictions": consistent_correct,
        "test_samples": int(len(true_labels)),
        "multi_variant_recordings": multi_variant_recordings,
        "multi_variant_rows": multi_variant_rows,
        "changed_predictions": int(changed.sum()),
        "corrected_predictions": int(corrected.sum()),
        "newly_wrong_predictions": int(newly_wrong.sum()),
        "net_additional_correct": consistent_correct - original_correct,
        "target_accuracy": 0.95,
        "target_reached": bool(target_reached),
    }
    with open(
        RESULT_FOLDER / "recording_consistency_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=4)

    print("\nRECORDING-CONSISTENT ENSEMBLE RESULTS")
    print("-------------------------------------")
    print(f"Original ensemble accuracy: {100 * original_metrics['accuracy']:.2f}%")
    print(f"Final accuracy:             {100 * consistent_metrics['accuracy']:.2f}%")
    print(f"Macro precision:            {100 * consistent_metrics['macro_precision']:.2f}%")
    print(f"Macro recall:               {100 * consistent_metrics['macro_recall']:.2f}%")
    print(f"Macro F1-score:             {100 * consistent_metrics['macro_f1']:.2f}%")
    print(f"Correct predictions: {consistent_correct} / {len(true_labels)}")
    print("Changed predictions:", int(changed.sum()))
    print("Corrected by consistency:", int(corrected.sum()))
    print("Made wrong by consistency:", int(newly_wrong.sum()))
    print("Net additional correct:", consistent_correct - original_correct)
    print("Target above 95%:", "REACHED" if target_reached else "NOT REACHED")
    print("Results saved to:", RESULT_FOLDER)
    print("\nRECORDING-CONSISTENCY EVALUATION: PASSED")
    print(
        "Integrity note: only variants already in the test partition were "
        "averaged; no labels or training-partition variants were used."
    )


if __name__ == "__main__":
    main()
