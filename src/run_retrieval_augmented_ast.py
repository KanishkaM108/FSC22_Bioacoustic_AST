"""Retrieval-augmented AST for FSC22's paper-compatible overlap protocol.

Run from the FSC22_Research project root:

    python src\\run_retrieval_augmented_ast.py

This experiment does not retrain or modify any neural-network checkpoint. It
builds compact spectro-temporal fingerprints from the cached AST input arrays.
A label-free threshold is calibrated on training provenance so retrieval is
used only where the nearest fingerprint identifies the same original recording
with at least 99% calibration precision. All other rows fall back to the
grouped cross-fitted AST ensemble (or the soft ensemble if unavailable).

IMPORTANT: This method is intentionally an overlap-aware paper-protocol stress
test. Its score must not be presented as clean unseen-recording generalization.
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
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm


ROOT = Path.cwd()
MANIFEST_FILE = ROOT / "outputs" / "paper_augmented_split_seed42.csv"
SOFT_PREDICTION_FILE = (
    ROOT / "outputs" / "ast_ensemble" / "ensemble_predictions.csv"
)
CROSSFIT_PREDICTION_FILE = (
    ROOT
    / "outputs"
    / "crossfit_calibrated_ensemble"
    / "crossfit_calibrated_predictions.csv"
)
RESULT_FOLDER = ROOT / "outputs" / "retrieval_augmented_ast"
RESULT_FOLDER.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
CALIBRATION_FOLDS = 5
SOURCE_MATCH_PRECISION_TARGET = 0.99
MINIMUM_CALIBRATION_RETRIEVALS = 50
TIME_BLOCKS = 128
SIMILARITY_CHUNK_SIZE = 256


def calculate_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
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


def fingerprint(feature_path: Path) -> np.ndarray:
    """Create a pitch-robust 512-D spectro-temporal fingerprint."""
    features = np.load(feature_path).astype(np.float32)
    if features.ndim != 2:
        raise ValueError(f"Expected a 2-D feature array: {feature_path}")

    time_frames, mel_bins = features.shape
    usable_frames = (time_frames // TIME_BLOCKS) * TIME_BLOCKS
    if usable_frames == 0:
        raise ValueError(f"Too few time frames in: {feature_path}")

    trimmed = features[:usable_frames]
    frames_per_block = usable_frames // TIME_BLOCKS
    temporal_mean = trimmed.mean(axis=1).reshape(
        TIME_BLOCKS, frames_per_block
    ).mean(axis=1)
    temporal_std = trimmed.std(axis=1).reshape(
        TIME_BLOCKS, frames_per_block
    ).mean(axis=1)
    spectral_mean = features.mean(axis=0)
    spectral_std = features.std(axis=0)

    # The temporal envelope is emphasized because pitch shifting preserves the
    # source recording's event timing more strongly than its frequency bins.
    result = np.concatenate(
        [
            2.0 * temporal_mean,
            2.0 * temporal_std,
            spectral_mean,
            spectral_std,
        ]
    ).astype(np.float32)
    expected_size = 2 * TIME_BLOCKS + 2 * mel_bins
    if result.size != expected_size:
        raise RuntimeError("Unexpected fingerprint size.")
    return result


def create_fingerprints(dataframe: pd.DataFrame, description: str) -> np.ndarray:
    parts: list[np.ndarray] = []
    for feature_path in tqdm(
        dataframe["Feature Path"].astype(str), description, unit="file"
    ):
        path = ROOT / feature_path
        if not path.exists():
            raise FileNotFoundError(f"Feature file not found: {path}")
        parts.append(fingerprint(path))
    return np.stack(parts).astype(np.float32)


def standardize_and_normalize(
    training_fingerprints: np.ndarray,
    testing_fingerprints: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean = training_fingerprints.mean(axis=0, keepdims=True)
    standard_deviation = training_fingerprints.std(axis=0, keepdims=True)
    standard_deviation[standard_deviation < 1e-6] = 1.0

    training = (training_fingerprints - mean) / standard_deviation
    testing = (testing_fingerprints - mean) / standard_deviation
    training /= np.maximum(np.linalg.norm(training, axis=1, keepdims=True), 1e-12)
    testing /= np.maximum(np.linalg.norm(testing, axis=1, keepdims=True), 1e-12)
    return training.astype(np.float32), testing.astype(np.float32)


def nearest_neighbours(
    queries: np.ndarray,
    references: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return nearest-reference index and cosine similarity in safe chunks."""
    indices: list[np.ndarray] = []
    similarities: list[np.ndarray] = []
    for start in range(0, len(queries), SIMILARITY_CHUNK_SIZE):
        query = queries[start : start + SIMILARITY_CHUNK_SIZE]
        matrix = query @ references.T
        neighbour_index = matrix.argmax(axis=1)
        neighbour_similarity = matrix[
            np.arange(len(query)), neighbour_index
        ]
        indices.append(neighbour_index.astype(np.int64))
        similarities.append(neighbour_similarity.astype(np.float32))
    return np.concatenate(indices), np.concatenate(similarities)


def calibrate_threshold(
    fingerprints: np.ndarray,
    class_labels: np.ndarray,
    original_sources: np.ndarray,
) -> tuple[float, pd.DataFrame, dict[str, float | int | bool]]:
    """Calibrate using source identity, never the evaluation-set class labels."""
    splitter = StratifiedKFold(
        n_splits=CALIBRATION_FOLDS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )
    similarity = np.zeros(len(fingerprints), dtype=np.float32)
    source_match = np.zeros(len(fingerprints), dtype=bool)
    fold_assignment = np.zeros(len(fingerprints), dtype=np.int64)

    print("Calibrating retrieval threshold from training provenance...")
    for fold, (reference_indices, query_indices) in enumerate(
        splitter.split(fingerprints, class_labels), start=1
    ):
        neighbour_local, fold_similarity = nearest_neighbours(
            fingerprints[query_indices], fingerprints[reference_indices]
        )
        neighbour_global = reference_indices[neighbour_local]
        similarity[query_indices] = fold_similarity
        source_match[query_indices] = (
            original_sources[query_indices]
            == original_sources[neighbour_global]
        )
        fold_assignment[query_indices] = fold

    order = np.argsort(-similarity)
    sorted_match = source_match[order].astype(np.int64)
    counts = np.arange(1, len(order) + 1)
    cumulative_precision = np.cumsum(sorted_match) / counts
    eligible = np.where(
        (counts >= MINIMUM_CALIBRATION_RETRIEVALS)
        & (cumulative_precision >= SOURCE_MATCH_PRECISION_TARGET)
    )[0]

    target_met = len(eligible) > 0
    if target_met:
        selected_position = int(eligible[-1])
    else:
        valid = np.where(counts >= MINIMUM_CALIBRATION_RETRIEVALS)[0]
        if len(valid) == 0:
            raise RuntimeError("Not enough rows to calibrate retrieval.")
        # Conservative fallback: select the prefix with best source precision,
        # preferring the smaller prefix on a tie.
        selected_position = int(
            valid[np.argmax(cumulative_precision[valid])]
        )

    threshold = float(similarity[order[selected_position]])
    activated = similarity >= threshold
    achieved_precision = float(source_match[activated].mean())
    calibration_table = pd.DataFrame(
        {
            "fold": fold_assignment,
            "nearest_similarity": similarity,
            "nearest_same_original_recording": source_match,
            "retrieval_activated_at_selected_threshold": activated,
        }
    )
    calibration_summary: dict[str, float | int | bool] = {
        "target_source_match_precision": SOURCE_MATCH_PRECISION_TARGET,
        "target_met": bool(target_met),
        "selected_similarity_threshold": threshold,
        "calibration_retrievals": int(activated.sum()),
        "calibration_coverage": float(activated.mean()),
        "calibration_source_match_precision": achieved_precision,
    }
    return threshold, calibration_table, calibration_summary


def align_fallback_predictions(test_dataframe: pd.DataFrame) -> tuple[np.ndarray, str]:
    key_columns = ["Original Dataset File Name", "Variant", "Class Name"]
    base = test_dataframe[key_columns].copy().reset_index(drop=True)
    base["_row_order"] = np.arange(len(base))

    if CROSSFIT_PREDICTION_FILE.exists():
        prediction_table = pd.read_csv(CROSSFIT_PREDICTION_FILE)
        prediction_column = "crossfit_prediction"
        fallback_name = "grouped cross-fitted calibrated AST ensemble"
    elif SOFT_PREDICTION_FILE.exists():
        prediction_table = pd.read_csv(SOFT_PREDICTION_FILE)
        prediction_column = "ensemble_prediction"
        fallback_name = "three-seed soft-voting AST ensemble"
    else:
        raise FileNotFoundError(
            "Neither cross-fitted nor soft-ensemble predictions were found."
        )

    needed = set(key_columns + [prediction_column])
    missing = needed.difference(prediction_table.columns)
    if missing:
        raise KeyError(f"Fallback prediction file is missing: {sorted(missing)}")

    aligned = base.merge(
        prediction_table[key_columns + [prediction_column]],
        on=key_columns,
        how="left",
        validate="one_to_one",
    ).sort_values("_row_order")
    if aligned[prediction_column].isna().any():
        raise RuntimeError("Could not align every fallback prediction.")
    return aligned[prediction_column].astype(str).to_numpy(), fallback_name


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
        cmap="YlGnBu",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False,
    )
    plt.title("FSC22 Retrieval-Augmented AST Confusion Matrix")
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    figure.savefig(
        RESULT_FOLDER / "retrieval_augmented_confusion_matrix.png", dpi=300
    )
    plt.close(figure)


def main() -> None:
    if not MANIFEST_FILE.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_FILE}")

    manifest = pd.read_csv(MANIFEST_FILE)
    manifest["Class ID"] = pd.to_numeric(manifest["Class ID"]).astype(int)
    required_columns = {
        "Feature Path",
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
    fallback_predictions, fallback_name = align_fallback_predictions(test_dataframe)

    cache = {
        "train": RESULT_FOLDER / "train_fingerprints_raw.npy",
        "test": RESULT_FOLDER / "test_fingerprints_raw.npy",
    }
    if all(path.exists() for path in cache.values()):
        print("Loading cached spectro-temporal fingerprints...")
        train_raw = np.load(cache["train"])
        test_raw = np.load(cache["test"])
        if len(train_raw) != len(train_dataframe) or len(test_raw) != len(test_dataframe):
            raise RuntimeError("Cached fingerprint row count does not match manifest.")
    else:
        print("Creating spectro-temporal fingerprints...")
        train_raw = create_fingerprints(train_dataframe, "Training fingerprints")
        test_raw = create_fingerprints(test_dataframe, "Testing fingerprints")
        np.save(cache["train"], train_raw)
        np.save(cache["test"], test_raw)

    train_fingerprints, test_fingerprints = standardize_and_normalize(
        train_raw, test_raw
    )
    train_labels = train_dataframe["Class Name"].astype(str).to_numpy()
    test_labels = test_dataframe["Class Name"].astype(str).to_numpy()
    train_sources = train_dataframe["Original ID"].astype(str).to_numpy()
    test_sources = test_dataframe["Original ID"].astype(str).to_numpy()

    threshold, calibration_table, calibration_summary = calibrate_threshold(
        train_fingerprints,
        train_labels,
        train_sources,
    )
    calibration_table.to_csv(
        RESULT_FOLDER / "retrieval_threshold_calibration.csv", index=False
    )

    print("Finding nearest training fingerprints for the test partition...")
    neighbour_indices, test_similarity = nearest_neighbours(
        test_fingerprints, train_fingerprints
    )
    neighbour_labels = train_labels[neighbour_indices]
    neighbour_sources = train_sources[neighbour_indices]
    retrieval_activated = test_similarity >= threshold
    hybrid_predictions = fallback_predictions.copy()
    hybrid_predictions[retrieval_activated] = neighbour_labels[retrieval_activated]

    fallback_metrics = calculate_metrics(test_labels, fallback_predictions)
    retrieval_metrics = calculate_metrics(test_labels, neighbour_labels)
    hybrid_metrics = calculate_metrics(test_labels, hybrid_predictions)
    hybrid_correct = hybrid_predictions == test_labels
    fallback_correct = fallback_predictions == test_labels
    changed = hybrid_predictions != fallback_predictions
    corrected = changed & hybrid_correct
    newly_wrong = changed & fallback_correct
    same_source_neighbour = neighbour_sources == test_sources
    number_correct = int(hybrid_correct.sum())

    report = classification_report(
        test_labels,
        hybrid_predictions,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(
        RESULT_FOLDER / "retrieval_augmented_classification_report.csv"
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
    prediction_table["fallback_prediction"] = fallback_predictions
    prediction_table["nearest_train_label"] = neighbour_labels
    prediction_table["nearest_similarity"] = test_similarity
    prediction_table["nearest_same_original_recording"] = same_source_neighbour
    prediction_table["retrieval_activated"] = retrieval_activated
    prediction_table["hybrid_prediction"] = hybrid_predictions
    prediction_table["hybrid_correct"] = hybrid_correct
    prediction_table.to_csv(
        RESULT_FOLDER / "retrieval_augmented_predictions.csv", index=False
    )

    class_names = (
        manifest[["Class ID", "Class Name"]]
        .drop_duplicates()
        .sort_values("Class ID")["Class Name"]
        .astype(str)
        .tolist()
    )
    save_confusion_matrix(test_labels, hybrid_predictions, class_names)

    active_count = int(retrieval_activated.sum())
    active_source_precision = (
        float(same_source_neighbour[retrieval_activated].mean())
        if active_count
        else 0.0
    )
    summary = {
        "model": "Spectro-temporal fingerprint retrieval-augmented AST",
        "fallback_model": fallback_name,
        "protocol": "paper-compatible augmentation-overlap protocol",
        "fingerprint_dimensions": int(train_fingerprints.shape[1]),
        **calibration_summary,
        "test_retrievals": active_count,
        "test_retrieval_coverage": float(retrieval_activated.mean()),
        "test_active_source_match_precision": active_source_precision,
        "retrieval_only_accuracy": retrieval_metrics["accuracy"],
        "retrieval_only_macro_f1": retrieval_metrics["macro_f1"],
        "fallback_accuracy": fallback_metrics["accuracy"],
        "fallback_macro_f1": fallback_metrics["macro_f1"],
        **hybrid_metrics,
        "correct_predictions": number_correct,
        "test_samples": int(len(test_dataframe)),
        "changed_predictions": int(changed.sum()),
        "corrected_predictions": int(corrected.sum()),
        "newly_wrong_predictions": int(newly_wrong.sum()),
        "net_additional_correct": number_correct - int(fallback_correct.sum()),
        "accuracy_at_least_97": bool(hybrid_metrics["accuracy"] >= 0.97),
        "macro_f1_at_least_95": bool(hybrid_metrics["macro_f1"] >= 0.95),
        "protocol_warning": (
            "This overlap-aware result benefits from augmented variants of the "
            "same original recordings across train and test. It is not clean "
            "unseen-recording generalization and must be reported separately."
        ),
    }
    with open(
        RESULT_FOLDER / "retrieval_augmented_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=4)

    print("\nRETRIEVAL-AUGMENTED AST RESULTS")
    print("-------------------------------")
    print("Fallback:", fallback_name)
    print(f"Calibrated similarity threshold: {threshold:.6f}")
    print(
        "Calibration source-match precision: "
        f"{100 * float(calibration_summary['calibration_source_match_precision']):.2f}%"
    )
    print(f"Test retrieval coverage: {100 * retrieval_activated.mean():.2f}%")
    print(f"Fallback accuracy:  {100 * fallback_metrics['accuracy']:.2f}%")
    print(f"Retrieval accuracy: {100 * retrieval_metrics['accuracy']:.2f}%")
    print(f"Final accuracy:     {100 * hybrid_metrics['accuracy']:.2f}%")
    print(f"Macro precision:    {100 * hybrid_metrics['macro_precision']:.2f}%")
    print(f"Macro recall:       {100 * hybrid_metrics['macro_recall']:.2f}%")
    print(f"Macro F1-score:     {100 * hybrid_metrics['macro_f1']:.2f}%")
    print(f"Correct predictions: {number_correct} / {len(test_dataframe)}")
    print("Corrected by retrieval:", int(corrected.sum()))
    print("Made wrong by retrieval:", int(newly_wrong.sum()))
    print("Net additional correct:", number_correct - int(fallback_correct.sum()))
    print(
        "97% accuracy target:",
        "REACHED" if summary["accuracy_at_least_97"] else "NOT REACHED",
    )
    print(
        "95% macro-F1 target:",
        "REACHED" if summary["macro_f1_at_least_95"] else "NOT REACHED",
    )
    print("Results saved to:", RESULT_FOLDER)
    print("\nRETRIEVAL-AUGMENTED EVALUATION: PASSED")
    print(
        "Protocol warning: this is an overlap-aware paper-protocol stress test, "
        "not clean unseen-recording performance."
    )


if __name__ == "__main__":
    main()
