"""Compare nonlinear classifiers on the trained FSC22 AST embeddings."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, Dataset
from transformers import ASTForAudioClassification


MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
SEED = 42
NUM_CLASSES = 27
EXTRACTION_BATCH_SIZE = 8

ROOT = Path.cwd()
SPLIT_FILE = ROOT / "outputs" / "paper_split_seed42.csv"
FEATURE_FOLDER = ROOT / "outputs" / "ast_cache"
AST_MODEL_PATH = ROOT / "models" / "fsc22_ast_seed42.pt"
RESULT_FOLDER = ROOT / "outputs" / "embedding_analysis"
RESULT_FOLDER.mkdir(parents=True, exist_ok=True)


class FeatureDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        class_to_index: dict[int, int],
    ) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.class_to_index = class_to_index

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int):
        row = self.dataframe.iloc[index]
        filename = Path(str(row["Dataset File Name"])).stem + ".npy"
        features = np.load(FEATURE_FOLDER / filename).astype(np.float32)
        label = self.class_to_index[int(row["Class ID"])]
        return torch.from_numpy(features), torch.tensor(label), str(
            row["Dataset File Name"]
        )


@torch.inference_mode()
def extract_embeddings(
    model: ASTForAudioClassification,
    loader: DataLoader,
    device: torch.device,
):
    model.eval()
    all_embeddings: list[np.ndarray] = []
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_filenames: list[str] = []

    for features, labels, filenames in loader:
        features = features.to(device, non_blocking=True)
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

        all_embeddings.append(embeddings.float().cpu().numpy())
        all_logits.append(logits.float().cpu().numpy())
        all_labels.append(labels.numpy())
        all_filenames.extend(filenames)

    return (
        np.concatenate(all_embeddings),
        np.concatenate(all_logits),
        np.concatenate(all_labels),
        np.asarray(all_filenames),
    )


def calculate_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict:
    return {
        "test_accuracy": accuracy_score(labels, predictions),
        "test_macro_precision": precision_score(
            labels, predictions, average="macro", zero_division=0
        ),
        "test_macro_recall": recall_score(
            labels, predictions, average="macro", zero_division=0
        ),
        "test_macro_f1": f1_score(
            labels, predictions, average="macro", zero_division=0
        ),
    }


def main() -> None:
    if not SPLIT_FILE.exists():
        raise FileNotFoundError(f"Split file not found: {SPLIT_FILE}")
    if not AST_MODEL_PATH.exists():
        raise FileNotFoundError(f"AST checkpoint not found: {AST_MODEL_PATH}")

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
    test_dataframe = metadata[metadata["Split"] == "test"].copy()

    cache_paths = {
        "train_embeddings": RESULT_FOLDER / "train_embeddings.npy",
        "train_logits": RESULT_FOLDER / "train_logits.npy",
        "train_labels": RESULT_FOLDER / "train_labels.npy",
        "train_filenames": RESULT_FOLDER / "train_filenames.npy",
        "test_embeddings": RESULT_FOLDER / "test_embeddings.npy",
        "test_logits": RESULT_FOLDER / "test_logits.npy",
        "test_labels": RESULT_FOLDER / "test_labels.npy",
        "test_filenames": RESULT_FOLDER / "test_filenames.npy",
    }

    if all(path.exists() for path in cache_paths.values()):
        print("Loading previously extracted AST embeddings...")
        train_embeddings = np.load(cache_paths["train_embeddings"])
        train_logits = np.load(cache_paths["train_logits"])
        train_labels = np.load(cache_paths["train_labels"])
        train_filenames = np.load(cache_paths["train_filenames"])
        test_embeddings = np.load(cache_paths["test_embeddings"])
        test_logits = np.load(cache_paths["test_logits"])
        test_labels = np.load(cache_paths["test_labels"])
        test_filenames = np.load(cache_paths["test_filenames"])
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Loading the best AST checkpoint on", device)

        model = ASTForAudioClassification.from_pretrained(
            MODEL_NAME,
            num_labels=NUM_CLASSES,
            id2label=id_to_label,
            label2id=label_to_id,
            ignore_mismatched_sizes=True,
            local_files_only=True,
        )
        model.load_state_dict(
            torch.load(AST_MODEL_PATH, map_location="cpu", weights_only=True)
        )
        model.to(device)

        train_loader = DataLoader(
            FeatureDataset(train_dataframe, class_to_index),
            batch_size=EXTRACTION_BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        test_loader = DataLoader(
            FeatureDataset(test_dataframe, class_to_index),
            batch_size=EXTRACTION_BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )

        print("Extracting training embeddings...")
        (
            train_embeddings,
            train_logits,
            train_labels,
            train_filenames,
        ) = extract_embeddings(model, train_loader, device)
        print("Extracting testing embeddings...")
        (
            test_embeddings,
            test_logits,
            test_labels,
            test_filenames,
        ) = extract_embeddings(model, test_loader, device)

        np.save(cache_paths["train_embeddings"], train_embeddings)
        np.save(cache_paths["train_logits"], train_logits)
        np.save(cache_paths["train_labels"], train_labels)
        np.save(cache_paths["train_filenames"], train_filenames)
        np.save(cache_paths["test_embeddings"], test_embeddings)
        np.save(cache_paths["test_logits"], test_logits)
        np.save(cache_paths["test_labels"], test_labels)
        np.save(cache_paths["test_filenames"], test_filenames)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\nAST embedding shapes:")
    print("Training:", train_embeddings.shape)
    print("Testing:", test_embeddings.shape)

    classifiers = {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=10.0,
                        max_iter=5000,
                        class_weight="balanced",
                        random_state=SEED,
                    ),
                ),
            ]
        ),
        "Linear SVM": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    SVC(kernel="linear", C=1.0, class_weight="balanced"),
                ),
            ]
        ),
        "RBF SVM C=3": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    SVC(kernel="rbf", C=3.0, gamma="scale", class_weight="balanced"),
                ),
            ]
        ),
        "RBF SVM C=10": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    SVC(kernel="rbf", C=10.0, gamma="scale", class_weight="balanced"),
                ),
            ]
        ),
        "Cosine 1-NN": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    KNeighborsClassifier(
                        n_neighbors=1,
                        metric="cosine",
                        algorithm="brute",
                    ),
                ),
            ]
        ),
        "Cosine 3-NN": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    KNeighborsClassifier(
                        n_neighbors=3,
                        metric="cosine",
                        algorithm="brute",
                        weights="distance",
                    ),
                ),
            ]
        ),
        "Shrinkage LDA": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
                ),
            ]
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=500,
            max_features="sqrt",
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        ),
    }

    cross_validation = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=SEED,
    )

    comparison_rows: list[dict] = []
    fitted_models: dict[str, object] = {}
    test_predictions: dict[str, np.ndarray] = {}

    ast_predictions = train_logits.argmax(axis=1)
    ast_test_predictions = test_logits.argmax(axis=1)
    ast_metrics = calculate_metrics(test_labels, ast_test_predictions)
    comparison_rows.append(
        {
            "method": "Original AST head",
            "cv_macro_f1_mean": np.nan,
            "cv_macro_f1_std": np.nan,
            **ast_metrics,
        }
    )
    test_predictions["Original AST head"] = ast_test_predictions

    print("\nCLASSIFIER SWEEP")
    print("----------------")

    for name, classifier in classifiers.items():
        print("Testing", name, "...")
        scores = cross_val_score(
            classifier,
            train_embeddings,
            train_labels,
            cv=cross_validation,
            scoring="f1_macro",
            n_jobs=1,
        )
        classifier.fit(train_embeddings, train_labels)
        predictions = classifier.predict(test_embeddings)
        metrics = calculate_metrics(test_labels, predictions)

        comparison_rows.append(
            {
                "method": name,
                "cv_macro_f1_mean": scores.mean(),
                "cv_macro_f1_std": scores.std(ddof=1),
                **metrics,
            }
        )
        fitted_models[name] = classifier
        test_predictions[name] = predictions

    comparison = pd.DataFrame(comparison_rows)
    comparison = comparison.sort_values(
        "cv_macro_f1_mean",
        ascending=False,
        na_position="last",
    )
    comparison.to_csv(RESULT_FOLDER / "embedding_classifier_comparison.csv", index=False)

    print("\nEMBEDDING CLASSIFIER RESULTS")
    print("----------------------------")
    display = comparison.copy()
    for column in [
        "cv_macro_f1_mean",
        "cv_macro_f1_std",
        "test_accuracy",
        "test_macro_f1",
    ]:
        display[column] = display[column] * 100
    print(
        display[
            [
                "method",
                "cv_macro_f1_mean",
                "cv_macro_f1_std",
                "test_accuracy",
                "test_macro_f1",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.2f}")
    )

    eligible = comparison.dropna(subset=["cv_macro_f1_mean"])
    selected_name = str(eligible.iloc[0]["method"])
    selected_model = fitted_models[selected_name]
    selected_predictions = test_predictions[selected_name]
    selected_metrics = calculate_metrics(test_labels, selected_predictions)

    joblib.dump(
        selected_model,
        RESULT_FOLDER / "selected_embedding_classifier.joblib",
    )

    prediction_table = pd.DataFrame(
        {
            "Dataset File Name": test_filenames,
            "True Label": test_labels,
            "Predicted Label": selected_predictions,
            "Correct": test_labels == selected_predictions,
        }
    )
    prediction_table.to_csv(
        RESULT_FOLDER / "selected_embedding_predictions.csv",
        index=False,
    )

    report = classification_report(
        test_labels,
        selected_predictions,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(
        RESULT_FOLDER / "selected_embedding_classification_report.csv"
    )

    matrix = confusion_matrix(test_labels, selected_predictions)
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
    plt.title(f"FSC22 Confusion Matrix: {selected_name}")
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    figure.savefig(
        RESULT_FOLDER / "selected_embedding_confusion_matrix.png",
        dpi=300,
    )
    plt.close(figure)

    result = {
        "selection_rule": "highest three-fold training CV macro-F1",
        "selected_method": selected_name,
        "accuracy": float(selected_metrics["test_accuracy"]),
        "macro_precision": float(selected_metrics["test_macro_precision"]),
        "macro_recall": float(selected_metrics["test_macro_recall"]),
        "macro_f1": float(selected_metrics["test_macro_f1"]),
        "correct_predictions": int((selected_predictions == test_labels).sum()),
        "test_samples": int(len(test_labels)),
    }
    with open(
        RESULT_FOLDER / "selected_embedding_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(result, file, indent=4)

    print("\nSELECTED EMBEDDING CLASSIFIER")
    print("-----------------------------")
    print("Method:", selected_name)
    print(f"Accuracy:        {100 * selected_metrics['test_accuracy']:.2f}%")
    print(f"Macro precision: {100 * selected_metrics['test_macro_precision']:.2f}%")
    print(f"Macro recall:    {100 * selected_metrics['test_macro_recall']:.2f}%")
    print(f"Macro F1-score:  {100 * selected_metrics['test_macro_f1']:.2f}%")
    print(
        "Correct predictions:",
        int((selected_predictions == test_labels).sum()),
        "/",
        len(test_labels),
    )
    print("Results saved to:", RESULT_FOLDER)
    print("\nEMBEDDING ANALYSIS: PASSED")


if __name__ == "__main__":
    main()
