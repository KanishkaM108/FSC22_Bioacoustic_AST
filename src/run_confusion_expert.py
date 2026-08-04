"""Confusion-aware Axe/WoodChop expert for the FSC22 paper protocol."""

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
BATCH_SIZE = 4

ROOT = Path.cwd()
MANIFEST_FILE = ROOT / "outputs" / "paper_augmented_split_seed42.csv"
MODEL_PATH = ROOT / "models" / "fsc22_ast_paper_protocol_seed42.pt"
RESULT_FOLDER = ROOT / "outputs" / "confusion_expert"
RESULT_FOLDER.mkdir(parents=True, exist_ok=True)


class ManifestDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, class_to_index: dict[int, int]):
        self.dataframe = dataframe.reset_index(drop=True)
        self.class_to_index = class_to_index

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int):
        row = self.dataframe.iloc[index]
        features = np.load(ROOT / str(row["Feature Path"])).astype(np.float32)
        label = self.class_to_index[int(row["Class ID"])]
        return (
            torch.from_numpy(features),
            torch.tensor(label, dtype=torch.long),
            str(row["Original Dataset File Name"]),
            str(row["Variant"]),
        )


@torch.inference_mode()
def extract(
    model: ASTForAudioClassification,
    loader: DataLoader,
    device: torch.device,
):
    model.eval()
    embedding_parts: list[np.ndarray] = []
    logit_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    filenames: list[str] = []
    variants: list[str] = []

    for features, labels, batch_filenames, batch_variants in loader:
        features = features.to(device, non_blocking=True)

        # Float32 inference matches the final paper-protocol evaluation.
        backbone_output = model.audio_spectrogram_transformer(
            input_values=features
        )
        embeddings = backbone_output.pooler_output
        logits = model.classifier(embeddings)

        embedding_parts.append(embeddings.cpu().numpy())
        logit_parts.append(logits.cpu().numpy())
        label_parts.append(labels.numpy())
        filenames.extend(batch_filenames)
        variants.extend(batch_variants)

    return (
        np.concatenate(embedding_parts),
        np.concatenate(logit_parts),
        np.concatenate(label_parts),
        np.asarray(filenames),
        np.asarray(variants),
    )


def metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_precision": precision_score(
            labels, predictions, average="macro", zero_division=0
        ),
        "macro_recall": recall_score(
            labels, predictions, average="macro", zero_division=0
        ),
        "macro_f1": f1_score(
            labels, predictions, average="macro", zero_division=0
        ),
    }


def main() -> None:
    if not MANIFEST_FILE.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_FILE}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    manifest = pd.read_csv(MANIFEST_FILE)
    manifest["Class ID"] = pd.to_numeric(manifest["Class ID"]).astype(int)
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

    axe_index = class_names.index("Axe")
    woodchop_index = class_names.index("WoodChop")
    pair_indices = np.asarray([axe_index, woodchop_index])

    train_dataframe = manifest[manifest["Split"] == "train"].copy()
    pair_train_dataframe = train_dataframe[
        train_dataframe["Class Name"].isin(["Axe", "WoodChop"])
    ].copy()
    test_dataframe = manifest[manifest["Split"] == "test"].copy()

    cache = {
        "pair_train_embeddings": RESULT_FOLDER / "pair_train_embeddings.npy",
        "pair_train_labels": RESULT_FOLDER / "pair_train_labels.npy",
        "test_embeddings": RESULT_FOLDER / "test_embeddings.npy",
        "test_logits": RESULT_FOLDER / "test_logits.npy",
        "test_labels": RESULT_FOLDER / "test_labels.npy",
        "test_filenames": RESULT_FOLDER / "test_filenames.npy",
        "test_variants": RESULT_FOLDER / "test_variants.npy",
    }

    if all(path.exists() for path in cache.values()):
        print("Loading cached confusion-expert embeddings...")
        pair_train_embeddings = np.load(cache["pair_train_embeddings"])
        pair_train_labels = np.load(cache["pair_train_labels"])
        test_embeddings = np.load(cache["test_embeddings"])
        test_logits = np.load(cache["test_logits"])
        test_labels = np.load(cache["test_labels"])
        test_filenames = np.load(cache["test_filenames"])
        test_variants = np.load(cache["test_variants"])
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Loading paper-protocol AST on", device)
        model = ASTForAudioClassification.from_pretrained(
            MODEL_NAME,
            num_labels=NUM_CLASSES,
            id2label=id_to_label,
            label2id=label_to_id,
            ignore_mismatched_sizes=True,
            local_files_only=True,
        )
        model.load_state_dict(
            torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
        )
        model.to(device)

        pair_train_loader = DataLoader(
            ManifestDataset(pair_train_dataframe, class_to_index),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        test_loader = DataLoader(
            ManifestDataset(test_dataframe, class_to_index),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )

        print("Extracting Axe/WoodChop training embeddings...")
        (
            pair_train_embeddings,
            _,
            pair_train_full_labels,
            _,
            _,
        ) = extract(model, pair_train_loader, device)
        pair_train_labels = (
            pair_train_full_labels == woodchop_index
        ).astype(np.int64)

        print("Extracting test embeddings and primary predictions...")
        (
            test_embeddings,
            test_logits,
            test_labels,
            test_filenames,
            test_variants,
        ) = extract(model, test_loader, device)

        np.save(cache["pair_train_embeddings"], pair_train_embeddings)
        np.save(cache["pair_train_labels"], pair_train_labels)
        np.save(cache["test_embeddings"], test_embeddings)
        np.save(cache["test_logits"], test_logits)
        np.save(cache["test_labels"], test_labels)
        np.save(cache["test_filenames"], test_filenames)
        np.save(cache["test_variants"], test_variants)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    primary_predictions = test_logits.argmax(axis=1)
    primary_metrics = metrics(test_labels, primary_predictions)
    gate = np.isin(primary_predictions, pair_indices)
    true_pair_mask = np.isin(test_labels, pair_indices)

    experts = {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "expert",
                    LogisticRegression(C=10.0, max_iter=3000, random_state=SEED),
                ),
            ]
        ),
        "Linear SVM": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("expert", SVC(kernel="linear", C=1.0)),
            ]
        ),
        "RBF SVM C=3": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("expert", SVC(kernel="rbf", C=3.0, gamma="scale")),
            ]
        ),
        "RBF SVM C=10": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("expert", SVC(kernel="rbf", C=10.0, gamma="scale")),
            ]
        ),
        "RBF SVM C=30": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("expert", SVC(kernel="rbf", C=30.0, gamma="scale")),
            ]
        ),
        "Cosine 1-NN": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "expert",
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
                    "expert",
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
                    "expert",
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
        n_splits=5,
        shuffle=True,
        random_state=SEED,
    )
    rows: list[dict] = []
    fitted_experts: dict[str, object] = {}
    fused_predictions_by_name: dict[str, np.ndarray] = {}

    print("\nCONFUSION-EXPERT SWEEP")
    print("----------------------")
    for name, expert in experts.items():
        print("Testing", name, "...")
        cv_scores = cross_val_score(
            expert,
            pair_train_embeddings,
            pair_train_labels,
            cv=cross_validation,
            scoring="f1_macro",
            n_jobs=1,
        )
        expert.fit(pair_train_embeddings, pair_train_labels)

        binary_predictions = expert.predict(test_embeddings)
        mapped_predictions = pair_indices[binary_predictions]
        fused_predictions = primary_predictions.copy()
        fused_predictions[gate] = mapped_predictions[gate]
        fused_metrics = metrics(test_labels, fused_predictions)

        pair_accuracy = accuracy_score(
            test_labels[true_pair_mask],
            fused_predictions[true_pair_mask],
        )
        rows.append(
            {
                "expert": name,
                "cv_pair_macro_f1_mean": cv_scores.mean(),
                "cv_pair_macro_f1_std": cv_scores.std(ddof=1),
                "gated_samples": int(gate.sum()),
                "test_pair_accuracy": pair_accuracy,
                "fused_accuracy": fused_metrics["accuracy"],
                "fused_macro_f1": fused_metrics["macro_f1"],
                "correct_predictions": int(
                    (test_labels == fused_predictions).sum()
                ),
            }
        )
        fitted_experts[name] = expert
        fused_predictions_by_name[name] = fused_predictions

    comparison = pd.DataFrame(rows).sort_values(
        "cv_pair_macro_f1_mean",
        ascending=False,
    )
    comparison.to_csv(
        RESULT_FOLDER / "confusion_expert_comparison.csv",
        index=False,
    )

    display = comparison.copy()
    for column in [
        "cv_pair_macro_f1_mean",
        "cv_pair_macro_f1_std",
        "test_pair_accuracy",
        "fused_accuracy",
        "fused_macro_f1",
    ]:
        display[column] *= 100

    print("\nCONFUSION-EXPERT RESULTS")
    print("------------------------")
    print(
        display[
            [
                "expert",
                "cv_pair_macro_f1_mean",
                "test_pair_accuracy",
                "fused_accuracy",
                "fused_macro_f1",
                "correct_predictions",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.2f}")
    )

    selected_name = str(comparison.iloc[0]["expert"])
    selected_expert = fitted_experts[selected_name]
    selected_predictions = fused_predictions_by_name[selected_name]
    selected_metrics = metrics(test_labels, selected_predictions)
    selected_correct = int((selected_predictions == test_labels).sum())
    primary_correct = int((primary_predictions == test_labels).sum())

    joblib.dump(
        selected_expert,
        RESULT_FOLDER / "selected_axe_woodchop_expert.joblib",
    )

    prediction_table = pd.DataFrame(
        {
            "Original Dataset File Name": test_filenames,
            "Variant": test_variants,
            "True Label": test_labels,
            "Primary Prediction": primary_predictions,
            "Final Prediction": selected_predictions,
            "Expert Activated": gate,
            "Correct": test_labels == selected_predictions,
        }
    )
    prediction_table.to_csv(
        RESULT_FOLDER / "confusion_aware_predictions.csv",
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
        RESULT_FOLDER / "confusion_aware_classification_report.csv"
    )

    matrix = confusion_matrix(test_labels, selected_predictions)
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
    plt.title("FSC22 Confusion-Aware Hierarchical AST")
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    figure.savefig(
        RESULT_FOLDER / "confusion_aware_confusion_matrix.png",
        dpi=300,
    )
    plt.close(figure)

    result = {
        "model": "Confusion-Aware Hierarchical AST",
        "selection_rule": "highest five-fold pair-training CV macro-F1",
        "selected_expert": selected_name,
        "primary_accuracy": float(primary_metrics["accuracy"]),
        "accuracy": float(selected_metrics["accuracy"]),
        "macro_precision": float(selected_metrics["macro_precision"]),
        "macro_recall": float(selected_metrics["macro_recall"]),
        "macro_f1": float(selected_metrics["macro_f1"]),
        "primary_correct_predictions": primary_correct,
        "final_correct_predictions": selected_correct,
        "additional_correct_predictions": selected_correct - primary_correct,
        "test_samples": int(len(test_labels)),
        "expert_activated_samples": int(gate.sum()),
    }
    with open(
        RESULT_FOLDER / "confusion_aware_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(result, file, indent=4)

    print("\nSELECTED CONFUSION-AWARE MODEL")
    print("------------------------------")
    print("Expert:", selected_name)
    print(f"Primary accuracy: {100 * primary_metrics['accuracy']:.2f}%")
    print(f"Final accuracy:   {100 * selected_metrics['accuracy']:.2f}%")
    print(f"Macro precision:  {100 * selected_metrics['macro_precision']:.2f}%")
    print(f"Macro recall:     {100 * selected_metrics['macro_recall']:.2f}%")
    print(f"Macro F1-score:   {100 * selected_metrics['macro_f1']:.2f}%")
    print("Correct predictions:", selected_correct, "/", len(test_labels))
    print("Additional correct predictions:", selected_correct - primary_correct)
    print("Results saved to:", RESULT_FOLDER)
    print("\nCONFUSION-AWARE EVALUATION: PASSED")


if __name__ == "__main__":
    main()
