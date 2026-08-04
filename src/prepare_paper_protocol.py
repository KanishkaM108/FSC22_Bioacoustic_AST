"""Create the augmentation-before-split protocol described by the FSC22 paper."""

from __future__ import annotations

import json
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import ASTFeatureExtractor


MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
SAMPLE_RATE = 16000
PITCH_STEPS = (-2.0, 2.0)
SEED = 42

ROOT = Path.cwd()
ORIGINAL_SPLIT_FILE = ROOT / "outputs" / "paper_split_seed42.csv"
ORIGINAL_AST_CACHE = ROOT / "outputs" / "ast_cache"
PITCH_CACHE = ROOT / "outputs" / "ast_pitch_cache"
OUTPUT_MANIFEST = ROOT / "outputs" / "paper_augmented_split_seed42.csv"
PITCH_CACHE.mkdir(parents=True, exist_ok=True)


def relative_path(path: Path) -> str:
    return str(path.relative_to(ROOT))


def main() -> None:
    if not ORIGINAL_SPLIT_FILE.exists():
        raise FileNotFoundError(
            f"Original metadata split not found: {ORIGINAL_SPLIT_FILE}"
        )
    if len(list(ORIGINAL_AST_CACHE.glob("*.npy"))) != 2025:
        raise FileNotFoundError("Expected 2025 original cached AST features.")

    metadata = pd.read_csv(ORIGINAL_SPLIT_FILE)
    metadata = metadata.drop(columns=["Split"], errors="ignore")
    metadata["Class ID"] = pd.to_numeric(metadata["Class ID"]).astype(int)

    wav_paths = {path.name: path for path in ROOT.rglob("*.wav")}
    if len(wav_paths) != 2025:
        raise RuntimeError(f"Expected 2025 WAV files, found {len(wav_paths)}.")

    feature_extractor = ASTFeatureExtractor.from_pretrained(
        MODEL_NAME,
        local_files_only=True,
    )

    print("\nCreating pitch-shifted AST features...")
    missing_files: list[str] = []

    for filename in tqdm(
        metadata["Dataset File Name"].astype(str),
        total=len(metadata),
        desc="Pitch augmentation",
    ):
        audio_path = wav_paths.get(filename)
        if audio_path is None:
            missing_files.append(filename)
            continue

        stem = Path(filename).stem
        up_path = PITCH_CACHE / f"{stem}__pitch_up2.npy"
        down_path = PITCH_CACHE / f"{stem}__pitch_down2.npy"

        if up_path.exists() and down_path.exists():
            continue

        waveform, _ = librosa.load(
            audio_path,
            sr=SAMPLE_RATE,
            mono=True,
        )

        for pitch_steps, output_path in [
            (PITCH_STEPS[0], down_path),
            (PITCH_STEPS[1], up_path),
        ]:
            if output_path.exists():
                continue

            shifted_waveform = librosa.effects.pitch_shift(
                y=waveform,
                sr=SAMPLE_RATE,
                n_steps=pitch_steps,
            )
            extracted = feature_extractor(
                shifted_waveform,
                sampling_rate=SAMPLE_RATE,
                return_tensors="np",
            )
            np.save(
                output_path,
                extracted["input_values"][0].astype(np.float16),
            )

    if missing_files:
        raise FileNotFoundError(
            f"Could not locate {len(missing_files)} audio files."
        )

    pitch_files = list(PITCH_CACHE.glob("*.npy"))
    if len(pitch_files) != 4050:
        raise RuntimeError(
            f"Expected 4050 pitch features, found {len(pitch_files)}."
        )

    manifest_rows: list[dict] = []

    for _, row in metadata.iterrows():
        filename = str(row["Dataset File Name"])
        stem = Path(filename).stem
        common = {
            "Source File Name": row["Source File Name"],
            "Original Dataset File Name": filename,
            "Original ID": stem,
            "Class ID": int(row["Class ID"]),
            "Class Name": row["Class Name"],
        }

        manifest_rows.extend(
            [
                {
                    **common,
                    "Variant": "original",
                    "Feature Path": relative_path(
                        ORIGINAL_AST_CACHE / f"{stem}.npy"
                    ),
                },
                {
                    **common,
                    "Variant": "pitch_down_2",
                    "Feature Path": relative_path(
                        PITCH_CACHE / f"{stem}__pitch_down2.npy"
                    ),
                },
                {
                    **common,
                    "Variant": "pitch_up_2",
                    "Feature Path": relative_path(
                        PITCH_CACHE / f"{stem}__pitch_up2.npy"
                    ),
                },
            ]
        )

    manifest = pd.DataFrame(manifest_rows)

    # This intentionally follows the paper's described order: augment first,
    # then stratify all resulting samples into an 80/20 split.
    train_indices, test_indices = train_test_split(
        manifest.index,
        test_size=0.20,
        random_state=SEED,
        stratify=manifest["Class ID"],
    )
    manifest["Split"] = "train"
    manifest.loc[test_indices, "Split"] = "test"
    manifest.to_csv(OUTPUT_MANIFEST, index=False)

    train_data = manifest[manifest["Split"] == "train"]
    test_data = manifest[manifest["Split"] == "test"]
    train_original_ids = set(train_data["Original ID"])
    test_original_ids = set(test_data["Original ID"])
    overlapping_originals = train_original_ids.intersection(test_original_ids)

    cache_size_mb = sum(path.stat().st_size for path in pitch_files) / (1024**2)
    configuration = {
        "protocol": "augmentation before stratified 80/20 split",
        "pitch_steps": list(PITCH_STEPS),
        "random_seed": SEED,
        "total_samples": len(manifest),
        "training_samples": len(train_data),
        "testing_samples": len(test_data),
        "overlapping_original_ids": len(overlapping_originals),
        "additional_cache_size_mb": round(cache_size_mb, 2),
    }
    with open(
        ROOT / "outputs" / "paper_augmented_protocol.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(configuration, file, indent=4)

    train_counts = train_data.groupby("Class ID").size()
    test_counts = test_data.groupby("Class ID").size()

    print("\nPAPER-COMPATIBLE AUGMENTED PROTOCOL")
    print("-----------------------------------")
    print("Original recordings:", len(metadata))
    print("Pitch-shifted features:", len(pitch_files))
    print("Total augmented samples:", len(manifest))
    print("Training samples:", len(train_data))
    print("Testing samples:", len(test_data))
    print("Training samples per class:", train_counts.min())
    print("Testing samples per class:", test_counts.min())
    print("Original IDs occurring in both splits:", len(overlapping_originals))
    print(f"Additional cache size: {cache_size_mb:.1f} MB")
    print("Manifest saved to:", OUTPUT_MANIFEST)

    valid = (
        len(manifest) == 6075
        and len(train_data) == 4860
        and len(test_data) == 1215
        and train_counts.min() == 180
        and train_counts.max() == 180
        and test_counts.min() == 45
        and test_counts.max() == 45
    )
    print(
        "\nPAPER PROTOCOL PREPARATION:",
        "PASSED" if valid else "FAILED",
    )
if __name__ == "__main__":
    main()
