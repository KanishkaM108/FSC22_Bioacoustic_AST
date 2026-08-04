"""Prepare fixed-length FSC22 waveforms for WavLM fine-tuning.

Run from the FSC22_Research project root:

    python src\\prepare_wavlm_audio.py

The script reuses the existing paper-compatible augmented manifest, loads each
of the 2,025 original WAV files at 16 kHz, creates -2/+2 semitone variants, and
caches all 6,075 waveforms as float16 NumPy arrays. Expected additional disk
usage is approximately 1 GB. No internet connection is required.
"""

from __future__ import annotations

import json
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm


SAMPLE_RATE = 16000
DURATION_SECONDS = 5
TARGET_SAMPLES = SAMPLE_RATE * DURATION_SECONDS
PITCH_STEPS = {
    "pitch_down_2": -2.0,
    "pitch_up_2": 2.0,
}

ROOT = Path.cwd()
INPUT_MANIFEST = ROOT / "outputs" / "paper_augmented_split_seed42.csv"
AUDIO_CACHE = ROOT / "outputs" / "wavlm_audio_cache"
OUTPUT_MANIFEST = ROOT / "outputs" / "wavlm_augmented_split_seed42.csv"
SUMMARY_FILE = ROOT / "outputs" / "wavlm_audio_preparation.json"
AUDIO_CACHE.mkdir(parents=True, exist_ok=True)


def fixed_length(waveform: np.ndarray) -> np.ndarray:
    waveform = np.asarray(waveform, dtype=np.float32)
    if len(waveform) >= TARGET_SAMPLES:
        return waveform[:TARGET_SAMPLES]
    return np.pad(waveform, (0, TARGET_SAMPLES - len(waveform)))


def relative_path(path: Path) -> str:
    return str(path.relative_to(ROOT))


def cache_paths(stem: str) -> dict[str, Path]:
    return {
        "original": AUDIO_CACHE / f"{stem}__original.npy",
        "pitch_down_2": AUDIO_CACHE / f"{stem}__pitch_down2.npy",
        "pitch_up_2": AUDIO_CACHE / f"{stem}__pitch_up2.npy",
    }


def main() -> None:
    if not INPUT_MANIFEST.exists():
        raise FileNotFoundError(f"Manifest not found: {INPUT_MANIFEST}")

    manifest = pd.read_csv(INPUT_MANIFEST)
    required_columns = {
        "Original Dataset File Name",
        "Original ID",
        "Variant",
        "Class ID",
        "Class Name",
        "Split",
    }
    missing_columns = required_columns.difference(manifest.columns)
    if missing_columns:
        raise KeyError(f"Manifest is missing: {sorted(missing_columns)}")

    originals = (
        manifest[
            [
                "Original Dataset File Name",
                "Original ID",
            ]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    if len(originals) != 2025:
        raise RuntimeError(f"Expected 2025 originals, found {len(originals)}.")

    # Ignore any non-dataset WAV with the same name only if duplicates appear;
    # FSC22 validation previously established one unique file per dataset name.
    wav_candidates: dict[str, list[Path]] = {}
    for path in ROOT.rglob("*.wav"):
        wav_candidates.setdefault(path.name, []).append(path)

    missing_audio: list[str] = []
    duplicate_audio: list[str] = []
    for filename in originals["Original Dataset File Name"].astype(str):
        candidates = wav_candidates.get(filename, [])
        if not candidates:
            missing_audio.append(filename)
        elif len(candidates) > 1:
            duplicate_audio.append(filename)

    if missing_audio:
        raise FileNotFoundError(
            f"Could not locate {len(missing_audio)} original WAV files."
        )
    if duplicate_audio:
        raise RuntimeError(
            f"Found duplicate paths for {len(duplicate_audio)} WAV filenames."
        )

    print("\nPREPARING WAVLM AUDIO CACHE")
    print("---------------------------")
    print("Original recordings:", len(originals))
    print("Target sample rate:", SAMPLE_RATE)
    print("Samples per waveform:", TARGET_SAMPLES)
    print("Expected cached waveforms:", 3 * len(originals))
    print("Existing files will be reused.")

    for row in tqdm(
        originals.itertuples(index=False),
        total=len(originals),
        desc="WavLM audio preparation",
        unit="recording",
    ):
        filename = str(row[0])
        stem = str(row[1])
        outputs = cache_paths(stem)
        if all(path.exists() for path in outputs.values()):
            continue

        audio_path = wav_candidates[filename][0]
        waveform, _ = librosa.load(
            audio_path,
            sr=SAMPLE_RATE,
            mono=True,
        )
        if waveform.size == 0:
            raise ValueError(f"Empty waveform: {audio_path}")

        if not outputs["original"].exists():
            np.save(
                outputs["original"],
                fixed_length(waveform).astype(np.float16),
            )

        for variant, pitch_steps in PITCH_STEPS.items():
            output_path = outputs[variant]
            if output_path.exists():
                continue
            shifted = librosa.effects.pitch_shift(
                y=waveform,
                sr=SAMPLE_RATE,
                n_steps=pitch_steps,
            )
            np.save(
                output_path,
                fixed_length(shifted).astype(np.float16),
            )

    cached_files = list(AUDIO_CACHE.glob("*.npy"))
    expected_files = 3 * len(originals)
    if len(cached_files) != expected_files:
        raise RuntimeError(
            f"Expected {expected_files} cached files, found {len(cached_files)}."
        )

    audio_paths: list[str] = []
    for _, row in manifest.iterrows():
        stem = str(row["Original ID"])
        variant = str(row["Variant"])
        paths = cache_paths(stem)
        if variant not in paths:
            raise ValueError(f"Unexpected variant: {variant}")
        audio_paths.append(relative_path(paths[variant]))

    output_manifest = manifest.copy()
    output_manifest["Audio Path"] = audio_paths
    output_manifest.to_csv(OUTPUT_MANIFEST, index=False)

    training = output_manifest[output_manifest["Split"] == "train"]
    testing = output_manifest[output_manifest["Split"] == "test"]
    cache_size_mb = sum(path.stat().st_size for path in cached_files) / (1024**2)
    valid_arrays = 0
    for path in cached_files:
        array = np.load(path, mmap_mode="r")
        if array.shape == (TARGET_SAMPLES,) and array.dtype == np.float16:
            valid_arrays += 1

    summary = {
        "sample_rate": SAMPLE_RATE,
        "duration_seconds": DURATION_SECONDS,
        "samples_per_waveform": TARGET_SAMPLES,
        "pitch_steps": PITCH_STEPS,
        "original_recordings": int(len(originals)),
        "cached_waveforms": int(len(cached_files)),
        "valid_cached_waveforms": int(valid_arrays),
        "training_samples": int(len(training)),
        "testing_samples": int(len(testing)),
        "cache_size_mb": round(cache_size_mb, 2),
        "output_manifest": str(OUTPUT_MANIFEST),
    }
    with open(SUMMARY_FILE, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    passed = (
        len(cached_files) == 6075
        and valid_arrays == 6075
        and len(training) == 4860
        and len(testing) == 1215
    )
    print("\nWAVLM AUDIO PREPARATION")
    print("------------------------")
    print("Cached waveforms:", len(cached_files))
    print("Valid waveform arrays:", valid_arrays)
    print("Training samples:", len(training))
    print("Testing samples:", len(testing))
    print(f"Cache size: {cache_size_mb:.1f} MB")
    print("Manifest saved to:", OUTPUT_MANIFEST)
    print("\nWAVLM AUDIO PREPARATION:", "PASSED" if passed else "FAILED")


if __name__ == "__main__":
    main()
