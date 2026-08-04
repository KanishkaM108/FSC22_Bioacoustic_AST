"""Precompute Audio Spectrogram Transformer inputs for FSC22."""

from __future__ import annotations

import json
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import ASTFeatureExtractor


MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
SAMPLE_RATE = 16000
DURATION_SECONDS = 5.0

ROOT = Path.cwd()
SPLIT_FILE = ROOT / "outputs" / "paper_split_seed42.csv"
CACHE_FOLDER = ROOT / "outputs" / "ast_cache"
CACHE_FOLDER.mkdir(parents=True, exist_ok=True)


def main() -> None:
    if not SPLIT_FILE.exists():
        raise FileNotFoundError(f"Split file not found: {SPLIT_FILE}")

    metadata = pd.read_csv(SPLIT_FILE)
    wav_paths = {path.name: path for path in ROOT.rglob("*.wav")}

    if len(wav_paths) != 2025:
        raise RuntimeError(f"Expected 2025 WAV files, found {len(wav_paths)}.")

    # The checkpoint was downloaded in the previous step, so this avoids
    # another network request.
    feature_extractor = ASTFeatureExtractor.from_pretrained(
        MODEL_NAME,
        local_files_only=True,
    )

    missing_files: list[str] = []

    for filename in tqdm(
        metadata["Dataset File Name"].astype(str),
        desc="Creating AST features",
    ):
        audio_path = wav_paths.get(filename)
        output_path = CACHE_FOLDER / f"{Path(filename).stem}.npy"

        if audio_path is None:
            missing_files.append(filename)
            continue

        if output_path.exists():
            continue

        waveform, _ = librosa.load(
            audio_path,
            sr=SAMPLE_RATE,
            mono=True,
            duration=DURATION_SECONDS,
        )

        extracted = feature_extractor(
            waveform,
            sampling_rate=SAMPLE_RATE,
            return_tensors="np",
        )
        input_values = extracted["input_values"][0]

        # Float16 halves the cache size. Values are converted back to float32
        # by the training dataset before being sent to the model.
        np.save(output_path, input_values.astype(np.float16))

    cached_files = list(CACHE_FOLDER.glob("*.npy"))

    if not cached_files:
        raise RuntimeError("No AST feature files were created.")

    example_shape = list(np.load(cached_files[0]).shape)
    cache_size_mb = sum(path.stat().st_size for path in cached_files) / (1024**2)

    config = {
        "checkpoint": MODEL_NAME,
        "sample_rate": SAMPLE_RATE,
        "duration_seconds": DURATION_SECONDS,
        "feature_shape": example_shape,
        "feature_dtype": "float16",
        "number_of_files": len(cached_files),
        "cache_size_mb": round(cache_size_mb, 2),
    }

    with open(
        ROOT / "outputs" / "ast_feature_config.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(config, file, indent=4)

    print("\nAST FEATURE PREPARATION")
    print("-----------------------")
    print("AST feature files:", len(cached_files))
    print("Missing audio files:", len(missing_files))
    print("Feature shape:", example_shape)
    print(f"Cache size: {cache_size_mb:.1f} MB")
    print("Saved inside:", CACHE_FOLDER)

    if len(cached_files) == 2025 and not missing_files:
        print("\nAST FEATURE PREPARATION: PASSED")
    else:
        print("\nAST FEATURE PREPARATION: FAILED")


if __name__ == "__main__":
    main()
