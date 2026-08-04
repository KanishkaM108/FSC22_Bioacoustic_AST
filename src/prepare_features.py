from pathlib import Path
import json
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

root = Path.cwd()
split_file = root / "outputs" / "paper_split_seed42.csv"
cache_folder = root / "outputs" / "mel_cache"
cache_folder.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 22050
DURATION = 5
NUMBER_OF_SAMPLES = SAMPLE_RATE * DURATION
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512

metadata = pd.read_csv(split_file)

wav_paths = {
    wav_path.name: wav_path
    for wav_path in root.rglob("*.wav")
}

missing = []

for filename in tqdm(
    metadata["Dataset File Name"],
    desc="Creating Mel spectrograms"
):
    audio_path = wav_paths.get(filename)

    if audio_path is None:
        missing.append(filename)
        continue

    output_path = cache_folder / f"{Path(filename).stem}.npy"

    if output_path.exists():
        continue

    audio, _ = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    audio = librosa.util.fix_length(
        data=audio,
        size=NUMBER_OF_SAMPLES
    )

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=20,
        fmax=SAMPLE_RATE // 2,
        power=2.0
    )

    log_mel = librosa.power_to_db(
        mel,
        ref=np.max
    )

    np.save(output_path, log_mel.astype(np.float32))

config = {
    "sample_rate": SAMPLE_RATE,
    "duration_seconds": DURATION,
    "n_mels": N_MELS,
    "n_fft": N_FFT,
    "hop_length": HOP_LENGTH,
    "feature_shape": list(
        np.load(next(cache_folder.glob("*.npy"))).shape
    )
}

with open(
    root / "outputs" / "feature_config.json",
    "w"
) as file:
    json.dump(config, file, indent=4)

created_files = list(cache_folder.glob("*.npy"))

print("\nFEATURE PREPARATION")
print("-------------------")
print("Spectrogram files:", len(created_files))
print("Missing audio files:", len(missing))
print("Feature shape:", config["feature_shape"])
print("Saved inside:", cache_folder)

if len(created_files) == 2025 and not missing:
    print("\nFEATURE PREPARATION: PASSED")
else:
    print("\nFEATURE PREPARATION: FAILED")