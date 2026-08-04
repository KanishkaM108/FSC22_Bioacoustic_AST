from pathlib import Path
import pandas as pd

root = Path.cwd()
wav_files = list(root.rglob("*.wav"))
csv_files = list(root.rglob("*.csv"))

metadata = None
metadata_path = None

for csv_path in csv_files:
    try:
        df = pd.read_csv(csv_path)
        df.columns = [
            str(column).strip().lstrip("\ufeff")
            for column in df.columns
        ]

        required = {"Source File Name", "Dataset File Name",
                    "Class ID", "Class Name"}

        if required.issubset(df.columns):
            metadata = df
            metadata_path = csv_path
            break
    except Exception:
        continue

if metadata is None:
    raise FileNotFoundError("FSC22 metadata CSV was not found.")

metadata["Dataset File Name"] = (
    metadata["Dataset File Name"].astype(str).str.strip()
)
metadata["Class ID"] = pd.to_numeric(
    metadata["Class ID"], errors="raise"
).astype(int)

actual_files = {file.name for file in wav_files}
expected_files = set(metadata["Dataset File Name"])

missing_files = expected_files - actual_files
extra_files = actual_files - expected_files
class_counts = metadata.groupby(
    ["Class ID", "Class Name"]
).size()

print("\nFSC22 DATASET VALIDATION")
print("------------------------")
print("Metadata file:", metadata_path)
print("Metadata rows:", len(metadata))
print("WAV files:", len(wav_files))
print("Number of classes:", metadata["Class ID"].nunique())
print("Minimum samples per class:", class_counts.min())
print("Maximum samples per class:", class_counts.max())
print("Missing audio files:", len(missing_files))
print("Unexpected audio files:", len(extra_files))

valid = (
    len(metadata) == 2025
    and len(wav_files) == 2025
    and metadata["Class ID"].nunique() == 27
    and class_counts.min() == 75
    and class_counts.max() == 75
    and not missing_files
    and not extra_files
)

print("\nVALIDATION:", "PASSED" if valid else "FAILED")