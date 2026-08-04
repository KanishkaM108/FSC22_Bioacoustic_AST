from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

root = Path.cwd()

# Find the correct metadata CSV
metadata = None

for csv_path in root.rglob("*.csv"):
    try:
        df = pd.read_csv(csv_path)
        df.columns = [
            str(column).strip().lstrip("\ufeff")
            for column in df.columns
        ]

        required = {
            "Source File Name",
            "Dataset File Name",
            "Class ID",
            "Class Name",
        }

        if required.issubset(df.columns):
            metadata = df
            break
    except Exception:
        continue

if metadata is None:
    raise FileNotFoundError("FSC22 metadata CSV not found.")

metadata["Class ID"] = pd.to_numeric(
    metadata["Class ID"]
).astype(int)

# Reproducible stratified 80/20 split
train_indices, test_indices = train_test_split(
    metadata.index,
    test_size=0.20,
    random_state=42,
    stratify=metadata["Class ID"],
)

metadata["Split"] = "train"
metadata.loc[test_indices, "Split"] = "test"

output_folder = root / "outputs"
output_folder.mkdir(exist_ok=True)

output_file = output_folder / "paper_split_seed42.csv"
metadata.to_csv(output_file, index=False)

train_data = metadata[metadata["Split"] == "train"]
test_data = metadata[metadata["Split"] == "test"]

print("\nFSC22 PAPER-COMPATIBLE SPLIT")
print("----------------------------")
print("Training samples:", len(train_data))
print("Testing samples:", len(test_data))
print("Training samples per class:",
      train_data.groupby("Class ID").size().min())
print("Testing samples per class:",
      test_data.groupby("Class ID").size().min())
print("Saved to:", output_file)
print("\nSPLIT CREATION: PASSED")