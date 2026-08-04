"""Prepare a leakage-free FSC22 train/validation/test protocol.

The split is made on the underlying FreeSound source identifier before any
augmentation is attached.  Consequently, neither a clip, an augmented view of
that clip, nor another segment from the same source recording can cross split
boundaries.

Existing AST caches produced by ``prepare_ast_features.py`` and
``prepare_paper_protocol.py`` are reused.  Run those preparation scripts first
if the cache files do not yet exist.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


ROOT = Path.cwd()
DEFAULT_METADATA = (
    ROOT
    / "Metadata-20220916T202011Z-001"
    / "Metadata"
    / "Metadata V1.0 FSC22.csv"
)
ORIGINAL_AST_CACHE = ROOT / "outputs" / "ast_cache"
PITCH_CACHE = ROOT / "outputs" / "ast_pitch_cache"
DEFAULT_MANIFEST = ROOT / "outputs" / "clean_grouped_manifest_seed42.csv"
DEFAULT_PROTOCOL = ROOT / "outputs" / "clean_grouped_protocol_seed42.json"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a source-disjoint FSC22 split before augmentation"
    )
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outer-fold", type=int, default=0, choices=range(5))
    parser.add_argument("--inner-fold", type=int, default=0, choices=range(5))
    parser.add_argument("--output", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--protocol-output", type=Path, default=DEFAULT_PROTOCOL)
    return parser.parse_args()


def source_group_id(source_file_name: str) -> str:
    """Return the recording-level FreeSound identifier.

    FSC22 names commonly look like ``17548_A.wav``.  The trailing alphabetic
    segment identifies a cut/segment, while ``17548`` identifies the source
    recording.  Unknown naming patterns conservatively retain the full stem.
    """

    stem = Path(str(source_file_name)).stem.strip()
    match = re.fullmatch(r"(.+)_([A-Za-z]+)", stem)
    return match.group(1) if match else stem


def make_split(
    dataframe: pd.DataFrame,
    *,
    seed: int,
    outer_fold: int,
    inner_fold: int,
) -> pd.Series:
    labels = dataframe["Class ID"].astype(int).to_numpy()
    groups = dataframe["Source Group ID"].astype(str).to_numpy()

    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    outer_splits = list(outer.split(dataframe, labels, groups))
    train_validation_indices, test_indices = outer_splits[outer_fold]

    train_validation = dataframe.iloc[train_validation_indices]
    inner_labels = train_validation["Class ID"].astype(int).to_numpy()
    inner_groups = train_validation["Source Group ID"].astype(str).to_numpy()
    inner = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=seed + 1,
    )
    inner_splits = list(
        inner.split(train_validation, inner_labels, inner_groups)
    )
    train_relative, validation_relative = inner_splits[inner_fold]
    train_indices = train_validation_indices[train_relative]
    validation_indices = train_validation_indices[validation_relative]

    split = pd.Series("", index=dataframe.index, dtype="object")
    split.iloc[train_indices] = "train"
    split.iloc[validation_indices] = "validation"
    split.iloc[test_indices] = "test"
    if (split == "").any():
        raise RuntimeError("At least one recording was not assigned to a split.")
    return split


def relative_path(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))


def main() -> None:
    arguments = parse_arguments()
    metadata_path = arguments.metadata.resolve()
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    metadata = pd.read_csv(metadata_path)
    required = {
        "Source File Name",
        "Dataset File Name",
        "Class ID",
        "Class Name",
    }
    missing = required.difference(metadata.columns)
    if missing:
        raise KeyError(f"Metadata is missing columns: {sorted(missing)}")
    metadata = metadata[list(required)].copy()
    metadata["Class ID"] = pd.to_numeric(metadata["Class ID"]).astype(int)
    metadata["Source Group ID"] = metadata["Source File Name"].map(
        source_group_id
    )
    metadata["Original ID"] = metadata["Dataset File Name"].astype(str).map(
        lambda value: Path(value).stem
    )

    if metadata["Original ID"].duplicated().any():
        raise RuntimeError("Dataset File Name stems must uniquely identify clips.")
    # A FreeSound recording can legitimately contain several labelled acoustic
    # events in different extracted segments.  This does not invalidate the
    # group: every segment from that recording must remain in the same split.
    source_label_counts = metadata.groupby("Source Group ID")["Class ID"].nunique()
    multi_class_source_groups = int((source_label_counts > 1).sum())

    metadata["Split"] = make_split(
        metadata,
        seed=arguments.seed,
        outer_fold=arguments.outer_fold,
        inner_fold=arguments.inner_fold,
    )

    missing_features: list[str] = []
    rows: list[dict[str, object]] = []
    for _, row in metadata.iterrows():
        original_id = str(row["Original ID"])
        common = {
            "Source File Name": row["Source File Name"],
            "Source Group ID": row["Source Group ID"],
            "Original Dataset File Name": row["Dataset File Name"],
            "Original ID": original_id,
            "Class ID": int(row["Class ID"]),
            "Class Name": row["Class Name"],
            "Split": row["Split"],
        }
        variants = [
            ("original", ORIGINAL_AST_CACHE / f"{original_id}.npy"),
            ("pitch_down_2", PITCH_CACHE / f"{original_id}__pitch_down2.npy"),
            ("pitch_up_2", PITCH_CACHE / f"{original_id}__pitch_up2.npy"),
        ]
        for variant, feature_path in variants:
            if not feature_path.exists():
                missing_features.append(str(feature_path))
            rows.append(
                {
                    **common,
                    "Variant": variant,
                    "Feature Path": relative_path(feature_path),
                }
            )

    if missing_features:
        preview = "\n".join(missing_features[:5])
        raise FileNotFoundError(
            f"Missing {len(missing_features)} AST feature files. Run "
            f"prepare_ast_features.py and prepare_paper_protocol.py first.\n{preview}"
        )

    manifest = pd.DataFrame(rows)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(arguments.output, index=False)

    split_groups = {
        split: set(
            metadata.loc[metadata["Split"] == split, "Source Group ID"].astype(str)
        )
        for split in ("train", "validation", "test")
    }
    overlaps = {
        "train_validation": len(split_groups["train"] & split_groups["validation"]),
        "train_test": len(split_groups["train"] & split_groups["test"]),
        "validation_test": len(split_groups["validation"] & split_groups["test"]),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"Source leakage detected: {overlaps}")

    clip_counts = metadata.groupby("Split").size().to_dict()
    variant_counts = manifest.groupby("Split").size().to_dict()
    class_counts = (
        metadata.groupby(["Split", "Class Name"]).size().rename("clips").reset_index()
    )
    split_class_ranges = {
        split: {
            "minimum": int(part["clips"].min()),
            "maximum": int(part["clips"].max()),
        }
        for split, part in class_counts.groupby("Split")
    }
    protocol = {
        "name": "FSC22 clean source-grouped protocol",
        "split_unit": "FreeSound source recording derived from Source File Name",
        "split_method": "nested StratifiedGroupKFold (outer test, inner validation)",
        "seed": arguments.seed,
        "outer_fold": arguments.outer_fold,
        "inner_fold": arguments.inner_fold,
        "original_clips": len(metadata),
        "source_groups": int(metadata["Source Group ID"].nunique()),
        "multi_class_source_groups": multi_class_source_groups,
        "clip_counts": {key: int(value) for key, value in clip_counts.items()},
        "variant_counts": {key: int(value) for key, value in variant_counts.items()},
        "class_count_ranges": split_class_ranges,
        "source_group_overlaps": overlaps,
        "augmentation_policy": (
            "All variants inherit the split of their source recording; "
            "training uses variants, validation/test variants are TTA only"
        ),
        "test_lock": (
            "Test labels must not be used for model selection, calibration, "
            "thresholding, class weighting, or ensemble weighting"
        ),
    }
    arguments.protocol_output.parent.mkdir(parents=True, exist_ok=True)
    arguments.protocol_output.write_text(
        json.dumps(protocol, indent=4), encoding="utf-8"
    )

    print("\nFSC22 CLEAN SOURCE-GROUPED PROTOCOL")
    print("-----------------------------------")
    print("Original clips:", len(metadata))
    print("Independent source groups:", metadata["Source Group ID"].nunique())
    print("Source groups containing multiple labelled events:", multi_class_source_groups)
    for split in ("train", "validation", "test"):
        print(
            f"{split.title():10s}: {clip_counts.get(split, 0):4d} clips, "
            f"{variant_counts.get(split, 0):4d} feature rows, "
            f"{len(split_groups[split]):4d} source groups"
        )
    print("Source-group overlaps:", overlaps)
    print("Manifest saved to:", arguments.output)
    print("Protocol saved to:", arguments.protocol_output)
    print("\nCLEAN GROUPED PROTOCOL: PASSED")


if __name__ == "__main__":
    main()
