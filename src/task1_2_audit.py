import os
import hashlib
from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np

DATASET_DIR = Path("data/dataset")
OUTPUT_CSV = Path("reports/dataset_audit.csv")

def compute_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def audit_dataset(root_dir: Path):
    records = []
    corrupt_files = []

    # Assuming class folders are inside root_dir
    class_dirs = [d for d in root_dir.iterdir() if d.is_dir()]
    if not class_dirs:
        raise FileNotFoundError(f"No class subdirectories found in {root_dir}. Check your path!")

    for class_dir in sorted(class_dirs):
        label = class_dir.name
        for file_path in class_dir.rglob("*"):
            if file_path.is_dir():
                continue
            
            file_size_kb = file_path.stat().st_size / 1024.0
            sha256_hash = compute_sha256(file_path)

            try:
                with Image.open(file_path) as img:
                    img.verify()
                with Image.open(file_path) as img:
                    img.load()
                    width, height = img.size
                    mode = img.mode
                    fmt = img.format
                
                records.append({
                    "file_path": str(file_path),
                    "class": label,
                    "file_size_kb": round(file_size_kb, 2),
                    "width": width,
                    "height": height,
                    "colour_mode": mode,
                    "format": fmt,
                    "sha256": sha256_hash,
                    "is_corrupt": False
                })
            except Exception as e:
                corrupt_files.append((str(file_path), label, str(e)))
                records.append({
                    "file_path": str(file_path),
                    "class": label,
                    "file_size_kb": round(file_size_kb, 2),
                    "width": np.nan,
                    "height": np.nan,
                    "colour_mode": None,
                    "format": None,
                    "sha256": sha256_hash,
                    "is_corrupt": True
                })

    df = pd.DataFrame(records)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    valid_df = df[~df["is_corrupt"]].copy()
    
    # Duplicate detection based on exact SHA-256 content hash
    duplicate_mask = valid_df.duplicated(subset=["sha256"], keep="first")
    num_duplicates = duplicate_mask.sum()

    print("DATASET AUDIT SUMMARY")
    print(f"Total Files Scanned : {len(df)}")
    print(f"Unreadable/Corrupt  : {len(corrupt_files)}")
    print(f"Exact Duplicates    : {num_duplicates}")

    print("\n1. COUNTS PER CLASS (VALID IMAGES):")
    print(valid_df["class"].value_counts().to_frame(name="image_count"))

    print("\n2. COLOUR MODES & FORMATS:")
    print(pd.crosstab(valid_df["class"], valid_df["colour_mode"]))
    print()
    print(pd.crosstab(valid_df["class"], valid_df["format"]))

    print("\n3. MIN, MEDIAN, AND MAX OF WIDTH, HEIGHT, AND FILE SIZE (KB) PER CLASS:")
    stats = valid_df.groupby("class").agg(
        width_min=("width", "min"),
        width_median=("width", "median"),
        width_max=("width", "max"),
        height_min=("height", "min"),
        height_median=("height", "median"),
        height_max=("height", "max"),
        size_kb_min=("file_size_kb", "min"),
        size_kb_median=("file_size_kb", "median"),
        size_kb_max=("file_size_kb", "max"),
    )
    print(stats.to_string())

    # Save clean deduplicated dataframe for Task 1.3 and Task 2
    clean_df = valid_df[~duplicate_mask].copy()
    clean_df.to_csv("reports/clean_dataset_audit.csv", index=False)
    print(f"Clean deduplicated manifest saved to reports/clean_dataset_audit.csv ({len(clean_df)} images)")

if __name__ == "__main__":
    audit_dataset(DATASET_DIR)