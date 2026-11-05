import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

AUDIT_CSV = Path("reports/clean_dataset_audit.csv")
if not AUDIT_CSV.exists():
    raise FileNotFoundError("Please run src/task1_2_audit.py first to generate reports/clean_dataset_audit.csv!")

df = pd.read_csv(AUDIT_CSV)
classes = sorted(df["class"].unique())

FIG_DIR = Path("figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

for cls in classes:
    cls_df = df[df["class"] == cls]
    sample_rows = cls_df.sample(n=min(8, len(cls_df)), random_state=42)

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    fig.suptitle(f"Sample Visual Grid — Class: {cls}", fontsize=14, fontweight="bold", y=0.98)

    for ax, (_, row) in zip(axes.flatten(), sample_rows.iterrows()):
        try:
            with Image.open(row["file_path"]) as img:
                ax.imshow(img.convert("RGB"))
            dims = f"{int(row['width'])}x{int(row['height'])}"
            size_kb = f"{row['file_size_kb']:.1f} KB"
            fmt = str(row.get("format", "IMG"))
            ax.set_title(f"{dims} | {fmt}\n{size_kb}", fontsize=9)
        except Exception as e:
            ax.set_title("Load Error", fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    output_filename = FIG_DIR / f"task1_3_grid_{cls.lower()}.png"
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Generated] {output_filename}")

hist_configs = [
    ("width", "Image Width (Pixels)", "task1_3_hist_width.png"),
    ("height", "Image Height (Pixels)", "task1_3_hist_height.png"),
    ("file_size_kb", "File Size (KB)", "task1_3_hist_filesize.png"),
]

palette = {"#1f77b4", "#ff7f0e"}  
colors = ["#2b5c8f", "#d95f02"]

for col_name, label_title, fname in hist_configs:
    fig, ax = plt.subplots(figsize=(8, 5))

    for idx, cls in enumerate(classes):
        subset = df[df["class"] == cls][col_name].dropna()
        p99 = subset.quantile(0.99)
        filtered_subset = subset[subset <= p99]

        ax.hist(
            filtered_subset,
            bins=35,
            alpha=0.55,
            label=f"{cls} (n={len(subset)})",
            color=colors[idx % len(colors)],
            edgecolor="black",
            linewidth=0.8
        )

    ax.set_title(f"Distribution Comparison: {label_title}", fontweight="bold", pad=12)
    ax.set_xlabel(label_title)
    ax.set_ylabel("Image Count (Frequency)")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(frameon=True, loc="upper right")

    plt.tight_layout()
    hist_out = FIG_DIR / fname
    plt.savefig(hist_out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Generated] {hist_out}")

print("\nAll figures generated successfully inside 'figures/' directory.")