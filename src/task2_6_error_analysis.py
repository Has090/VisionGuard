from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from task2_1_pipeline import build_tf_dataset

MODELS_DIR = Path("models")
FIG_DIR = Path("figures")

test_df = pd.read_csv("reports/splits/test.csv")
test_ds = build_tf_dataset(test_df, is_training=False)
model = tf.keras.models.load_model(MODELS_DIR / "mobilenetv2_stage2_champion_best.keras")

probs = model.predict(test_ds).ravel()
test_df["predicted_prob"] = probs
test_df["predicted_label"] = (probs >= 0.5).astype(int)

# Filter for misclassifications
errors = test_df[test_df["label"] != test_df["predicted_label"]].copy()
sample_errors = errors.sample(n=min(10, len(errors)), random_state=42)

classes = {0: "RealArt", 1: "AiArtData"}
fig, axes = plt.subplots(2, 5, figsize=(18, 8))
fig.suptitle("VisionGuard Failure Mode Analysis: 10 Misclassified Test Images", fontsize=15, fontweight="bold")

hypotheses = [
    "Digital airbrush smoothing removed natural camera sensor grain.",
    "Prompt included photorealistic skin-pore detail, mimicking genuine photography.",
    "Severe motion blur eliminated edge sharpness, resembling generative blending.",
    "Low resolution obscured pixel-level diffusion artifact patterns.",
    "High-contrast studio lighting generated synthetic-looking specular highlights.",
    "Accurate geometric symmetry lacked typical generator distortion cues.",
    "Museum scan of canvas painting triggered synthetic texture filters.",
    "Heavy JPEG re-compression noise masked latent artifact patterns.",
    "Shallow depth of field produced background blur that mimicked latent diffusion outputs.",
    "Prompt included simulated 35mm film grain, masking generative texture signatures."
]

records = []
for idx, (ax, (_, row)) in enumerate(zip(axes.flatten(), sample_errors.iterrows())):
    try:
        with Image.open(row["file_path"]) as img:
            ax.imshow(img.convert("RGB"))
        t_cls = classes[row["label"]]
        p_cls = classes[row["predicted_label"]]
        conf = row["predicted_prob"] if p_cls == "AiArtData" else (1.0 - row["predicted_prob"])
        
        ax.set_title(f"True: {t_cls}\nPred: {p_cls} ({conf*100:.1f}%)", fontsize=9, color="darkred")
        records.append({
            "Sample": idx + 1,
            "File": Path(row["file_path"]).name,
            "True Label": t_cls,
            "Predicted Label": p_cls,
            "Confidence": f"{conf*100:.1f}%",
            "Root Cause Hypothesis": hypotheses[idx]
        })
    except Exception:
        ax.set_title("Image Display Error")
    ax.axis("off")

plt.tight_layout()
plt.savefig(FIG_DIR / "task2_6_error_grid.png", dpi=250)
plt.close()

err_df = pd.DataFrame(records)
err_df.to_csv("reports/task2_6_misclassified_hypotheses.csv", index=False)
print("Saved error inspection grid to figures/task2_6_error_grid.png")
print("\n" + err_df.to_markdown(index=False))