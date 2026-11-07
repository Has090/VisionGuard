from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)
from task2_1_pipeline import build_tf_dataset

MODELS_DIR = Path("models")
FIG_DIR = Path("figures")
REPORTS_DIR = Path("reports")

test_df = pd.read_csv("reports/splits/test.csv")
val_df = pd.read_csv("reports/splits/val.csv")

val_ds = build_tf_dataset(val_df, is_training=False)
test_ds = build_tf_dataset(test_df, is_training=False)
y_test = test_df["label"].values
y_val = val_df["label"].values

baseline_model = tf.keras.models.load_model(MODELS_DIR / "baseline_cnn_best.keras")
mobilenet_model = tf.keras.models.load_model(MODELS_DIR / "mobilenetv2_stage2_champion_best.keras")

val_preds = mobilenet_model.predict(val_ds).ravel()
thresholds = np.linspace(0.1, 0.9, 81)
best_f1, optimal_thresh = 0.0, 0.5

for th in thresholds:
    bin_preds = (val_preds >= th).astype(int)
    current_f1 = f1_score(y_val, bin_preds, zero_division=0)
    if current_f1 > best_f1:
        best_f1 = current_f1
        optimal_thresh = th

print(f"\n[Validation Optimization] Optimal F1 Threshold: {optimal_thresh:.3f} (Val F1: {best_f1:.4f})")

y_pred_base = baseline_model.predict(test_ds).ravel()
y_pred_mobile = mobilenet_model.predict(test_ds).ravel()

def compute_metrics(y_true, probs, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    return {
        "Accuracy": round(accuracy_score(y_true, preds), 4),
        "Precision": round(precision_score(y_true, preds, zero_division=0), 4),
        "Recall": round(recall_score(y_true, preds, zero_division=0), 4),
        "F1-Score": round(f1_score(y_true, preds, zero_division=0), 4),
        "ROC-AUC": round(roc_auc_score(y_true, probs), 4),
    }

m_base = compute_metrics(y_test, y_pred_base, 0.5)
m_mobile_def = compute_metrics(y_test, y_pred_mobile, 0.5)
m_mobile_opt = compute_metrics(y_test, y_pred_mobile, optimal_thresh)

summary_df = pd.DataFrame([
    {"Model": "Baseline Custom CNN (Default Thresh 0.50)", **m_base},
    {"Model": "MobileNetV2 Champion (Default Thresh 0.50)", **m_mobile_def},
    {"Model": f"MobileNetV2 Champion (Optimized Thresh {optimal_thresh:.2f})", **m_mobile_opt}
])

print("\n")
print("HELD-OUT TEST SET EVALUATION COMPARISON")
print("\n")
print(summary_df.to_markdown(index=False))
summary_df.to_csv(REPORTS_DIR / "task2_5_test_metrics.csv", index=False)

# 4. ROC Curves
fpr_b, tpr_b, _ = roc_curve(y_test, y_pred_base)
fpr_m, tpr_m, _ = roc_curve(y_test, y_pred_mobile)

plt.figure(figsize=(7, 6))
plt.plot(fpr_b, tpr_b, label=f"Baseline CNN (AUC = {m_base['ROC-AUC']:.3f})", color="#1f77b4", lw=2)
plt.plot(fpr_m, tpr_m, label=f"MobileNetV2 (AUC = {m_mobile_def['ROC-AUC']:.3f})", color="#2ca02c", lw=2.5)
plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
plt.title("ROC Curves: Baseline CNN vs MobileNetV2", fontsize=12, fontweight="bold")
plt.xlabel("False Positive Rate (1 - Specificity)")
plt.ylabel("True Positive Rate (Sensitivity / Recall)")
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "task2_5_roc_curves.png", dpi=250)
plt.close()

# 5. Confusion Matrices
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
cm_base = confusion_matrix(y_test, (y_pred_base >= 0.5).astype(int))
cm_mobile = confusion_matrix(y_test, (y_pred_mobile >= optimal_thresh).astype(int))
labels = ["RealArt", "AiArtData"]

sns.heatmap(cm_base, annot=True, fmt="d", cmap="Blues", ax=ax1, xticklabels=labels, yticklabels=labels)
ax1.set_title("Baseline CNN (Threshold 0.50)")
ax1.set_xlabel("Predicted")
ax1.set_ylabel("Actual")

sns.heatmap(cm_mobile, annot=True, fmt="d", cmap="Greens", ax=ax2, xticklabels=labels, yticklabels=labels)
ax2.set_title(f"MobileNetV2 (Optimized Threshold {optimal_thresh:.2f})")
ax2.set_xlabel("Predicted")
ax2.set_ylabel("Actual")

plt.tight_layout()
plt.savefig(FIG_DIR / "task2_5_confusion_matrices.png", dpi=250)
plt.close()
print("Evaluation complete. Figures saved to figures/.")