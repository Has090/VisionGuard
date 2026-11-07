import os
import subprocess
from pathlib import Path
from dotenv import load_dotenv
import mlflow
import mlflow.keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)
import tensorflow as tf

from task2_1_pipeline import build_tf_dataset, IMAGE_SIZE, SEED

# 1. Connect to DagsHub MLflow Remote
load_dotenv()
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("VisionGuard_Production_Tracking")

# 2. Metadata Helpers
def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()[:8]
    except Exception:
        return "manual_commit"

def get_dvc_hash() -> str:
    try:
        with open("data/dataset.dvc", "r") as f:
            for line in f:
                if "md5:" in line:
                    return line.split(":")[1].strip()
    except Exception:
        return "manual_dvc_hash"
    return "unknown"

# 3. Load Splits & Datasets
train_df = pd.read_csv("reports/splits/train.csv")
val_df = pd.read_csv("reports/splits/val.csv")
test_df = pd.read_csv("reports/splits/test.csv")

train_ds = build_tf_dataset(train_df, is_training=True)
val_ds = build_tf_dataset(val_df, is_training=False)
test_ds = build_tf_dataset(test_df, is_training=False)
y_test = test_df["label"].values

print("\n" + "="*70)
print("EXECUTING SUB-TASK 3.4: COMPREHENSIVE ARTIFACTS LOGGING")
print("="*70)

with mlflow.start_run(run_name="VG_Champion_Artifacts_Suite") as run:
    run_id = run.info.run_id
    print(f"Active DagsHub MLflow Run ID: {run_id}")

    # Set Reproducibility Tags
    mlflow.set_tag("task", "3.4_artifacts_management")
    mlflow.set_tag("git_commit", get_git_commit())
    mlflow.set_tag("dvc_data_version", get_dvc_hash())
    mlflow.set_tag("random_seed", str(SEED))
    mlflow.set_tag("model_type", "MobileNetV2_Champion")

    # Hyperparameters
    params = {
        "architecture": "MobileNetV2",
        "unfrozen_layers": 15,
        "dropout_rate": 0.3,
        "learning_rate": 1e-5,
        "batch_size": 32,
        "epochs": 4,
        "optimizer": "Adam",
        "loss": "binary_crossentropy"
    }
    mlflow.log_params(params)

    # Build and Train Champion MobileNetV2
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        include_top=False,
        weights="imagenet"
    )
    backbone.trainable = True
    for layer in backbone.layers[:-15]:
        layer.trainable = False

    inputs = tf.keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs, name="VisionGuard_MobileNetV2_Champion")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="roc_auc")]
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=4,
        verbose=1
    )

    # Epoch Metrics
    for ep in range(len(history.history["loss"])):
        mlflow.log_metric("train_loss", float(history.history["loss"][ep]), step=ep)
        mlflow.log_metric("train_accuracy", float(history.history["accuracy"][ep]), step=ep)
        mlflow.log_metric("val_loss", float(history.history["val_loss"][ep]), step=ep)
        mlflow.log_metric("val_accuracy", float(history.history["val_accuracy"][ep]), step=ep)

    # Evaluate on Test Set
    y_probs = model.predict(test_ds).ravel()
    y_preds = (y_probs >= 0.5).astype(int)

    test_metrics = {
        "test_accuracy": float(accuracy_score(y_test, y_preds)),
        "test_precision": float(precision_score(y_test, y_preds, zero_division=0)),
        "test_recall": float(recall_score(y_test, y_preds, zero_division=0)),
        "test_f1": float(f1_score(y_test, y_preds, zero_division=0)),
        "test_roc_auc": float(roc_auc_score(y_test, y_probs))
    }
    for m_name, m_val in test_metrics.items():
        mlflow.log_metric(m_name, m_val)

    # =================================================================
    # 3.4 ARTIFACT PERSISTENCE
    # =================================================================
    temp_dir = Path("temp_artifacts")
    temp_dir.mkdir(exist_ok=True)

    # Artifact 1: Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["RealArt (0)", "AiArtData (1)"],
        yticklabels=["RealArt (0)", "AiArtData (1)"],
        ax=ax
    )
    ax.set_title("Held-Out Test Confusion Matrix", fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
    cm_file = temp_dir / "confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(cm_file, dpi=250)
    plt.close()
    mlflow.log_artifact(str(cm_file), artifact_path="evaluation_plots")
    print(" -> Logged: evaluation_plots/confusion_matrix.png")

    # Artifact 2: ROC Curve Plot with AUC
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#2ca02c", lw=2.5, label=f"ROC Curve (AUC = {test_metrics['test_roc_auc']:.4f})")
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", alpha=0.6)
    ax.set_title("Test Receiver Operating Characteristic (ROC)", fontsize=12, fontweight="bold")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    roc_file = temp_dir / "roc_curve.png"
    plt.tight_layout()
    plt.savefig(roc_file, dpi=250)
    plt.close()
    mlflow.log_artifact(str(roc_file), artifact_path="evaluation_plots")
    print(" -> Logged: evaluation_plots/roc_curve.png")

    # Artifact 3: Exact Split Manifest CSVs
    mlflow.log_artifact("reports/splits/train.csv", artifact_path="splits")
    mlflow.log_artifact("reports/splits/val.csv", artifact_path="splits")
    mlflow.log_artifact("reports/splits/test.csv", artifact_path="splits")
    print(" -> Logged: splits/ (train.csv, val.csv, test.csv)")

    # Artifact 4: Environment Lock File
    req_file = temp_dir / "requirements_lock.txt"
    with open(req_file, "w") as f:
        subprocess.run(["pip", "freeze"], stdout=f)
    mlflow.log_artifact(str(req_file), artifact_path="environment")
    print(" -> Logged: environment/requirements_lock.txt")

    # Artifact 5: Serialized Keras Model & Architecture
    mlflow.keras.log_model(model, artifact_path="champion_model")
    print(" -> Logged: champion_model/ (MLmodel, model weights, conda.yaml)")

    # Clean local temporary folder
    for f in temp_dir.glob("*"):
        f.unlink()
    temp_dir.rmdir()

    print("\n")
    print("SUCCESS: All 5 Required Artifact Classes Uploaded to DagsHub!")
