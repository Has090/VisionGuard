import os
import subprocess
from pathlib import Path
from dotenv import load_dotenv
import mlflow
import mlflow.keras
import mlflow.tensorflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
import tensorflow as tf

from task2_1_pipeline import build_tf_dataset, IMAGE_SIZE, SEED
from task2_2_baseline_cnn import build_baseline_cnn

# Load DagsHub credentials from .env
load_dotenv()
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("VisionGuard_Production_Tracking")


# Helper functions to retrieve Git and DVC metadata
def get_git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("ascii")
            .strip()[:8]
        )
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


# Load data splits
train_df = pd.read_csv("reports/splits/train.csv")
val_df = pd.read_csv("reports/splits/val.csv")
test_df = pd.read_csv("reports/splits/test.csv")

train_ds = build_tf_dataset(train_df, is_training=True)
val_ds = build_tf_dataset(val_df, is_training=False)
test_ds = build_tf_dataset(test_df, is_training=False)
y_test = test_df["label"].values


# -------------------------------------------------------------------
# Approach A: Custom Keras Callback for Manual Per-Epoch Metric Logging
# -------------------------------------------------------------------
class MLflowEpochCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for metric_name, metric_val in logs.items():
            mlflow.log_metric(metric_name, float(metric_val), step=epoch)


def run_manual_instrumentation():
    print("\n" + "=" * 70)
    print("STARTING APPROACH A: MANUAL LOGGING VIA KERAS CALLBACK")
    print("=" * 70)

    with mlflow.start_run(run_name="VG_MobileNetV2_Manual"):
        # 1. Tags
        mlflow.set_tag("logging_mode", "manual_callback")
        mlflow.set_tag("git_commit", get_git_commit())
        mlflow.set_tag("dvc_data_version", get_dvc_hash())
        mlflow.set_tag("random_seed", str(SEED))
        mlflow.set_tag("architecture", "MobileNetV2_FineTuned")

        # 2. Hyperparameters
        params = {
            "learning_rate": 1e-5,
            "batch_size": 32,
            "unfrozen_layers": 15,
            "dropout_rate": 0.3,
            "optimizer": "Adam",
            "loss_function": "binary_crossentropy",
            "image_size": f"{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}",
        }
        mlflow.log_params(params)

        # 3. Model setup
        backbone = tf.keras.applications.MobileNetV2(
            input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
            include_top=False,
            weights="imagenet",
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
        model = tf.keras.Model(inputs, outputs)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss="binary_crossentropy",
            metrics=["accuracy", tf.keras.metrics.AUC(name="roc_auc")],
        )

        # 4. Train with manual callback
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=3,
            callbacks=[MLflowEpochCallback()],
            verbose=1,
        )

        # 5. Evaluate on held-out test split and log final test metrics
        y_probs = model.predict(test_ds).ravel()
        y_preds = (y_probs >= 0.5).astype(int)

        test_metrics = {
            "test_accuracy": float(accuracy_score(y_test, y_preds)),
            "test_precision": float(precision_score(y_test, y_preds, zero_division=0)),
            "test_recall": float(recall_score(y_test, y_preds, zero_division=0)),
            "test_f1": float(f1_score(y_test, y_preds, zero_division=0)),
            "test_roc_auc": float(roc_auc_score(y_test, y_probs)),
        }
        for m_name, m_val in test_metrics.items():
            mlflow.log_metric(m_name, m_val)

        print("[Done] Approach A logged successfully to DagsHub.")


# -------------------------------------------------------------------
# Approach B: MLflow TensorFlow Autologging
# -------------------------------------------------------------------
def run_autolog_instrumentation():
    print("\n" + "=" * 70)
    print("STARTING APPROACH B: MLFLOW TENSORFLOW AUTOLOGGING")
    print("=" * 70)

    # Enable native TensorFlow autologging
    mlflow.tensorflow.autolog(
        log_models=True,
        disable=False,
        exclusive=False,
        log_datasets=False,
    )

    with mlflow.start_run(run_name="VG_Baseline_Autolog"):
        # Tags
        mlflow.set_tag("logging_mode", "tf_autolog")
        mlflow.set_tag("git_commit", get_git_commit())
        mlflow.set_tag("dvc_data_version", get_dvc_hash())
        mlflow.set_tag("random_seed", str(SEED))

        model = build_baseline_cnn()
        model.fit(train_ds, validation_data=val_ds, epochs=3, verbose=1)

    # Disable autologging so it does not interfere with future scripts
    mlflow.tensorflow.autolog(disable=True)
    print("[Done] Approach B logged successfully to DagsHub.")


if __name__ == "__main__":
    run_manual_instrumentation()
    run_autolog_instrumentation()