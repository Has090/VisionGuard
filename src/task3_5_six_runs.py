import os
from pathlib import Path
from dotenv import load_dotenv
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import tensorflow as tf

from task2_1_pipeline import build_tf_dataset, IMAGE_SIZE, SEED
from task2_2_baseline_cnn import build_baseline_cnn

# Connect to DagsHub MLflow Remote
load_dotenv()
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("VisionGuard_6_Runs_Comparison")

# Load Partitions
train_df = pd.read_csv("reports/splits/train.csv")
val_df = pd.read_csv("reports/splits/val.csv")
test_df = pd.read_csv("reports/splits/test.csv")
y_test = test_df["label"].values

def train_and_log_run(run_name: str, config: dict):
    print("\n")
    print(f"RUNNING EXPERIMENT: {run_name}")
    print(f"Hypothesis: {config['hypothesis']}")
    print("\n")

    with mlflow.start_run(run_name=run_name):
        # 1. Log Hyperparameters
        for k, v in config.items():
            mlflow.log_param(k, v)

        # 2. Build Datasets (Respecting Augmentation Toggle)
        train_ds = build_tf_dataset(train_df, is_training=config["use_augmentation"])
        val_ds = build_tf_dataset(val_df, is_training=False)
        test_ds = build_tf_dataset(test_df, is_training=False)

        # 3. Build Model Architecture
        if config["backbone"] == "Baseline_CNN":
            model = build_baseline_cnn(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
        else:
            base_model = tf.keras.applications.MobileNetV2(
                input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
                include_top=False,
                weights="imagenet"
            )

            unfreeze = config["unfrozen_layers"]
            if unfreeze == 0:
                base_model.trainable = False
            else:
                base_model.trainable = True
                for layer in base_model.layers[:-unfreeze]:
                    layer.trainable = False

            inputs = tf.keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
            x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
            x = base_model(x, training=False)
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            x = tf.keras.layers.Dropout(config["dropout_rate"])(x)
            x = tf.keras.layers.Dense(64, activation="relu")(x)
            outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
            model = tf.keras.Model(inputs, outputs)

            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]),
                loss="binary_crossentropy",
                metrics=["accuracy", tf.keras.metrics.AUC(name="roc_auc")]
            )

        # 4. Train Model
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=config["epochs"],
            verbose=1
        )

        # 5. Log Per-Epoch Validation Metrics
        for ep in range(len(history.history["loss"])):
            mlflow.log_metric("val_loss", float(history.history["val_loss"][ep]), step=ep)
            mlflow.log_metric("val_accuracy", float(history.history["val_accuracy"][ep]), step=ep)

        # 6. Evaluate on Held-Out Test Set
        y_probs = model.predict(test_ds).ravel()
        y_preds = (y_probs >= 0.5).astype(int)

        test_acc = float(accuracy_score(y_test, y_preds))
        test_prec = float(precision_score(y_test, y_preds, zero_division=0))
        test_rec = float(recall_score(y_test, y_preds, zero_division=0))
        test_f1 = float(f1_score(y_test, y_preds, zero_division=0))
        test_auc = float(roc_auc_score(y_test, y_probs))

        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("test_precision", test_prec)
        mlflow.log_metric("test_recall", test_rec)
        mlflow.log_metric("test_f1", test_f1)
        mlflow.log_metric("test_roc_auc", test_auc)

        print(f"Results for {run_name}:")
        print(f"  Test Accuracy: {test_acc:.4f} | Precision: {test_prec:.4f} | Recall: {test_rec:.4f} | F1: {test_f1:.4f} | AUC: {test_auc:.4f}")

# -------------------------------------------------------------------
# The 6 Controlled Configurations
# -------------------------------------------------------------------
experiments_suite = [
    (
        "Run_1_Baseline_CNN",
        {
            "backbone": "Baseline_CNN",
            "learning_rate": 5e-4,
            "unfrozen_layers": 0,
            "dropout_rate": 0.2,
            "use_augmentation": True,
            "epochs": 4,
            "hypothesis": "Benchmark custom CNN without transfer learning priors."
        }
    ),
    (
        "Run_2_MobileNet_Frozen",
        {
            "backbone": "MobileNetV2",
            "learning_rate": 1e-3,
            "unfrozen_layers": 0,
            "dropout_rate": 0.3,
            "use_augmentation": True,
            "epochs": 4,
            "hypothesis": "Transfer learning using completely frozen ImageNet feature extractor."
        }
    ),
    (
        "Run_3_MobileNet_FineTune15",
        {
            "backbone": "MobileNetV2",
            "learning_rate": 1e-5,
            "unfrozen_layers": 15,
            "dropout_rate": 0.3,
            "use_augmentation": True,
            "epochs": 4,
            "hypothesis": "Champion candidate: Unfreeze last 15 layers with calibrated low learning rate."
        }
    ),
    (
        "Run_4_MobileNet_HighLR",
        {
            "backbone": "MobileNetV2",
            "learning_rate": 1e-3,
            "unfrozen_layers": 15,
            "dropout_rate": 0.3,
            "use_augmentation": True,
            "epochs": 4,
            "hypothesis": "Testing failure mode: High learning rate (1e-3) with unfrozen layers."
        }
    ),
    (
        "Run_5_MobileNet_NoAugment",
        {
            "backbone": "MobileNetV2",
            "learning_rate": 1e-5,
            "unfrozen_layers": 15,
            "dropout_rate": 0.3,
            "use_augmentation": False,
            "epochs": 4,
            "hypothesis": "Assessing impact when disabling training data augmentation."
        }
    ),
    (
        "Run_6_MobileNet_HighDropout",
        {
            "backbone": "MobileNetV2",
            "learning_rate": 1e-5,
            "unfrozen_layers": 15,
            "dropout_rate": 0.5,
            "use_augmentation": True,
            "epochs": 4,
            "hypothesis": "Evaluating heavier regularization (dropout=0.5) to curb memorization."
        }
    )
]

if __name__ == "__main__":
    for run_name, config in experiments_suite:
        train_and_log_run(run_name, config)
    print("\n")
    print("ALL 6 EXPERIMENTS COMPLETED AND LOGGED TO DAGSHUB MLFLOW!")
    print("\n")