import os
from pathlib import Path
import matplotlib.pyplot as plt
import mlflow
import mlflow.keras
import tensorflow as tf

# Configure local SQLite database tracking backend
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Toy_Local_Experiment")

# Load built-in MNIST dataset for the toy experiment
(x_train, y_train), (x_val, y_val) = tf.keras.datasets.mnist.load_data()
x_train, x_val = x_train[:2000] / 255.0, x_val[:500] / 255.0
y_train, y_val = y_train[:2000], y_val[:500]

def train_toy_run(learning_rate: float, run_name: str):
    with mlflow.start_run(run_name=run_name):
        # 1. Log static parameters
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("epochs", 4)
        mlflow.log_param("batch_size", 64)
        mlflow.log_param("optimizer", "Adam")
        mlflow.set_tag("developer", "VisionGuard_Student")
        mlflow.set_tag("environment", "local_toy")

        # 2. Build and compile simple architecture
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(28, 28)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        # 3. Train
        history = model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=4,
            batch_size=64,
            verbose=1,
        )

        # 4. Log per-epoch metrics
        for epoch in range(len(history.history["loss"])):
            mlflow.log_metric("train_loss", float(history.history["loss"][epoch]), step=epoch)
            mlflow.log_metric("train_accuracy", float(history.history["accuracy"][epoch]), step=epoch)
            mlflow.log_metric("val_loss", float(history.history["val_loss"][epoch]), step=epoch)
            mlflow.log_metric("val_accuracy", float(history.history["val_accuracy"][epoch]), step=epoch)

        # 5. Generate, save, and log a loss plot artifact
        plt.figure(figsize=(6, 4))
        plt.plot(history.history["loss"], label="Train Loss", color="blue")
        plt.plot(history.history["val_loss"], label="Val Loss", color="orange")
        plt.title(f"Training & Validation Loss (LR = {learning_rate})")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plot_path = f"toy_loss_lr_{learning_rate}.png"
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()

        mlflow.log_artifact(plot_path)
        if os.path.exists(plot_path):
            os.remove(plot_path)

        # 6. Log model weights
        mlflow.keras.log_model(model, artifact_path="toy_model")
        print(f"\n[Run Completed] {run_name} successfully recorded.")

if __name__ == "__main__":
    print("\n")
    print("STARTING SUB-TASK 3.1: LOCAL MLFLOW TOY EXPERIMENT")
    train_toy_run(learning_rate=0.01, run_name="toy_run_high_lr")
    train_toy_run(learning_rate=0.001, run_name="toy_run_low_lr")
    print("\n")
    print("All runs logged successfully to sqlite:///mlflow.db")