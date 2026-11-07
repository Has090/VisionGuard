import os
from pathlib import Path
from dotenv import load_dotenv
import mlflow
import mlflow.keras
import tensorflow as tf

# 1. Load credentials securely from .env
load_dotenv()

tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
username = os.getenv("MLFLOW_TRACKING_USERNAME")
token = os.getenv("MLFLOW_TRACKING_PASSWORD")

if not tracking_uri or not username or not token:
    raise ValueError("Missing DagsHub credentials! Check your .env file.")

# Set MLflow environment variables for remote authentication
os.environ["MLFLOW_TRACKING_USERNAME"] = username
os.environ["MLFLOW_TRACKING_PASSWORD"] = token
mlflow.set_tracking_uri(tracking_uri)

# Set remote experiment name
mlflow.set_experiment("DagsHub_Remote_Toy_Experiment")

# 2. Load small subset of MNIST for verification
(x_train, y_train), (x_val, y_val) = tf.keras.datasets.mnist.load_data()
x_train, x_val = x_train[:1500] / 255.0, x_val[:400] / 255.0
y_train, y_val = y_train[:1500], y_val[:400]

print("\n")
print(f"Tracking URI: {tracking_uri}")
print("Connecting to DagsHub MLflow Server...")

with mlflow.start_run(run_name="dagshub_remote_toy_run"):
    # Log parameters
    mlflow.log_param("dataset", "MNIST_Toy_Subset")
    mlflow.log_param("epochs", 3)
    mlflow.log_param("batch_size", 64)
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_param("optimizer", "Adam")
    mlflow.set_tag("environment", "dagshub_remote")
    mlflow.set_tag("developer", "VisionGuard_Student")

    # Build simple network
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Train
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=3,
        batch_size=64,
        verbose=1
    )

    # Log metrics per epoch
    for epoch in range(len(history.history["loss"])):
        mlflow.log_metric("loss", float(history.history["loss"][epoch]), step=epoch)
        mlflow.log_metric("accuracy", float(history.history["accuracy"][epoch]), step=epoch)
        mlflow.log_metric("val_loss", float(history.history["val_loss"][epoch]), step=epoch)
        mlflow.log_metric("val_accuracy", float(history.history["val_accuracy"][epoch]), step=epoch)

    # Log model to DagsHub remote artifact storage
    mlflow.keras.log_model(model, artifact_path="toy_remote_model")
    print("\n")
    print("SUCCESS: Toy run and model artifact logged to DagsHub!")
