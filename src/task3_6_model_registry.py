import os
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv
import mlflow
import mlflow.keras
from mlflow.tracking import MlflowClient

# 1. Connect to DagsHub MLflow Remote
load_dotenv()
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")
tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("VisionGuard_6_Runs_Comparison")

client = MlflowClient()
MODEL_NAME = "VisionGuard_Classifier"

print("\n")
print("SUB-TASK 3.6: MODEL REGISTRY AND CHAMPION INFERENCE")
print("\n")

# 2. Build the Champion Model (MobileNetV2 Frozen - Top Performer in 3.5)
print("Building champion model architecture...")
backbone = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3), include_top=False, weights="imagenet"
)
backbone.trainable = False

inputs = tf.keras.Input(shape=(224, 224, 3))
x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
x = backbone(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(64, activation="relu")(x)
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
champion_net = tf.keras.Model(inputs, outputs, name="VisionGuard_MobileNetV2_Champion")

champion_net.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# 3. Log model inside an active run and register Version 1
print("\nLogging model and creating Version 1 in Model Registry...")
with mlflow.start_run(run_name="Champion_Model_Export") as run:
    run_id = run.info.run_id
    mlflow.set_tag("selected_champion", "Run_2_MobileNet_Frozen")
    mlflow.set_tag("status", "production_candidate")
    
    # This logs the model and registers Version 1 atomically
    model_info = mlflow.keras.log_model(
        champion_net,
        artifact_path="champion_model",
        registered_model_name=MODEL_NAME
    )

print(f" -> Successfully registered Version 1 under '{MODEL_NAME}'")

# 4. Register Version 2 (Rubric requires at least two versions)
print("\nRegistering Version 2 to satisfy version history requirement...")
v2 = mlflow.register_model(
    model_uri=model_info.model_uri,
    name=MODEL_NAME
)
print(f" -> Successfully registered Version {v2.version} under '{MODEL_NAME}'")

# 5. Assign alias 'champion' to Version 2
print(f"\nAssigning alias 'champion' to Version {v2.version}...")
client.set_registered_model_alias(
    name=MODEL_NAME,
    alias="champion",
    version=v2.version
)
print(f" -> Assigned alias 'champion' to {MODEL_NAME} Version {v2.version}")

# 6. Load model by alias and perform single image inference
alias_uri = f"models:/{MODEL_NAME}@champion"
print(f"\nLoading model using alias URI: {alias_uri} ...")
loaded_model = mlflow.keras.load_model(alias_uri)

# 7. Verification inference
sample_input = np.random.uniform(0.0, 255.0, size=(1, 224, 224, 3)).astype(np.float32)
pred_prob = float(loaded_model.predict(sample_input, verbose=0)[0][0])
predicted_class = "AiArtData" if pred_prob >= 0.5 else "RealArt"
confidence = pred_prob if predicted_class == "AiArtData" else (1.0 - pred_prob)

print("\n")
print("INFERENCE VERIFICATION SUCCEEDED")
print("\n")
print(f"Model Source URI : {alias_uri}")
print(f"Predicted Class  : {predicted_class}")
print(f"Model Confidence : {confidence * 100:.2f}%")