import shutil
import json
from pathlib import Path
import tensorflow as tf

SOURCE_MODEL = Path("models/mobilenetv2_stage2_champion_best.keras")
TARGET_DIR = Path("app/model")
TARGET_DIR.mkdir(parents=True, exist_ok=True)
TARGET_MODEL = TARGET_DIR / "champion_model.keras"

print("=" * 70)
print("SUB-TASK 4.1: MODEL PACKAGING FOR LOCAL SERVING")
print("=" * 70)

# 1. Verify source model existence
if not SOURCE_MODEL.exists():
    # Fallback to stage 1 model if stage 2 not named identically
    alternatives = list(Path("models").glob("*.keras"))
    if not alternatives:
        raise FileNotFoundError("No trained .keras model found in models/ directory!")
    SOURCE_MODEL = alternatives[0]
    print(f"Using found checkpoint: {SOURCE_MODEL}")

print(f"Copying champion model from {SOURCE_MODEL} -> {TARGET_MODEL}...")
shutil.copyfile(SOURCE_MODEL, TARGET_MODEL)

# 2. Verify model can be loaded cleanly locally
print("Validating model load from local disk...")
model = tf.keras.models.load_model(TARGET_MODEL)
model_summary = []
model.summary(print_fn=lambda x: model_summary.append(x))

# 3. Export model metadata for the web application
metadata = {
    "model_name": "VisionGuard_MobileNetV2_Champion",
    "architecture": "MobileNetV2 Transfer Learning",
    "input_shape": [224, 224, 3],
    "target_classes": {"0": "RealArt", "1": "AiArtData"},
    "decision_threshold": 0.50,
    "preprocessing": "mobilenet_v2.preprocess_input [-1.0, 1.0]",
    "framework": f"TensorFlow {tf.__version__}",
    "source_checkpoint": str(SOURCE_MODEL)
}

metadata_file = TARGET_DIR / "model_metadata.json"
with open(metadata_file, "w") as f:
    json.dump(metadata, f, indent=4)

print(f"Model successfully packaged into '{TARGET_MODEL}'")
print(f"Metadata exported to '{metadata_file}'")