import json
import io
import time
from pathlib import Path
from PIL import Image
import numpy as np
import tensorflow as tf

MODEL_PATH = Path(__file__).resolve().parent / "model" / "champion_model.keras"
METADATA_PATH = Path(__file__).resolve().parent / "model" / "model_metadata.json"

class VisionGuardPredictor:
    def __init__(self, model_path=MODEL_PATH, metadata_path=METADATA_PATH):
        if not model_path.exists():
            raise FileNotFoundError(f"Champion model not found at {model_path}. Run task4_1_export_model.py first.")
        
        # Load model into memory ONCE at initialization
        self.model = tf.keras.models.load_model(model_path)
        
        # Load metadata
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                "decision_threshold": 0.50,
                "target_classes": {"0": "RealArt", "1": "AiArtData"}
            }
        
        self.threshold = float(self.metadata.get("decision_threshold", 0.50))
        self.image_size = (224, 224)

    def preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        """
        Deterministic preprocessing:
        1. Open image from bytes and convert strictly to 3-channel RGB.
        2. Resize using bilinear interpolation to (224, 224).
        3. Apply MobileNetV2 normalisation: scale pixels to [-1.0, 1.0].
        4. Expand batch dimension to (1, 224, 224, 3).
        """
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        img = img.resize(self.image_size, Image.Resampling.BILINEAR)
        img_array = np.array(img, dtype=np.float32)
        
        # MobileNetV2 scaling: (x / 127.5) - 1.0
        normalized = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        batch = np.expand_dims(normalized, axis=0)
        return batch

    def predict(self, image_bytes: bytes) -> dict:
        """
        Runs inference on raw image bytes and returns structured prediction dictionary.
        """
        start_time = time.perf_counter()
        
        # Preprocess
        input_tensor = self.preprocess_image(image_bytes)
        
        # Inference
        raw_prob = float(self.model.predict(input_tensor, verbose=0)[0][0])
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        
        # Threshold decision
        is_ai = raw_prob >= self.threshold
        predicted_class = "AiArtData" if is_ai else "RealArt"
        confidence = raw_prob if is_ai else (1.0 - raw_prob)
        
        return {
            "prediction": predicted_class,
            "raw_score": round(raw_prob, 4),
            "confidence": round(confidence * 100, 2),
            "threshold_applied": self.threshold,
            "latency_ms": latency_ms,
            "status": "success"
        }

# Global singleton instance for the app
predictor = None

def get_predictor():
    global predictor
    if predictor is None:
        predictor = VisionGuardPredictor()
    return predictor