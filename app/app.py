import os
import time
import json
import logging
import requests
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify, render_template

from app.inference import get_predictor
from app.validation import validate_image_payload, ValidationError
from app.database import (
    init_db,
    record_prediction,
    record_feedback,
    fetch_recent_history,
    get_system_analytics
)

CLOUD_MODEL_URL = "https://visionguard-api-cloud-1.onrender.com/predict"
# Initialize Flask Application
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB strict limit
APP_START_TIME = time.time()

# 1. Initialize SQLite Database Schema on App Startup
init_db()

# Ensure reports directory exists for serving logs
LOGS_DIR = Path("reports")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Configure Structured Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "app_serving.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("VisionGuard_Server")


# =====================================================================
# Middleware: Request Timing & Structured Logging
# =====================================================================
@app.before_request
def record_start_time():
    request.start_time = time.time()


@app.after_request
def log_http_transaction(response):
    duration_ms = round((time.time() - getattr(request, "start_time", time.time())) * 1000, 2)
    client_ip = request.remote_addr or "127.0.0.1"
    logger.info(
        f"{request.method} {request.path} | Status: {response.status_code} | "
        f"Latency: {duration_ms}ms | Client IP: {client_ip}"
    )
    return response


# =====================================================================
# Error Handlers
# =====================================================================
@app.errorhandler(ValidationError)
def handle_validation_error(err):
    return jsonify({
        "status": "error",
        "error_type": "ValidationError",
        "message": err.message
    }), err.status_code


@app.errorhandler(413)
def handle_payload_too_large(err):
    return jsonify({
        "status": "error",
        "error_type": "PayloadTooLarge",
        "message": "Uploaded file exceeds the maximum 10 MB payload limit."
    }), 413


@app.errorhandler(404)
def handle_not_found(err):
    return jsonify({
        "status": "error",
        "error_type": "NotFound",
        "message": "The requested endpoint does not exist."
    }), 404


@app.errorhandler(500)
def handle_internal_error(err):
    logger.error(f"Internal server error: {str(err)}")
    return jsonify({
        "status": "error",
        "error_type": "InternalServerError",
        "message": "An unexpected error occurred during model processing."
    }), 500


# =====================================================================
# Core Endpoints
# =====================================================================
@app.route("/", methods=["GET"])
def index():
    """Serves the web dashboard."""
    predictor = get_predictor()
    return render_template("index.html", metadata=predictor.metadata)


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint: verifies system uptime, model readiness, and threshold."""
    try:
        predictor = get_predictor()
        uptime_seconds = round(time.time() - APP_START_TIME, 2)
        return jsonify({
            "status": "healthy",
            "model_loaded": predictor.model is not None,
            "architecture": predictor.metadata.get("architecture", "MobileNetV2"),
            "threshold": predictor.threshold,
            "uptime_seconds": uptime_seconds,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }), 200
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 503


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename selected"}), 400

    filename = file.filename
    start_time = time.time()

    try:
        # Prepare file stream payload for the external Cloud API
        files = {"file": (filename, file.stream, file.mimetype)}
        
        # Dispatch inference request to the Cloud Model API
        cloud_response = requests.post(CLOUD_MODEL_URL, files=files, timeout=60)
        
        if cloud_response.status_code != 200:
            return jsonify({
                "error": "Cloud model inference failed",
                "details": cloud_response.text
            }), cloud_response.status_code
        
        result = cloud_response.json()
        latency_ms = round((time.time() - start_time) * 1000, 2)

        prediction = result.get("prediction")
        confidence = float(result.get("confidence", 0.0))
        raw_score = float(result.get("raw_score", 0.0))
        threshold_applied = float(result.get("threshold_applied", 0.5))
        client_ip = request.remote_addr or "127.0.0.1"

        # Log transaction into the local PostgreSQL database
        record_prediction(
            model_name="VisionGuard_Cloud_ONNX_Render",
            filename=filename,
            prediction=prediction,
            confidence=confidence,
            raw_score=raw_score,
            threshold_applied=threshold_applied,
            latency_ms=latency_ms,
            client_ip=client_ip
        )

        # Return standard response format expected by both Web UI and API clients
        if request.headers.get("Accept") == "application/json" or request.is_json:
            return jsonify({
                "status": "success",
                "prediction": prediction,
                "confidence": confidence,
                "raw_score": raw_score,
                "threshold_applied": threshold_applied,
                "latency_ms": latency_ms,
                "model_source": "Render_Cloud_Inference"
            }), 200

        return render_template(
            "index.html",
            prediction=prediction,
            confidence=confidence,
            latency_ms=latency_ms,
            filename=filename,
            model_name="VisionGuard_Cloud_ONNX_Render"
        )

    except requests.exceptions.RequestException as e:
        return jsonify({"error": "Failed to connect to Cloud API", "details": str(e)}), 503


@app.route("/feedback", methods=["POST"])
def feedback():
    """
    Feedback endpoint: records human-in-the-loop labels linked directly
    to predictions in SQLite.
    """
    payload = request.get_json(silent=True)
    if not payload:
        raise ValidationError("Request body must be a valid JSON object.", 400)

    prediction = payload.get("prediction")
    user_verdict = payload.get("user_verdict")  # Expects 'correct' or 'incorrect'
    notes = payload.get("notes", "")
    prediction_id = payload.get("prediction_id")

    if not prediction or user_verdict not in {"correct", "incorrect"}:
        raise ValidationError("Fields 'prediction' and 'user_verdict' ('correct' or 'incorrect') are required.", 400)

    # Persist feedback to SQLite database
    client_ip = request.remote_addr or "127.0.0.1"
    feedback_id = record_feedback(
        prediction_id=prediction_id,
        prediction=prediction,
        user_verdict=user_verdict,
        notes=notes,
        client_ip=client_ip
    )

    return jsonify({
        "status": "success",
        "feedback_id": feedback_id,
        "message": "User feedback recorded in database."
    }), 200


@app.route("/history", methods=["GET"])
def history():
    """Task 5 Endpoint: Returns recent predictions and attached feedback from SQLite."""
    try:
        records = fetch_recent_history(limit=50)
        return jsonify({
            "status": "success",
            "count": len(records),
            "history": records
        }), 200
    except Exception as e:
        logger.error(f"Failed to fetch history: {str(e)}")
        return jsonify({
            "status": "error",
            "message": "Unable to query database history."
        }), 500


@app.route("/analytics", methods=["GET"])
def analytics():
    """Task 5 Endpoint: Aggregated counts, accuracy, and latency."""
    try:
        stats = get_system_analytics()
        return jsonify({
            "status": "success",
            "analytics": stats
        }), 200
    except Exception as e:
        logger.error(f"Failed to fetch analytics: {str(e)}")
        return jsonify({
            "status": "error",
            "message": "Unable to compute database analytics."
        }), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)