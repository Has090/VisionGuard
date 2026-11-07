import io
import pytest
from PIL import Image
from app.app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def create_dummy_image_bytes(format="JPEG", size=(100, 100), color=(200, 50, 50)):
    """Helper to generate in-memory image bytes."""
    buf = io.BytesIO()
    img = Image.new("RGB", size, color=color)
    img.save(buf, format=format)
    buf.seek(0)
    return buf


# 1. Test /health
def test_health_endpoint(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "uptime_seconds" in data


# 2. Test valid JPEG prediction
def test_predict_valid_jpeg(client):
    img_buf = create_dummy_image_bytes(format="JPEG")
    res = client.post(
        "/predict",
        data={"file": (img_buf, "test_sample.jpg")},
        content_type="multipart/form-data"
    )
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "success"
    assert data["prediction"] in {"AiArtData", "RealArt"}
    assert 0.0 <= data["confidence"] <= 100.0
    assert "latency_ms" in data


# 3. Test invalid extension rejection (Sub-task 4.4)
def test_predict_invalid_extension(client):
    res = client.post(
        "/predict",
        data={"file": (io.BytesIO(b"fake binary"), "malicious_script.exe")},
        content_type="multipart/form-data"
    )
    assert res.status_code == 415
    data = res.get_json()
    assert "Unsupported file extension" in data["message"]


# 4. Test corrupted image rejection (Sub-task 4.4 magic byte check)
def test_predict_corrupted_image(client):
    fake_jpg = io.BytesIO(b"This is not a real JPEG image byte sequence!")
    res = client.post(
        "/predict",
        data={"file": (fake_jpg, "corrupt.jpg")},
        content_type="multipart/form-data"
    )
    assert res.status_code == 400
    data = res.get_json()
    assert "Corrupted or invalid image file" in data["message"]


# 5. Test empty file rejection
def test_predict_empty_file(client):
    empty = io.BytesIO(b"")
    res = client.post(
        "/predict",
        data={"file": (empty, "empty.jpg")},
        content_type="multipart/form-data"
    )
    assert res.status_code == 400
    data = res.get_json()
    assert "empty" in data["message"]


# 6. Test micro-image rejection (Dimension Guard < 10x10)
def test_predict_micro_dimension(client):
    tiny = create_dummy_image_bytes(format="PNG", size=(5, 5))
    res = client.post(
        "/predict",
        data={"file": (tiny, "tiny.png")},
        content_type="multipart/form-data"
    )
    assert res.status_code == 400
    data = res.get_json()
    assert "too small" in data["message"]


# 7. Test feedback endpoint
def test_feedback_submission(client):
    payload = {
        "prediction": "AiArtData",
        "user_verdict": "correct",
        "notes": "Pytest automated verification"
    }
    res = client.post("/feedback", json=payload)
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "success"