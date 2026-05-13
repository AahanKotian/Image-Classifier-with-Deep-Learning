"""
tests/test_model.py — Unit and integration tests for the classifier.

Run with:
    pytest tests/ -v
"""

import io
import json
import os
import sys
import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ─────────────────────────────────────────────
# Preprocessing Tests
# ─────────────────────────────────────────────

from utils.preprocess import load_and_resize, normalize, batch_from_file, cutout


class TestPreprocessing:
    """Tests for image preprocessing utilities."""

    def test_load_rgb_from_bytes(self):
        """Create a tiny PNG in-memory and verify load_and_resize."""
        from PIL import Image
        img = Image.new("RGB", (100, 80), color=(128, 64, 32))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        arr = load_and_resize(buf.read(), target_size=(32, 32), color_mode="rgb")
        assert arr.shape == (32, 32, 3)
        assert arr.dtype == np.float32
        assert arr.min() >= 0.0
        assert arr.max() <= 1.0

    def test_load_grayscale(self):
        from PIL import Image
        img = Image.new("RGB", (64, 64), color=(200, 200, 200))
        buf = io.BytesIO()
        img.save(buf, format="PNG")

        arr = load_and_resize(buf.getvalue(), target_size=(28, 28), color_mode="grayscale")
        assert arr.shape == (28, 28, 1), f"Expected (28,28,1), got {arr.shape}"

    def test_normalize_passthrough(self):
        """Without mean/std, array should be returned as-is."""
        arr = np.array([[[0.5, 0.3, 0.1]]], dtype=np.float32)
        out = normalize(arr)
        np.testing.assert_array_equal(arr, out)

    def test_normalize_with_stats(self):
        """Z-score normalization should shift values correctly."""
        arr = np.ones((1, 1, 3), dtype=np.float32) * 0.5
        mean = [0.5, 0.5, 0.5]
        std  = [0.25, 0.25, 0.25]
        out = normalize(arr, mean, std)
        np.testing.assert_allclose(out, np.zeros_like(out), atol=1e-6)

    def test_cutout_shape_preserved(self):
        img = np.ones((32, 32, 3), dtype=np.float32) * 0.5
        out = cutout(img, n_holes=2, hole_size=0.25)
        assert out.shape == img.shape
        # Some values should be zeroed
        assert out.min() == 0.0


# ─────────────────────────────────────────────
# Model Architecture Tests
# ─────────────────────────────────────────────

class TestModelArchitecture:
    """Tests for CNN model construction."""

    def test_build_cnn_mnist(self):
        """Model should build without error for MNIST config."""
        import tensorflow as tf
        from model.train import build_cnn

        model = build_cnn(input_shape=(28, 28, 1), num_classes=10, color_mode="grayscale")
        assert model is not None
        assert model.output_shape == (None, 10)

    def test_build_cnn_cifar10(self):
        """Model should build for CIFAR-10 config."""
        import tensorflow as tf
        from model.train import build_cnn

        model = build_cnn(input_shape=(32, 32, 3), num_classes=10, color_mode="rgb")
        assert model.output_shape == (None, 10)

    def test_forward_pass_mnist(self):
        """Forward pass with random data should return valid probabilities."""
        import tensorflow as tf
        from model.train import build_cnn

        model = build_cnn((28, 28, 1), 10, "grayscale")
        x = np.random.rand(4, 28, 28, 1).astype(np.float32)
        y = model(x, training=False).numpy()

        assert y.shape == (4, 10)
        np.testing.assert_allclose(y.sum(axis=1), np.ones(4), atol=1e-5)
        assert (y >= 0).all() and (y <= 1).all()

    def test_forward_pass_custom_classes(self):
        from model.train import build_cnn
        model = build_cnn((32, 32, 3), 5, "rgb")
        x = np.random.rand(2, 32, 32, 3).astype(np.float32)
        y = model(x, training=False).numpy()
        assert y.shape == (2, 5)


# ─────────────────────────────────────────────
# Flask API Tests
# ─────────────────────────────────────────────

@pytest.fixture
def client():
    """Create a Flask test client."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "web"))
    from app import app
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def _make_png_bytes(width=32, height=32, color="rgb"):
    """Generate a minimal test PNG."""
    from PIL import Image
    mode = "RGB" if color == "rgb" else "L"
    img = Image.new(mode, (width, height), color=128)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestFlaskAPI:
    def test_health_endpoint(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.get_json()
        assert "status" in data
        assert data["status"] == "ok"

    def test_classes_endpoint(self, client):
        r = client.get("/classes")
        assert r.status_code == 200
        data = r.get_json()
        assert "class_names" in data
        assert "num_classes" in data

    def test_predict_no_file(self, client):
        r = client.post("/predict")
        assert r.status_code == 400
        data = r.get_json()
        assert "error" in data

    def test_predict_with_image(self, client):
        """Should return a prediction (demo mode if no model is trained)."""
        img_bytes = _make_png_bytes()
        r = client.post(
            "/predict",
            data={"image": (io.BytesIO(img_bytes), "test.png")},
            content_type="multipart/form-data",
        )
        assert r.status_code == 200
        data = r.get_json()
        assert "prediction" in data
        assert "confidence" in data
        assert "top_k" in data
        assert isinstance(data["confidence"], float)
        assert 0.0 <= data["confidence"] <= 1.0

    def test_index_renders(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert b"Neural" in r.data or b"classifier" in r.data.lower()


# ─────────────────────────────────────────────
# Data Pipeline Tests
# ─────────────────────────────────────────────

class TestDataPipeline:
    def test_augmentation_pipeline_mnist(self):
        """Augmentation pipeline should be buildable."""
        from model.train import build_augmentation_pipeline
        aug = build_augmentation_pipeline("grayscale")
        assert aug is not None

    def test_augmentation_pipeline_rgb(self):
        from model.train import build_augmentation_pipeline
        aug = build_augmentation_pipeline("rgb")
        assert aug is not None
