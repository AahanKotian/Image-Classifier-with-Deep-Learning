"""
app.py — Flask REST API serving CNN predictions.

Routes:
    GET  /              → Web UI
    POST /predict       → JSON prediction from uploaded image
    GET  /health        → Health check
    GET  /classes       → List class names
"""

import io
import json
import os
import time

import numpy as np
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from PIL import Image
import tensorflow as tf
from tensorflow import keras


app = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────────
# Model Loading
# ─────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "..", "model", "saved")

_model = None
_meta = None


def _load_model():
    """Lazy-load the model on first request."""
    global _model, _meta

    if _model is not None:
        return _model, _meta

    # Prefer CIFAR-10 > MNIST > transfer
    candidates = [
        ("cifar10_best.h5", "cifar10_meta.json"),
        ("mnist_best.h5", "mnist_meta.json"),
        ("transfer_best.h5", "transfer_meta.json"),
    ]

    for model_file, meta_file in candidates:
        model_path = os.path.join(SAVE_DIR, model_file)
        meta_path = os.path.join(SAVE_DIR, meta_file)
        if os.path.exists(model_path) and os.path.exists(meta_path):
            print(f"🔵 Loading model: {model_file}")
            _model = keras.models.load_model(model_path)
            with open(meta_path) as f:
                _meta = json.load(f)
            print(f"✅ Loaded — classes: {_meta['class_names']}")
            return _model, _meta

    # Demo mode: return None (UI still works, returns demo response)
    print("⚠️  No trained model found. Running in demo mode.")
    return None, None


# ─────────────────────────────────────────────
# Image Preprocessing
# ─────────────────────────────────────────────

def preprocess_image(image_bytes: bytes, input_shape: list) -> np.ndarray:
    """
    Decode raw image bytes and resize/normalize to match model input.

    Args:
        image_bytes: Raw bytes from the uploaded file
        input_shape: [height, width, channels] from model metadata

    Returns:
        np.ndarray of shape (1, H, W, C), normalized to [0, 1]
    """
    h, w, c = input_shape
    img = Image.open(io.BytesIO(image_bytes))

    # Convert to grayscale or RGB as needed
    if c == 1:
        img = img.convert("L")
    else:
        img = img.convert("RGB")

    img = img.resize((w, h), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0

    if c == 1:
        arr = np.expand_dims(arr, -1)   # (H, W) → (H, W, 1)

    return np.expand_dims(arr, 0)       # → (1, H, W, C)


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    model, meta = _load_model()
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "dataset": meta.get("dataset") if meta else None,
    })


@app.route("/classes")
def classes():
    _, meta = _load_model()
    if meta is None:
        return jsonify({"class_names": [], "num_classes": 0})
    return jsonify({
        "class_names": meta["class_names"],
        "num_classes": meta["num_classes"],
    })


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided. Send as form field 'image'."}), 400

    image_file = request.files["image"]
    if image_file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    image_bytes = image_file.read()

    model, meta = _load_model()

    # ── Demo mode (no trained model) ─────────────────────────────
    if model is None:
        import random
        demo_classes = ["cat", "dog", "bird", "car", "airplane"]
        pred = random.choice(demo_classes)
        conf = round(random.uniform(0.70, 0.99), 4)
        return jsonify({
            "prediction": pred,
            "confidence": conf,
            "top_k": [{"class": c, "confidence": round(random.uniform(0.01, 0.15), 4)}
                      for c in demo_classes if c != pred][:3],
            "inference_time_ms": round(random.uniform(5, 20), 1),
            "demo_mode": True,
            "note": "Train a model first with: python model/train.py --dataset cifar10",
        })

    # ── Real inference ────────────────────────────────────────────
    try:
        img = preprocess_image(image_bytes, meta["input_shape"])
    except Exception as e:
        return jsonify({"error": f"Could not read image: {str(e)}"}), 422

    t0 = time.perf_counter()
    preds = model.predict(img, verbose=0)[0]          # shape: (num_classes,)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    class_names = meta["class_names"]
    pred_idx = int(np.argmax(preds))
    confidence = float(preds[pred_idx])

    # Top-k (k=5 or all if fewer classes)
    k = min(5, len(class_names))
    top_indices = np.argsort(preds)[::-1][:k]
    top_k = [
        {"class": class_names[i], "confidence": round(float(preds[i]), 4)}
        for i in top_indices
    ]

    return jsonify({
        "prediction": class_names[pred_idx],
        "confidence": round(confidence, 4),
        "top_k": top_k,
        "inference_time_ms": round(elapsed_ms, 2),
    })


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    print(f"🚀 Starting server at http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=debug)
