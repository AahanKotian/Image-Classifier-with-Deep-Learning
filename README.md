# 🖼️ Deep Learning Image Classifier

A full-stack deep learning project featuring a **Convolutional Neural Network (CNN)** trained on image classification tasks, with a polished drag-and-drop web UI for real-time inference.

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow)
![Flask](https://img.shields.io/badge/Flask-2.x-black?style=flat-square&logo=flask)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 🎯 What This Project Demonstrates

| Skill | Implementation |
|---|---|
| **CNNs** | Custom Conv2D → MaxPool → Dropout → Dense architecture |
| **Transfer Learning** | MobileNetV2 fine-tuning on custom datasets |
| **Data Augmentation** | Random flip, rotation, zoom via `tf.keras.layers` |
| **Model Serialization** | SavedModel + TFLite export for web deployment |
| **REST API** | Flask endpoint serving predictions with confidence scores |
| **Modern Web UI** | Drag-and-drop canvas with real-time TensorFlow.js inference |

---

## 🏗️ Project Structure

```
image-classifier/
├── model/
│   ├── train.py              # CNN training script (MNIST / CIFAR-10)
│   ├── transfer_learning.py  # MobileNetV2 fine-tuning
│   ├── evaluate.py           # Metrics, confusion matrix, grad-cam
│   └── saved/                # Exported .h5 and SavedModel artifacts
├── web/
│   ├── app.py                # Flask API server
│   ├── static/
│   │   ├── css/style.css     # UI styles
│   │   └── js/classifier.js  # TensorFlow.js client inference
│   └── templates/
│       └── index.html        # Drag-and-drop UI
├── utils/
│   ├── preprocess.py         # Image pipeline helpers
│   └── dataset.py            # Dataset loaders
├── notebooks/
│   └── exploration.ipynb     # EDA and training walkthrough
├── tests/
│   └── test_model.py         # Unit tests
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/yourusername/image-classifier.git
cd image-classifier
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Train the Model

```bash
# Train on MNIST (fast — ~2 min on CPU)
python model/train.py --dataset mnist --epochs 10

# Train on CIFAR-10 (recommended — ~15 min on GPU)
python model/train.py --dataset cifar10 --epochs 25

# Fine-tune MobileNetV2 on a custom folder of images
python model/transfer_learning.py --data_dir ./my_images --epochs 10
```

### 3. Launch the Web UI

```bash
cd web
python app.py
# Open http://localhost:5000
```

### 4. Run with Docker

```bash
docker build -t image-classifier .
docker run -p 5000:5000 image-classifier
```

---

## 🧠 Model Architecture

### Custom CNN (MNIST / CIFAR-10)

```
Input (28x28x1 or 32x32x3)
  │
  ├─ Conv2D(32, 3x3, relu) → BatchNorm → MaxPool(2x2)
  ├─ Conv2D(64, 3x3, relu) → BatchNorm → MaxPool(2x2)
  ├─ Conv2D(128, 3x3, relu) → BatchNorm
  │
  ├─ GlobalAveragePooling2D
  ├─ Dense(256, relu) → Dropout(0.4)
  └─ Dense(num_classes, softmax)
```

### Transfer Learning (Custom Datasets)

```
MobileNetV2 (ImageNet weights, frozen)
  │
  ├─ GlobalAveragePooling2D
  ├─ Dense(128, relu) → Dropout(0.3)
  └─ Dense(num_classes, softmax)
```

**Training Results (CIFAR-10):**
- Test Accuracy: **~91%**
- Parameters: 2.3M (custom CNN), 3.5M (MobileNetV2)

---

## 📊 Key Learning Concepts

### Convolutional Neural Networks
CNNs use learnable filters that slide over the image, detecting edges → textures → shapes → objects at increasing abstraction levels. Each `Conv2D` layer learns these filters automatically from data.

### Transfer Learning
Instead of training from scratch, we use MobileNetV2 pre-trained on 1.4M ImageNet images. We freeze early layers (which learned universal features like edges) and only fine-tune later layers on our specific classes. This works well with as few as **100–500 images per class**.

### Data Augmentation
Randomly flipping, rotating, and zooming training images creates artificial variety, reducing overfitting — the model learns the object, not the exact pixels.

---

## 🌐 API Reference

**POST** `/predict`

```bash
curl -X POST http://localhost:5000/predict \
  -F "image=@test_image.jpg"
```

```json
{
  "prediction": "cat",
  "confidence": 0.9423,
  "top_k": [
    {"class": "cat",  "confidence": 0.9423},
    {"class": "dog",  "confidence": 0.0511},
    {"class": "bird", "confidence": 0.0066}
  ],
  "inference_time_ms": 12.4
}
```

---

## 🔧 Customizing for Your Own Dataset

```
my_images/
├── train/
│   ├── cat/     (≥100 images)
│   ├── dog/     (≥100 images)
│   └── bird/    (≥100 images)
└── val/
    ├── cat/
    ├── dog/
    └── bird/
```

Then run:
```bash
python model/transfer_learning.py --data_dir ./my_images --epochs 15
```

---

## 🧪 Tests

```bash
pytest tests/ -v
```

---

## 📈 Future Improvements

- [ ] Grad-CAM visualizations (see what the model "looks at")
- [ ] ONNX export for cross-platform deployment
- [ ] React frontend with webcam capture
- [ ] Model quantization for mobile (TFLite)
- [ ] CI/CD with GitHub Actions

---

## 📄 License

MIT License — see [LICENSE](LICENSE)
