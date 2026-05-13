"""
train.py — Train a CNN image classifier from scratch.

Supports MNIST (digits) and CIFAR-10 (objects).

Usage:
    python model/train.py --dataset mnist --epochs 10
    python model/train.py --dataset cifar10 --epochs 25 --batch_size 64
"""

import argparse
import os
import json
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


# ─────────────────────────────────────────────
# Dataset Loaders
# ─────────────────────────────────────────────

DATASETS = {
    "mnist": {
        "loader": keras.datasets.mnist,
        "num_classes": 10,
        "class_names": [str(i) for i in range(10)],
        "input_shape": (28, 28, 1),
        "color_mode": "grayscale",
    },
    "cifar10": {
        "loader": keras.datasets.cifar10,
        "num_classes": 10,
        "class_names": [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck",
        ],
        "input_shape": (32, 32, 3),
        "color_mode": "rgb",
    },
    "fashion_mnist": {
        "loader": keras.datasets.fashion_mnist,
        "num_classes": 10,
        "class_names": [
            "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Boot",
        ],
        "input_shape": (28, 28, 1),
        "color_mode": "grayscale",
    },
}


def load_dataset(name: str):
    """Load and preprocess a built-in Keras dataset."""
    cfg = DATASETS[name]
    (x_train, y_train), (x_test, y_test) = cfg["loader"].load_data()

    # Normalize to [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Add channel dim for grayscale datasets
    if cfg["color_mode"] == "grayscale":
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)

    # One-hot encode labels
    y_train = keras.utils.to_categorical(y_train, cfg["num_classes"])
    y_test = keras.utils.to_categorical(y_test, cfg["num_classes"])

    print(f"✅ Loaded {name}: train={x_train.shape}, test={x_test.shape}")
    return (x_train, y_train), (x_test, y_test), cfg


# ─────────────────────────────────────────────
# Data Augmentation
# ─────────────────────────────────────────────

def build_augmentation_pipeline(color_mode: str) -> keras.Sequential:
    """Build a data augmentation pipeline as a Keras layer."""
    layers_list = [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.1, 0.1),
    ]
    # Extra augmentation for color images
    if color_mode == "rgb":
        layers_list.append(layers.RandomContrast(0.1))

    return keras.Sequential(layers_list, name="augmentation")


# ─────────────────────────────────────────────
# Model Architecture
# ─────────────────────────────────────────────

def build_cnn(input_shape: tuple, num_classes: int, color_mode: str) -> keras.Model:
    """
    Build a 3-block CNN with BatchNorm and Dropout.

    Architecture:
        Input → [Conv → BN → ReLU → MaxPool] x3
              → GlobalAvgPool → Dense(256) → Dropout → Output
    """
    aug = build_augmentation_pipeline(color_mode)

    inputs = keras.Input(shape=input_shape, name="image_input")

    # Augmentation (only active during training)
    x = aug(inputs)

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # Classifier head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs, outputs, name="cnn_classifier")
    return model


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def train(args):
    # Load data
    (x_train, y_train), (x_test, y_test), cfg = load_dataset(args.dataset)
    input_shape = cfg["input_shape"]
    num_classes = cfg["num_classes"]
    class_names = cfg["class_names"]

    # Build model
    model = build_cnn(input_shape, num_classes, cfg["color_mode"])
    model.summary()

    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Callbacks
    save_dir = os.path.join(os.path.dirname(__file__), "saved")
    os.makedirs(save_dir, exist_ok=True)

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(save_dir, f"{args.dataset}_best.h5"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            verbose=1,
        ),
        keras.callbacks.TensorBoard(
            log_dir=os.path.join(save_dir, "logs"),
            histogram_freq=1,
        ),
    ]

    # Train
    print(f"\n🚀 Training on {args.dataset} for {args.epochs} epochs...\n")
    t0 = time.time()
    history = model.fit(
        x_train, y_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1,
    )
    elapsed = time.time() - t0
    print(f"\n⏱  Training complete in {elapsed:.1f}s")

    # Evaluate
    print("\n📊 Evaluating on test set...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"   Test accuracy : {test_acc * 100:.2f}%")
    print(f"   Test loss     : {test_loss:.4f}")

    # Save final model + metadata
    model.save(os.path.join(save_dir, f"{args.dataset}_final.h5"))
    meta = {
        "dataset": args.dataset,
        "num_classes": num_classes,
        "class_names": class_names,
        "input_shape": list(input_shape),
        "test_accuracy": round(float(test_acc), 4),
        "test_loss": round(float(test_loss), 4),
    }
    with open(os.path.join(save_dir, f"{args.dataset}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"✅ Model + metadata saved to {save_dir}/")

    # Plot training curves
    _plot_history(history, args.dataset, save_dir)

    # Confusion matrix on a subset
    y_pred = np.argmax(model.predict(x_test[:1000], verbose=0), axis=1)
    y_true = np.argmax(y_test[:1000], axis=1)
    _plot_confusion_matrix(y_true, y_pred, class_names, args.dataset, save_dir)

    print("\n📈 Plots saved. Done!")
    return model, history


# ─────────────────────────────────────────────
# Visualisation Helpers
# ─────────────────────────────────────────────

def _plot_history(history, dataset: str, save_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Training History — {dataset.upper()}", fontsize=14, fontweight="bold")

    axes[0].plot(history.history["accuracy"], label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="Train")
    axes[1].plot(history.history["val_loss"], label="Val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{dataset}_training_curves.png"), dpi=150)
    plt.close()


def _plot_confusion_matrix(y_true, y_pred, class_names, dataset, save_dir):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {dataset.upper()}")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{dataset}_confusion_matrix.png"), dpi=150)
    plt.close()


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a CNN image classifier")
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()),
        default="mnist",
        help="Dataset to train on (default: mnist)",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    args = parser.parse_args()

    train(args)
