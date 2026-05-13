"""
evaluate.py — Evaluate a trained model: metrics, confusion matrix, Grad-CAM.

Usage:
    python model/evaluate.py --model saved/cifar10_best.h5 --dataset cifar10
    python model/evaluate.py --model saved/transfer_best.h5 --meta saved/transfer_meta.json --image test.jpg
"""

import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report
import seaborn as sns

from train import load_dataset, DATASETS


# ─────────────────────────────────────────────
# Grad-CAM
# ─────────────────────────────────────────────

def make_gradcam_heatmap(img_array, model, last_conv_layer_name: str, pred_index=None):
    """
    Generate a Grad-CAM heatmap for a single image.
    Highlights which regions most influenced the prediction.
    """
    grad_model = keras.Model(
        model.inputs,
        [model.get_layer(last_conv_layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_gradcam(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4):
    """Overlay Grad-CAM heatmap on the original image."""
    heatmap_resized = np.uint8(
        255 * np.kron(heatmap, np.ones((image.shape[0] // heatmap.shape[0] + 1,
                                        image.shape[1] // heatmap.shape[1] + 1)))
        [:image.shape[0], :image.shape[1]]
    )
    colormap = cm.get_cmap("jet")
    heatmap_colored = colormap(heatmap_resized / 255.0)[:, :, :3]
    heatmap_colored = np.uint8(255 * heatmap_colored)

    if image.max() <= 1.0:
        image = np.uint8(255 * image)
    if image.ndim == 2 or (image.ndim == 3 and image.shape[-1] == 1):
        image = np.stack([image.squeeze()] * 3, axis=-1)

    superimposed = heatmap_colored * alpha + image * (1 - alpha)
    return np.uint8(superimposed)


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────

def evaluate_on_dataset(model_path: str, dataset_name: str, num_samples: int = 2000):
    print(f"\n📊 Evaluating {model_path} on {dataset_name}...")
    model = keras.models.load_model(model_path)

    cfg = DATASETS[dataset_name]
    (_, _), (x_test, y_test), _ = load_dataset(dataset_name)
    x_sample = x_test[:num_samples]
    y_sample = y_test[:num_samples]

    loss, acc = model.evaluate(x_sample, y_sample, verbose=0)
    print(f"   Accuracy : {acc*100:.2f}%")
    print(f"   Loss     : {loss:.4f}")

    y_pred = np.argmax(model.predict(x_sample, verbose=0), axis=1)
    y_true = np.argmax(y_sample, axis=1)

    print("\n📋 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=cfg["class_names"]))

    # Confusion matrix
    save_dir = os.path.dirname(model_path)
    _plot_confusion(y_true, y_pred, cfg["class_names"], dataset_name, save_dir)

    # Grad-CAM on 5 random test images
    _plot_gradcam_grid(model, x_sample, y_true, y_pred, cfg["class_names"],
                       dataset_name, save_dir)


def _plot_confusion(y_true, y_pred, class_names, dataset, save_dir):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.title(f"Confusion Matrix — {dataset}")
    plt.tight_layout()
    out = os.path.join(save_dir, f"{dataset}_eval_confusion.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"   Confusion matrix saved → {out}")


def _find_last_conv(model) -> str:
    """Find the name of the last Conv2D layer."""
    for layer in reversed(model.layers):
        if isinstance(layer, keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in model")


def _plot_gradcam_grid(model, x_test, y_true, y_pred, class_names, dataset, save_dir):
    try:
        last_conv = _find_last_conv(model)
    except ValueError:
        print("   ⚠️  Grad-CAM skipped (no Conv2D layer found)")
        return

    indices = np.random.choice(len(x_test), size=5, replace=False)
    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    fig.suptitle("Grad-CAM Visualisations", fontsize=14, fontweight="bold")

    for i, idx in enumerate(indices):
        img = x_test[idx]
        img_batch = np.expand_dims(img, 0)
        heatmap = make_gradcam_heatmap(img_batch, model, last_conv)
        overlay = overlay_gradcam(img, heatmap)

        axes[0, i].imshow(img.squeeze(), cmap="gray" if img.shape[-1] == 1 else None)
        axes[0, i].set_title(
            f"True: {class_names[y_true[idx]]}\nPred: {class_names[y_pred[idx]]}",
            fontsize=8,
            color="green" if y_true[idx] == y_pred[idx] else "red",
        )
        axes[0, i].axis("off")

        axes[1, i].imshow(overlay)
        axes[1, i].set_title("Grad-CAM", fontsize=8)
        axes[1, i].axis("off")

    plt.tight_layout()
    out = os.path.join(save_dir, f"{dataset}_gradcam.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"   Grad-CAM grid saved → {out}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained CNN classifier")
    parser.add_argument("--model", required=True, help="Path to .h5 model file")
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()),
        default=None,
        help="Built-in dataset to evaluate on",
    )
    args = parser.parse_args()

    if args.dataset:
        evaluate_on_dataset(args.model, args.dataset)
    else:
        print("Please specify --dataset")
