"""
transfer_learning.py — Fine-tune MobileNetV2 on a custom image dataset.

Expects a folder structured as:
    data_dir/
    ├── train/
    │   ├── class_a/
    │   └── class_b/
    └── val/
        ├── class_a/
        └── class_b/

Usage:
    python model/transfer_learning.py --data_dir ./my_images --epochs 15
"""

import argparse
import json
import os
import time

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32


# ─────────────────────────────────────────────
# Data Pipeline
# ─────────────────────────────────────────────

def build_datasets(data_dir: str, batch_size: int = BATCH_SIZE):
    """
    Build train/val tf.data.Dataset from directory structure.
    Applies augmentation to training set only.
    """
    train_ds = keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, "train"),
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=True,
        seed=42,
    )
    val_ds = keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, "val"),
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=False,
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"✅ Found {num_classes} classes: {class_names}")

    # MobileNetV2 preprocessing (scales to [-1, 1])
    preprocess = tf.keras.applications.mobilenet_v2.preprocess_input

    # Augmentation pipeline
    augment = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.15),
        layers.RandomContrast(0.1),
    ], name="augmentation")

    def prepare_train(images, labels):
        images = augment(images, training=True)
        return preprocess(images), labels

    def prepare_val(images, labels):
        return preprocess(images), labels

    # Performance tuning
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.map(prepare_train, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    val_ds = val_ds.map(prepare_val, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

    return train_ds, val_ds, class_names, num_classes


# ─────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────

def build_transfer_model(num_classes: int, freeze_base: bool = True) -> keras.Model:
    """
    MobileNetV2 base + custom classifier head.

    Phase 1: Train only the head (base frozen).
    Phase 2: Unfreeze top layers of base for fine-tuning.
    """
    base = MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = not freeze_base
    print(f"   Base trainable layers: {sum(1 for l in base.layers if l.trainable)}")

    inputs = keras.Input(shape=(*IMAGE_SIZE, 3), name="image_input")
    x = base(inputs, training=False)   # training=False keeps BN in inference mode
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    return keras.Model(inputs, outputs, name="mobilenetv2_transfer")


def unfreeze_top_layers(model: keras.Model, num_layers: int = 30):
    """Unfreeze the last `num_layers` of the base model for fine-tuning."""
    base = model.get_layer("mobilenetv2_1.00_224")
    base.trainable = True
    for layer in base.layers[:-num_layers]:
        layer.trainable = False
    trainable = sum(1 for l in base.layers if l.trainable)
    print(f"   Unfroze last {num_layers} base layers ({trainable} trainable total)")


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def train(args):
    save_dir = os.path.join(os.path.dirname(__file__), "saved")
    os.makedirs(save_dir, exist_ok=True)

    # Data
    train_ds, val_ds, class_names, num_classes = build_datasets(
        args.data_dir, args.batch_size
    )

    # ── Phase 1: Head only ──────────────────────────────────────────
    print("\n🔵 Phase 1: Training classifier head (base frozen)...")
    model = build_transfer_model(num_classes, freeze_base=True)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary(expand_nested=False)

    callbacks_p1 = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=4,
            restore_best_weights=True, verbose=1
        ),
    ]
    t0 = time.time()
    model.fit(
        train_ds, validation_data=val_ds,
        epochs=min(args.epochs // 2 + 1, 10),
        callbacks=callbacks_p1, verbose=1,
    )

    # ── Phase 2: Fine-tune top base layers ─────────────────────────
    print("\n🟠 Phase 2: Fine-tuning top base layers...")
    unfreeze_top_layers(model, num_layers=30)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),   # Much lower LR for fine-tuning
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    ckpt_path = os.path.join(save_dir, "transfer_best.h5")
    callbacks_p2 = [
        keras.callbacks.ModelCheckpoint(
            ckpt_path, monitor="val_accuracy",
            save_best_only=True, verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5,
            restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, verbose=1
        ),
    ]
    history = model.fit(
        train_ds, validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks_p2, verbose=1,
    )
    elapsed = time.time() - t0
    print(f"\n⏱  Total training time: {elapsed:.1f}s")

    # Save + metadata
    model.save(os.path.join(save_dir, "transfer_final.h5"))
    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    meta = {
        "model_type": "mobilenetv2_transfer",
        "num_classes": num_classes,
        "class_names": class_names,
        "input_shape": [*IMAGE_SIZE, 3],
        "val_accuracy": round(float(val_acc), 4),
        "val_loss": round(float(val_loss), 4),
    }
    meta_path = os.path.join(save_dir, "transfer_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✅ Final val accuracy: {val_acc*100:.2f}%")
    print(f"   Saved model → {save_dir}/transfer_final.h5")
    print(f"   Saved meta  → {meta_path}")
    return model


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune MobileNetV2 on a custom image dataset"
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Path to dataset folder with train/ and val/ subfolders",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Fine-tuning epochs")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    train(args)
