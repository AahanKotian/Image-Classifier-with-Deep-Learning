"""
dataset.py — Dataset loading utilities for custom image folders.

Provides helpers for splitting raw image directories into
train/val splits and creating tf.data pipelines.
"""

import os
import shutil
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras


# ─────────────────────────────────────────────
# Train / Val Split
# ─────────────────────────────────────────────

def split_dataset(
    source_dir: str,
    output_dir: str,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> dict:
    """
    Split a flat class folder into train/ and val/ subfolders.

    Input structure:
        source_dir/
            cat/  (n images)
            dog/  (n images)

    Output structure:
        output_dir/
            train/
                cat/
                dog/
            val/
                cat/
                dog/

    Args:
        source_dir: Root folder with one subfolder per class.
        output_dir: Where to write the split.
        val_fraction: Fraction of images to use for validation.
        seed: Random seed for reproducibility.

    Returns:
        dict with class names and counts per split.
    """
    random.seed(seed)
    source = Path(source_dir)
    output = Path(output_dir)

    class_dirs = sorted([d for d in source.iterdir() if d.is_dir()])
    if not class_dirs:
        raise ValueError(f"No class subdirectories found in {source_dir}")

    stats = {}
    for class_dir in class_dirs:
        class_name = class_dir.name
        images = list(class_dir.glob("*"))
        images = [f for f in images if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]

        random.shuffle(images)
        n_val   = max(1, int(len(images) * val_fraction))
        val_imgs   = images[:n_val]
        train_imgs = images[n_val:]

        for split, imgs in [("train", train_imgs), ("val", val_imgs)]:
            split_dir = output / split / class_name
            split_dir.mkdir(parents=True, exist_ok=True)
            for img_path in imgs:
                shutil.copy2(img_path, split_dir / img_path.name)

        stats[class_name] = {"train": len(train_imgs), "val": len(val_imgs)}
        print(f"  {class_name}: {len(train_imgs)} train / {n_val} val")

    print(f"\n✅ Dataset split saved to {output_dir}")
    return stats


# ─────────────────────────────────────────────
# tf.data Pipeline
# ─────────────────────────────────────────────

def make_tf_dataset(
    directory: str,
    image_size: tuple = (224, 224),
    batch_size: int   = 32,
    augment: bool     = False,
    seed: int         = 42,
) -> tuple[tf.data.Dataset, list[str]]:
    """
    Create a tf.data.Dataset from an image directory.

    Args:
        directory: Path to directory with class subfolders.
        image_size: (H, W) to resize images to.
        batch_size: Batch size.
        augment: Whether to apply data augmentation.
        seed: Random seed.

    Returns:
        (dataset, class_names) tuple.
    """
    raw_ds = keras.utils.image_dataset_from_directory(
        directory,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=(augment),   # Shuffle training, not validation
        seed=seed,
    )
    class_names = raw_ds.class_names

    aug_layer = keras.Sequential([
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(0.12),
        keras.layers.RandomZoom(0.12),
        keras.layers.RandomContrast(0.1),
    ]) if augment else None

    AUTOTUNE = tf.data.AUTOTUNE

    def preprocess(images, labels):
        images = tf.cast(images, tf.float32) / 255.0
        if augment and aug_layer is not None:
            images = aug_layer(images, training=True)
        return images, labels

    ds = raw_ds.map(preprocess, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    return ds, class_names


# ─────────────────────────────────────────────
# Quick sanity-check
# ─────────────────────────────────────────────

def describe_dataset(directory: str) -> None:
    """Print class distribution for a directory of images."""
    root = Path(directory)
    print(f"\n📂 Dataset: {directory}")
    total = 0
    for cls_dir in sorted(root.rglob("*")):
        if cls_dir.is_dir() and not any(cls_dir.iterdir().__class__ == type(cls_dir)):
            images = list(cls_dir.glob("*"))
            images = [f for f in images if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
            if images:
                print(f"  {cls_dir.relative_to(root)}: {len(images)} images")
                total += len(images)
    print(f"  Total: {total} images\n")
