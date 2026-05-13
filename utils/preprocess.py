"""
preprocess.py — Image preprocessing utilities.

Shared helpers for both training pipeline and inference.
"""

import io
import os
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps


# ─────────────────────────────────────────────
# Core Preprocessing
# ─────────────────────────────────────────────

def load_and_resize(
    source,
    target_size: tuple[int, int],
    color_mode: str = "rgb",
) -> np.ndarray:
    """
    Load an image from a file path or bytes and resize it.

    Args:
        source: File path (str/Path) or raw bytes.
        target_size: (width, height) tuple.
        color_mode: 'rgb', 'grayscale', or 'rgba'.

    Returns:
        np.ndarray float32 in [0, 1], shape (H, W, C).
    """
    if isinstance(source, (str, Path)):
        img = Image.open(source)
    elif isinstance(source, bytes):
        img = Image.open(io.BytesIO(source))
    elif isinstance(source, Image.Image):
        img = source
    else:
        raise TypeError(f"Unsupported source type: {type(source)}")

    # Auto-orient based on EXIF
    img = ImageOps.exif_transpose(img)

    # Color conversion
    mode_map = {"rgb": "RGB", "grayscale": "L", "rgba": "RGBA"}
    img = img.convert(mode_map[color_mode])

    # Resize
    img = img.resize(target_size, Image.LANCZOS)

    arr = np.array(img, dtype=np.float32) / 255.0

    # Ensure channel dim for grayscale
    if color_mode == "grayscale":
        arr = np.expand_dims(arr, -1)

    return arr


def normalize(arr: np.ndarray, mean=None, std=None) -> np.ndarray:
    """
    Normalize an image array.

    If mean/std are None, normalizes to [0, 1].
    Otherwise applies per-channel z-score normalization (e.g. ImageNet stats).

    ImageNet mean: (0.485, 0.456, 0.406)
    ImageNet std:  (0.229, 0.224, 0.225)
    """
    if mean is None or std is None:
        return arr  # already in [0,1]

    mean = np.array(mean, dtype=np.float32)
    std  = np.array(std,  dtype=np.float32)
    return (arr - mean) / std


def batch_from_file(path: str, input_shape: list) -> np.ndarray:
    """
    Load a single image file and return a (1, H, W, C) batch.

    Args:
        path: Path to image file.
        input_shape: [H, W, C] from model metadata.

    Returns:
        np.ndarray of shape (1, H, W, C).
    """
    h, w, c = input_shape
    color = "grayscale" if c == 1 else "rgb"
    arr = load_and_resize(path, (w, h), color_mode=color)
    return np.expand_dims(arr, 0)


# ─────────────────────────────────────────────
# Augmentation Helpers (NumPy, no TF required)
# ─────────────────────────────────────────────

def random_horizontal_flip(img: np.ndarray, p: float = 0.5) -> np.ndarray:
    if np.random.random() < p:
        return img[:, ::-1, :]
    return img


def random_crop(img: np.ndarray, crop_fraction: float = 0.9) -> np.ndarray:
    h, w = img.shape[:2]
    new_h = int(h * crop_fraction)
    new_w = int(w * crop_fraction)
    top  = np.random.randint(0, h - new_h + 1)
    left = np.random.randint(0, w - new_w + 1)
    cropped = img[top:top+new_h, left:left+new_w]
    # Resize back
    from PIL import Image as PILImage
    mode = "L" if img.shape[-1] == 1 else "RGB"
    pil = PILImage.fromarray((cropped.squeeze() * 255).astype(np.uint8), mode)
    pil = pil.resize((w, h), PILImage.LANCZOS)
    out = np.array(pil, dtype=np.float32) / 255.0
    if img.shape[-1] == 1:
        out = np.expand_dims(out, -1)
    return out


def cutout(img: np.ndarray, n_holes: int = 1, hole_size: float = 0.2) -> np.ndarray:
    """CutOut regularisation: zero out random rectangular patches."""
    h, w = img.shape[:2]
    size_h = int(h * hole_size)
    size_w = int(w * hole_size)
    out = img.copy()
    for _ in range(n_holes):
        cy = np.random.randint(0, h)
        cx = np.random.randint(0, w)
        y1, y2 = max(0, cy - size_h // 2), min(h, cy + size_h // 2)
        x1, x2 = max(0, cx - size_w // 2), min(w, cx + size_w // 2)
        out[y1:y2, x1:x2] = 0.0
    return out


# ─────────────────────────────────────────────
# Dataset Stats
# ─────────────────────────────────────────────

def compute_channel_stats(image_dir: str, sample_size: int = 500):
    """
    Compute per-channel mean and std from a directory of images.
    Useful for custom dataset normalization.
    """
    paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        paths.extend(Path(image_dir).rglob(ext))

    if not paths:
        raise FileNotFoundError(f"No images found in {image_dir}")

    sample = np.random.choice(paths, size=min(sample_size, len(paths)), replace=False)

    pixels = []
    for p in sample:
        try:
            img = np.array(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
            pixels.append(img.reshape(-1, 3))
        except Exception:
            continue

    if not pixels:
        raise RuntimeError("Could not load any images")

    all_pixels = np.concatenate(pixels, axis=0)
    mean = all_pixels.mean(axis=0).tolist()
    std  = all_pixels.std(axis=0).tolist()

    print(f"Channel mean: {[f'{v:.4f}' for v in mean]}")
    print(f"Channel std:  {[f'{v:.4f}' for v in std]}")
    return mean, std
