"""Loading images as greyscale arrays in [0, 1]."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

__all__ = ["load_gray", "to_gray", "IMAGE_SUFFIXES", "find_images"]

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".gif"}

ImageLike = Union[str, Path, bytes, np.ndarray, Image.Image]


def to_gray(image: np.ndarray) -> np.ndarray:
    """A float64 greyscale copy in [0, 1] of an 8-bit, 16-bit or float image, grey or colour.

    Colour is reduced with the ITU-R BT.601 luma weights (0.299, 0.587, 0.114),
    as OpenCV and Pillow do.
    """
    a = np.asarray(image)
    if a.dtype == np.uint8:
        a = a.astype(np.float64) / 255.0
    elif a.dtype == np.uint16:
        a = a.astype(np.float64) / 65535.0
    else:
        a = a.astype(np.float64)
        if a.size and a.max() > 1.0:
            a = a / 255.0
    if a.ndim == 3:
        if a.shape[2] == 4:
            a = a[:, :, :3]
        a = a @ np.array([0.299, 0.587, 0.114]) if a.shape[2] == 3 else a[:, :, 0]
    if a.ndim != 2:
        raise ValueError(f"expected a 2-D or 3-D image, got shape {a.shape}")
    return np.ascontiguousarray(np.clip(a, 0.0, 1.0))


def load_gray(source: ImageLike) -> np.ndarray:
    """Loads a file, bytes, Pillow image or array as greyscale in [0, 1]."""
    if isinstance(source, np.ndarray):
        return to_gray(source)
    if isinstance(source, Image.Image):
        image = source
    elif isinstance(source, (bytes, bytearray)):
        image = Image.open(io.BytesIO(source))
    else:
        image = Image.open(Path(source))
    image.load()
    if image.mode in ("I;16", "I;16B", "I"):
        return to_gray(np.asarray(image, dtype=np.float64) / 65535.0)
    return to_gray(np.asarray(image.convert("RGB")))


def find_images(*paths: "str | Path") -> list[Path]:
    """Expands files, directories (recursively) and globs into a sorted list of image files."""
    found: list[Path] = []
    for item in paths:
        path = Path(item)
        if path.is_dir():
            found.extend(p for p in sorted(path.rglob("*")) if p.suffix.lower() in IMAGE_SUFFIXES)
        elif any(ch in str(item) for ch in "*?["):
            found.extend(p for p in sorted(Path().glob(str(item))) if p.suffix.lower() in IMAGE_SUFFIXES)
        elif path.exists():
            found.append(path)
        else:
            raise FileNotFoundError(f"no such file or directory: {item}")
    return found
