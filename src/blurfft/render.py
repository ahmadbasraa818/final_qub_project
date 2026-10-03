"""Pictures of the analysis: the spectrum and the blur map (Pillow and numpy only)."""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

from . import _core
from .detector import BlurMapResult
from .imageio import ImageLike, load_gray
from .precision import parse_precision

__all__ = ["full_spectrum", "spectrum_picture", "map_overlay", "save_map_overlay", "png_base64", "thumbnail"]

VERMILION = np.array([232, 64, 31], dtype=np.float64)
PAPER = np.array([244, 244, 241], dtype=np.float64)
INK = np.array([18, 19, 20], dtype=np.float64)


def full_spectrum(half: np.ndarray, width: int) -> np.ndarray:
    """The full H x W spectrum from the half-plane, using F(-u, -v) = conj F(u, v)."""
    h, cols = half.shape
    full = np.empty((h, width), dtype=np.complex128)
    full[:, :cols] = half
    if width > cols:
        v = np.arange(cols, width)
        u = (-np.arange(h)) % h
        full[:, cols:] = np.conj(half[u][:, width - v])
    return full


def spectrum_picture(gray: np.ndarray, precision: str = "float64", size: int = 512) -> np.ndarray:
    """The centred log-magnitude spectrum as ink on paper, uint8 RGB.

    The median log magnitude prints as no ink and the 99.8th percentile as
    full ink, so structure stands out from the noise floor at any precision.
    """
    p = parse_precision(precision)
    h, w = gray.shape
    scale = min(1.0, size / max(h, w))
    if scale < 1:
        gray = np.asarray(Image.fromarray((gray * 255).astype(np.uint8)).resize((max(2, int(w * scale)), max(2, int(h * scale))), Image.LANCZOS), dtype=np.float64) / 255.0
        h, w = gray.shape
    half = _core.spectrum(np.ascontiguousarray(gray), *p.core_args())
    magnitude = np.fft.fftshift(np.log1p(np.abs(full_spectrum(half, w))))
    floor, top = np.median(magnitude), np.percentile(magnitude, 99.8)
    level = np.clip((magnitude - floor) / max(top - floor, 1e-9), 0, 1) ** 1.8
    rgb = PAPER * (1 - level[..., None]) + INK * level[..., None]
    return rgb.astype(np.uint8)


def thumbnail(gray: np.ndarray, size: int = 640) -> np.ndarray:
    h, w = gray.shape
    scale = min(1.0, size / max(h, w))
    image = Image.fromarray((np.clip(gray, 0, 1) * 255).astype(np.uint8))
    if scale < 1:
        image = image.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    return np.asarray(image)


def map_overlay(image: Union[ImageLike, np.ndarray], result: BlurMapResult, size: int = 900) -> np.ndarray:
    """The image in grey with blurred tiles tinted vermilion by their probability.

    Flat tiles (too little detail to judge) are left untinted.
    """
    gray = load_gray(image)
    h, w = gray.shape
    prob = result.probability
    # Each pixel takes the mean probability of the tiles covering it.
    total = np.zeros((h, w))
    count = np.zeros((h, w))
    for r in range(prob.shape[0]):
        for c in range(prob.shape[1]):
            value = prob[r, c]
            if not np.isfinite(value):
                continue
            y, x = r * result.stride, c * result.stride
            total[y : y + result.tile, x : x + result.tile] += value
            count[y : y + result.tile, x : x + result.tile] += 1
    alpha = np.where(count > 0, total / np.maximum(count, 1), 0.0)
    alpha = np.clip((alpha - 0.5) / 0.5, 0, 1) * 0.75  # tint only where blur is more likely than not
    base = np.repeat((0.35 + 0.65 * gray)[..., None], 3, axis=2) * 255.0
    rgb = base * (1 - alpha[..., None]) + VERMILION * alpha[..., None]
    picture = Image.fromarray(rgb.astype(np.uint8))
    scale = min(1.0, size / max(h, w))
    if scale < 1:
        picture = picture.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    return np.asarray(picture)


def save_map_overlay(image: ImageLike, result: BlurMapResult, out: "str | Path") -> Path:
    out = Path(out)
    Image.fromarray(map_overlay(image, result)).save(out)
    return out


def png_base64(array: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(array)).save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")
