"""Existing blur detectors, for comparison.

Every function takes a greyscale image in [0, 1] and returns a score where
higher means sharper, as each method is usually reported.

* Variance of the Laplacian: Pech-Pacheco et al., "Diatom autofocusing in
  brightfield microscopy: a comparative study", ICPR 2000. The most widely
  used blur check.
* Tenengrad: Krotkov, "Focusing", IJCV 1987. Mean squared Sobel gradient.
* FFT high-pass: Rosebrock, "OpenCV Fast Fourier Transform (FFT) for blur
  detection", PyImageSearch 2020, and whdcumt/BlurDetection, the two FFT
  detectors the project supervisor pointed to. The low frequencies are
  removed from the centred spectrum, the image is transformed back, and the
  score is the mean log magnitude of what remains.
* The original project: the low-frequency share of the zero-padded spectrum,
  computed with the original transform in FloatX (8-bit exponent, 12-bit
  mantissa) arithmetic. It reported "0 = very sharp, 1 = very blurry", so its
  score is negated here to keep higher = sharper.
"""

from __future__ import annotations

import numpy as np

from . import _core

__all__ = ["laplacian_variance", "tenengrad", "fft_highpass", "original_project", "BASELINES"]


def laplacian_variance(gray: np.ndarray) -> float:
    g = gray * 255.0
    lap = g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] - 4.0 * g[1:-1, 1:-1]
    return float(lap.var())


def tenengrad(gray: np.ndarray) -> float:
    g = gray * 255.0
    gx = (g[:-2, 2:] + 2 * g[1:-1, 2:] + g[2:, 2:]) - (g[:-2, :-2] + 2 * g[1:-1, :-2] + g[2:, :-2])
    gy = (g[2:, :-2] + 2 * g[2:, 1:-1] + g[2:, 2:]) - (g[:-2, :-2] + 2 * g[:-2, 1:-1] + g[:-2, 2:])
    return float(np.mean(gx * gx + gy * gy))


def fft_highpass(gray: np.ndarray, size: int = 60, reference_width: int = 500) -> float:
    """Rosebrock's detector. It was tuned on images resized to 500 px wide with a
    60 px square removed from the centre; the square is scaled to the image so
    the cut-off frequency is the same at any size."""
    g = gray * 255.0
    h, w = g.shape
    spectrum = np.fft.fftshift(np.fft.fft2(g))
    cy, cx = h // 2, w // 2
    half = max(1, int(round(size * w / reference_width)))
    spectrum[max(0, cy - half) : cy + half, max(0, cx - half) : cx + half] = 0
    recon = np.fft.ifft2(np.fft.ifftshift(spectrum))
    magnitude = 20.0 * np.log(np.abs(recon) + 1e-12)
    return float(np.mean(magnitude))


def original_project(gray: np.ndarray) -> float:
    """The original project's measure, with its transform and FloatX<8, 12> arithmetic."""
    magnitude = _core.legacy_fft2_magnitude(gray * 255.0, "emulated", 8, 12)
    rows, cols = magnitude.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros_like(magnitude)
    mask[: crow // 2, :] = 1
    mask[(crow // 2) * 3 :, :] = 1
    mask[:, : ccol // 2] = 1
    mask[:, (ccol // 2) * 3 :] = 1
    total = magnitude.sum()
    blurriness = (magnitude * mask).sum() / total if total > 0 else 0.0
    return -float(blurriness)


BASELINES = {
    "Variance of the Laplacian": laplacian_variance,
    "Tenengrad": tenengrad,
    "FFT high-pass (Rosebrock)": fft_highpass,
    "Original project": original_project,
}
