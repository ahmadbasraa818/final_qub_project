"""Exports reference values for testing a port of the blur detector.

A port (such as the JavaScript one on thecodingexplorer.com) can check itself
against these numbers: for each 8-bit test image, the model's 14 features, the
probability of blur and the radius estimate, computed here in float64; and one
Gaussian blur, to check the port blurs the way the benchmark does.

    python scripts/export_reference_values.py > reference-values.json
"""

from __future__ import annotations

import json

import numpy as np

import blurfft
from blurfft import _core, default_model
from blurfft.dataset import convolve, gaussian_kernel
from blurfft.detector import MEASURE_DEFAULTS
from blurfft.model import feature_vector


def texture(height: int, width: int, seed: int) -> np.ndarray:
    """Smoothed noise with some fine grain, stored as 8-bit pixels."""
    rng = np.random.default_rng(seed)
    noise = rng.random((height + 8, width + 8))
    smooth = convolve(noise, gaussian_kernel(1.2))[4 : 4 + height, 4 : 4 + width]
    image = 0.5 + 2.5 * (smooth - smooth.mean()) + 0.03 * rng.standard_normal((height, width))
    return eight_bit(image)


def eight_bit(image: np.ndarray) -> np.ndarray:
    return np.clip(np.round(image * 255.0), 0, 255).astype(np.uint8)


def detector_case(name: str, pixels: np.ndarray) -> dict:
    model = default_model()
    image = pixels / 255.0
    metrics = _core.analyse(np.ascontiguousarray(image), "float64", 11, 52, threads=1,
                            band_edges=list(model.band_edges), **MEASURE_DEFAULTS)
    features = feature_vector(metrics)
    return {
        "name": name,
        "width": int(pixels.shape[1]),
        "height": int(pixels.shape[0]),
        "pixels": pixels.ravel().tolist(),
        "features": [float(v) for v in features],
        "probability": float(model.probability(features)),
        "radius": float(model.sigma(features)),
    }


def blur_case(pixels: np.ndarray, sigma: float) -> dict:
    blurred = convolve(pixels / 255.0, gaussian_kernel(sigma))
    return {
        "sigma": sigma,
        "width": int(pixels.shape[1]),
        "height": int(pixels.shape[0]),
        "pixels": pixels.ravel().tolist(),
        "blurred": [float(v) for v in blurred.ravel()],
    }


def main() -> None:
    sharp = texture(40, 56, 1)
    blurred = eight_bit(convolve(sharp / 255.0, gaussian_kernel(2.0)))
    cases = [
        detector_case("sharp texture, 40 x 56", sharp),
        detector_case("the same, Gaussian blur of 2 px", blurred),
        detector_case("texture, 33 x 47 (odd sizes)", texture(33, 47, 2)),
        detector_case("texture, 48 x 64 (Nyquist row and column)", texture(48, 64, 3)),
    ]
    print(json.dumps({
        "generated_by": f"blurfft {blurfft.__version__}, scripts/export_reference_values.py",
        "model": default_model().provenance,
        "cases": cases,
        "blur": blur_case(texture(12, 16, 4), 1.5),
    }))


if __name__ == "__main__":
    main()
