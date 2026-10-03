"""The decision model: from the spectrum to a blur probability, radius and type.

Blur is a low-pass filter, so the evidence is how much power an image keeps
at each spatial frequency. The features are, for seven radial bands between
0.02 and 0.5 cycles per pixel, the log of the band's mean power and of its
power in its weakest direction (motion blur removes detail in one direction
only). A logistic regression on them learns how much each band matters: a
high-pass filter fitted to the data, where the variance of the Laplacian fixes
one in advance. The same features feed a linear regression for the blur
radius, and spectral anisotropy tells motion blur from defocus.

Small and transparent, as the brief asked: FFT-based, no neural network. The
parameters are fitted on the ground-truth benchmark (evaluate.py) and stored
in model.json next to this file.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

__all__ = ["BAND_EDGES", "FEATURES", "QUANTISATION_VARIANCE", "feature_vector", "features_from_band_power", "BlurModel",
           "fit_logistic", "default_model", "MODEL_PATH"]

#: Radial band edges in cycles per pixel (0.5 is the Nyquist frequency).
BAND_EDGES = (0.02, 0.05, 0.10, 0.18, 0.26, 0.34, 0.42, 0.50)
_BANDS = [f"{lo:.2f}-{hi:.2f}" for lo, hi in zip(BAND_EDGES[:-1], BAND_EDGES[1:])]
FEATURES = tuple(f"log_power_{b}" for b in _BANDS) + tuple(f"log_weakest_direction_power_{b}" for b in _BANDS)
MODEL_PATH = Path(__file__).with_name("model.json")


#: Variance of 8-bit quantisation noise (step 1/255, uniform rounding error).
QUANTISATION_VARIANCE = (1.0 / 255.0) ** 2 / 12.0


def features_from_band_power(band_power: np.ndarray, band_power_min: np.ndarray, window_energy: float) -> np.ndarray:
    """Log power spectral density per band (mean, then weakest direction), above the 8-bit noise floor.

    Works on one image's vectors or on arrays of them (last axis = bands).

    Dividing a bin's power by the window's energy gives the power spectral
    density per pixel, which does not depend on the image's size: a 4000-pixel
    photo and a 200-pixel crop of the same scene have the same density.

    Every real photograph carries at least the noise of 8-bit quantisation,
    white with variance (1/255)^2 / 12, which in these units is a constant
    floor under every band. Adding it keeps the features physical for images
    that never had it (blurred in floating point, rendered) and changes real
    photographs very little.
    """
    mean = np.log(np.asarray(band_power, dtype=np.float64) / window_energy + QUANTISATION_VARIANCE)
    weakest = np.log(np.asarray(band_power_min, dtype=np.float64) / window_energy + QUANTISATION_VARIANCE)
    return np.concatenate([mean, weakest], axis=-1)


def feature_vector(metrics: Mapping[str, object]) -> np.ndarray:
    return features_from_band_power(np.asarray(metrics["band_power"]), np.asarray(metrics["band_power_min"]),
                                    float(metrics["window_energy"]))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def fit_logistic(x: np.ndarray, y: np.ndarray, l2: float = 1.0, balanced: bool = True,
                 iterations: int = 100) -> tuple[np.ndarray, float]:
    """Logistic regression by Newton's method (IRLS) with an L2 penalty.

    x is n x d and standardised, y holds 0/1. With ``balanced`` each class
    carries equal total weight, so a benchmark with more blurred than sharp
    images does not bias the decision towards "blurred".
    """
    n, d = x.shape
    y = np.asarray(y, dtype=np.float64)
    design = np.hstack([x, np.ones((n, 1))])
    if balanced:
        positives, negatives = max(1.0, y.sum()), max(1.0, n - y.sum())
        sample_weight = np.where(y == 1, n / (2 * positives), n / (2 * negatives))
    else:
        sample_weight = np.ones(n)
    beta = np.zeros(d + 1)
    penalty = np.diag([l2] * d + [0.0])
    for _ in range(iterations):
        p = _sigmoid(design @ beta)
        gradient = design.T @ (sample_weight * (p - y)) + penalty @ beta
        hessian = design.T @ (design * (sample_weight * p * (1 - p))[:, None]) + penalty + 1e-9 * np.eye(d + 1)
        step = np.linalg.solve(hessian, gradient)
        beta -= step
        if np.max(np.abs(step)) < 1e-10:
            break
    return beta[:d], float(beta[d])


@dataclass
class BlurModel:
    mean: list[float]
    scale: list[float]
    weights: list[float]
    bias: float
    band_edges: list[float] = field(default_factory=lambda: list(BAND_EDGES))
    threshold: float = 0.5
    # log(sigma) = sigma_coef . [z, z^2] + sigma_bias, with z the standardised features;
    # a ridge regression fitted on images with sigma >= 1
    sigma_coef: list[float] = field(default_factory=list)
    sigma_bias: float = 0.0
    motion_anisotropy: float = 0.5
    # Region rule (after Liu, Li and Jia 2008): an image is also blurred when at
    # least tile_majority of its judged tiles are, given min_tiles of them, and
    # partly blurred from tile_partial. 0.7 is the lowest majority that adds no
    # false alarms on the benchmark's sharp images.
    tile_majority: float = 0.7
    tile_partial: float = 0.3
    min_tiles: int = 4
    provenance: dict = field(default_factory=dict)

    def standardise(self, features: np.ndarray) -> np.ndarray:
        return (np.asarray(features, dtype=np.float64) - np.asarray(self.mean)) / np.asarray(self.scale)

    def probability(self, features: np.ndarray) -> np.ndarray:
        """Probability that the image is blurred (vectorised over leading axes)."""
        return _sigmoid(self.standardise(features) @ np.asarray(self.weights) + self.bias)

    def sigma(self, features: np.ndarray) -> np.ndarray:
        """Estimated blur radius as an equivalent Gaussian sigma, in pixels."""
        z = self.standardise(features)
        log_sigma = np.concatenate([z, z * z], axis=-1) @ np.asarray(self.sigma_coef) + self.sigma_bias
        return np.exp(np.clip(log_sigma, -5, 5))

    @classmethod
    def fit(cls, features: np.ndarray, labels: np.ndarray, sigmas: Sequence[float], l2: float = 1.0,
            ridge: float = 1.0, provenance: dict | None = None) -> "BlurModel":
        """Fits the classifier on labelled rows and the radius regression on rows with sigma >= 1.

        The radius regression is quadratic in the standardised features: once
        strong blur drives the upper bands to the noise floor, a linear fit flattens out.
        """
        features = np.asarray(features, dtype=np.float64)
        mean = features.mean(axis=0)
        scale = features.std(axis=0)
        scale[scale == 0] = 1.0
        z = (features - mean) / scale
        weights, bias = fit_logistic(z, np.asarray(labels, dtype=np.float64), l2=l2)
        sig = np.asarray(sigmas, dtype=np.float64)
        blurred = sig >= 1.0
        zb = z[blurred]
        design = np.hstack([zb, zb * zb, np.ones((len(zb), 1))])
        penalty = ridge * np.eye(design.shape[1])
        penalty[-1, -1] = 0.0
        coef = np.linalg.solve(design.T @ design + penalty, design.T @ np.log(sig[blurred]))
        return cls(mean=mean.tolist(), scale=scale.tolist(), weights=weights.tolist(), bias=bias,
                   sigma_coef=coef[:-1].tolist(), sigma_bias=float(coef[-1]), provenance=provenance or {})

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2) + "\n"

    @classmethod
    def from_json(cls, text: str) -> "BlurModel":
        return cls(**json.loads(text))

    def save(self, path: "str | Path" = MODEL_PATH) -> None:
        Path(path).write_text(self.to_json())

    @classmethod
    def load(cls, path: "str | Path" = MODEL_PATH) -> "BlurModel":
        return cls.from_json(Path(path).read_text())


_DEFAULT: BlurModel | None = None


def default_model() -> BlurModel:
    """The model fitted on the benchmark and shipped with the package."""
    global _DEFAULT
    if _DEFAULT is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError("model.json is missing; run `blurfft evaluate --fit` to create it")
        _DEFAULT = BlurModel.load(MODEL_PATH)
    return _DEFAULT
