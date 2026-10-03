"""The blur detector: an image in, a verdict, a severity and the evidence out."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np

from . import _core
from .imageio import ImageLike, load_gray
from .model import BAND_EDGES, BlurModel, default_model, feature_vector, features_from_band_power
from .precision import Precision, parse_precision

__all__ = ["BlurDetector", "BlurReport", "BlurMapResult", "measure", "MEASURE_DEFAULTS", "severity_for"]

#: Settings for the spectral measures, shared by training and detection.
MEASURE_DEFAULTS = {"window": "hann", "cutoff": 0.25, "fit_low": 0.05, "fit_high": 0.35, "radial_bins": 64}


def measure(gray: np.ndarray, precision: "str | Precision" = "float64", threads: int = 0, **overrides) -> dict:
    """The spectral measures of a greyscale image, with the FFT in ``precision``."""
    p = parse_precision(precision)
    options = {**MEASURE_DEFAULTS, "band_edges": list(BAND_EDGES), **overrides}
    return _core.analyse(np.ascontiguousarray(gray, dtype=np.float64), *p.core_args(), threads=threads, **options)


def severity_for(sigma: float, blurry: bool) -> str:
    if not blurry or sigma < 0.8:
        return "none"
    if sigma < 1.5:
        return "slight"
    if sigma < 3.0:
        return "moderate"
    return "strong"


@dataclass
class BlurReport:
    blurry: bool
    verdict: str  # sharp, partly blurred or blurred
    probability: float  # that the whole image is blurred, 0 to 1
    tile_share: float  # share of the judged tiles that look blurred
    severity: str  # none, slight, moderate or strong
    sigma: float  # estimated blur radius (equivalent Gaussian sigma), pixels
    blur_type: Optional[str]  # motion (one dominant direction) or defocus (none: out of focus, or shake in several directions)
    motion_angle: Optional[float]  # degrees from the x axis, y down, for motion blur
    precision: str
    fft_ms: float
    total_ms: float
    height: int
    width: int
    metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        verdict = {"blurred": "BLURRED", "partly blurred": "partly blurred"}.get(self.verdict, "sharp")
        if self.blurry:
            detail = f"{self.severity}, ~{self.sigma:.1f} px"
        elif self.verdict == "partly blurred":
            detail = f"{self.tile_share:.0%} of the picture"
        else:
            detail = f"{1 - self.probability:.0%} confident"
        kind = ""
        if self.blur_type == "motion" and self.motion_angle is not None:
            kind = f", motion at {self.motion_angle:.0f} deg"
        elif self.blur_type:
            kind = ", no single direction"
        return f"{verdict} (p={self.probability:.2f}; {detail}{kind}) [{self.precision}, FFT {self.fft_ms:.1f} ms]"


@dataclass
class BlurMapResult:
    probability: np.ndarray  # per tile, NaN where a tile is too flat to judge
    sigma: np.ndarray  # estimated blur radius per tile, pixels
    tile: int
    stride: int
    metrics: dict

    @property
    def judged(self) -> int:
        return int(np.isfinite(self.probability).sum())

    @property
    def blurred_share(self) -> float:
        finite = self.probability[np.isfinite(self.probability)]
        return float(np.mean(finite >= 0.5)) if finite.size else 0.0


class BlurDetector:
    """Detects blur from an image's spectrum, with the FFT in any precision.

    >>> detector = BlurDetector(precision="float16")
    >>> report = detector.analyse("photo.jpg")
    >>> report.blurry, report.severity
    """

    def __init__(self, precision: "str | Precision" = "float64", model: Optional[BlurModel] = None, threads: int = 0):
        self.precision = parse_precision(precision)
        self.model = model or default_model()
        self.threads = threads

    def analyse(self, image: ImageLike) -> BlurReport:
        """Whole-image verdict, backed by tiles.

        The image is blurred when its whole spectrum says so, or when most of
        its judged tiles do: dark or empty areas (night skies, plain walls) carry
        no detail either way and would otherwise dilute a whole-image average.
        Between tile_partial and tile_majority of blurred tiles, it is partly blurred.
        """
        start = time.perf_counter()
        gray = load_gray(image)
        metrics = measure(gray, self.precision, self.threads, band_edges=list(self.model.band_edges))
        features = feature_vector(metrics)
        probability = float(self.model.probability(features))
        tiles = self.map(gray)
        enough = tiles.judged >= self.model.min_tiles
        share = tiles.blurred_share
        whole = probability >= self.model.threshold
        blurry = whole or (enough and share >= self.model.tile_majority)
        verdict = "blurred" if blurry else ("partly blurred" if enough and share >= self.model.tile_partial else "sharp")
        sigma, anisotropy, orientation = 0.0, float(metrics["anisotropy"]), float(metrics["orientation_deg"])
        if whole:
            sigma = float(self.model.sigma(features))
        elif blurry:
            # Decided by the tiles, so their blurred ones describe the blur.
            blurred_tiles = np.isfinite(tiles.probability) & (np.nan_to_num(tiles.probability) >= 0.5)
            sigma = float(np.median(tiles.sigma[blurred_tiles]))
            anisotropy = float(np.median(tiles.metrics["anisotropy"][blurred_tiles]))
            # Directions are axial (0 and 180 degrees are the same), so average doubled angles.
            doubled = np.deg2rad(2 * tiles.metrics["orientation_deg"][blurred_tiles])
            orientation = float(np.rad2deg(np.arctan2(np.sin(doubled).mean(), np.cos(doubled).mean())) / 2) % 180
        blur_type, angle = None, None
        if blurry:
            if anisotropy >= self.model.motion_anisotropy:
                blur_type, angle = "motion", orientation
            else:
                blur_type = "defocus"
        total = (time.perf_counter() - start) * 1000
        public = {k: (float(v) if np.isscalar(v) else v.tolist()) for k, v in metrics.items() if k not in ("height", "width")}
        return BlurReport(
            blurry=bool(blurry), verdict=verdict, probability=probability, tile_share=share,
            severity=severity_for(sigma, blurry), sigma=sigma,
            blur_type=blur_type, motion_angle=angle, precision=self.precision.name,
            fft_ms=metrics["fft_seconds"] * 1000, total_ms=total, height=int(metrics["height"]),
            width=int(metrics["width"]), metrics=public,
        )

    def map(self, image: ImageLike, tile: Optional[int] = None, stride: Optional[int] = None) -> BlurMapResult:
        """Blur probability for overlapping tiles, for images that are only partly blurred.

        Tiles whose detail energy is under 2% of the image's median tile are
        too flat (sky, walls) to judge either way and come back as NaN.
        """
        gray = load_gray(image)
        h, w = gray.shape
        if tile is None:
            tile = int(min(128, max(64, min(h, w) // 3)))
        tile = int(min(tile, h, w))
        stride = stride or max(1, tile // 2)
        result = _core.blur_map(
            gray, tile, stride, *self.precision.core_args(),
            **{**MEASURE_DEFAULTS, "radial_bins": 32}, threads=self.threads, band_edges=list(self.model.band_edges),
        )
        energy = result["energy"]
        features = features_from_band_power(result["band_power"], result["band_power_min"], result["window_energy"])
        probability = self.model.probability(features)
        flat = energy < 0.02 * np.median(energy) if energy.size else energy.astype(bool)
        probability = np.where(flat, np.nan, probability)
        return BlurMapResult(probability=probability, sigma=self.model.sigma(features), tile=tile, stride=stride,
                             metrics=result)

