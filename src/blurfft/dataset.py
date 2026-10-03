"""A ground-truth benchmark: sharp photographs blurred by known amounts.

The plan called for checking detection against ground truth. Real blurred
photos rarely come with an exact amount of blur, so this module makes its own:
random crops of public-domain photographs, each blurred with a kernel of known
strength (Gaussian, defocus disk or motion streak), then put through a random
camera pipeline (sensor noise, JPEG) so sharp and blurred images look alike in
every way except the blur.

Every sample records its blur as an equivalent Gaussian radius ``sigma`` in
pixels (the kernel's standard deviation): a defocus disk of radius r has
sigma = r / 2 and a motion streak of length L has sigma = L / sqrt(12) along
its direction. Labels follow from sigma: sharp up to 0.5 px, blurred from
1.5 px, and the band in between left unlabelled, since people disagree there.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass, field
from typing import Iterator, Optional

import numpy as np
from PIL import Image

from .imageio import to_gray

__all__ = ["SOURCES", "Sample", "BlurSpec", "make_benchmark", "load_source", "gaussian_kernel", "disk_kernel",
           "motion_kernel", "convolve", "SHARP_MAX_SIGMA", "BLURRED_MIN_SIGMA"]

#: scikit-image sample photographs with no copyright restrictions (their licences
#: are listed in skimage.data), chosen to cover faces, animals, objects, text,
#: textures, astronomy and microscopy.
SOURCES = {
    "astronaut": "public domain (NASA)",
    "camera": "CC0 (Lav Varshney)",
    "chelsea": "CC0 (Stefan van der Walt)",
    "coffee": "CC0 (Rachel Michetti)",
    "coins": "no known copyright restrictions",
    "brick": "CC0 (CC0Textures)",
    "grass": "CC0 (CC0Textures)",
    "gravel": "CC0 (CC0Textures)",
    "rocket": "public domain (SpaceX)",
    "text": "public domain",
    "hubble_deep_field": "public domain (NASA)",
    "immunohistochemistry": "no known copyright restrictions",
}

SHARP_MAX_SIGMA = 0.5
BLURRED_MIN_SIGMA = 1.5


@dataclass(frozen=True)
class BlurSpec:
    kind: str  # none, gaussian, disk or motion
    amount: float = 0.0  # sigma, radius or length in pixels
    angle: float = 0.0  # motion direction in degrees, image coordinates (y down)

    @property
    def sigma(self) -> float:
        if self.kind == "gaussian":
            return self.amount
        if self.kind == "disk":
            return self.amount / 2.0
        if self.kind == "motion":
            return self.amount / math.sqrt(12.0)
        return 0.0

    @property
    def label(self) -> Optional[int]:
        """1 for blurred, 0 for sharp, None for the unlabelled band in between."""
        if self.sigma <= SHARP_MAX_SIGMA:
            return 0
        if self.sigma >= BLURRED_MIN_SIGMA:
            return 1
        return None


@dataclass
class Sample:
    image: np.ndarray
    source: str
    blur: BlurSpec
    noise: float
    jpeg_quality: Optional[int]
    crop: tuple[int, int, int, int]  # top, left, height, width
    meta: dict = field(default_factory=dict)

    @property
    def label(self) -> Optional[int]:
        return self.blur.label

    @property
    def sigma(self) -> float:
        return self.blur.sigma


#: The blurs applied to every crop: sharp references, then rising strengths.
BLURS = (
    BlurSpec("none"),
    BlurSpec("gaussian", 0.5),
    BlurSpec("gaussian", 1.0),
    BlurSpec("gaussian", 1.5),
    BlurSpec("gaussian", 2.0),
    BlurSpec("gaussian", 3.0),
    BlurSpec("gaussian", 4.5),
    BlurSpec("disk", 2.0),
    BlurSpec("disk", 3.0),
    BlurSpec("disk", 5.0),
    BlurSpec("motion", 4.0),
    BlurSpec("motion", 7.0),
    BlurSpec("motion", 13.0),
)


def load_source(name: str) -> np.ndarray:
    """A source photograph from scikit-image's sample data, greyscale in [0, 1]."""
    try:
        import skimage.data
    except ImportError as error:  # pragma: no cover - exercised only without the extra
        raise ImportError("the benchmark needs scikit-image: pip install 'blurfft[bench]'") from error
    loader = {"immunohistochemistry": skimage.data.immunohistochemistry}.get(name) or getattr(skimage.data, name)
    return to_gray(loader())


def gaussian_kernel(sigma: float) -> np.ndarray:
    radius = max(1, int(math.ceil(3.5 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    g = np.exp(-(x**2) / (2 * sigma * sigma))
    k = np.outer(g, g)
    return k / k.sum()


def _supersampled(radius: int, inside, samples: int = 8) -> np.ndarray:
    """A kernel whose cells hold the fraction of their area inside a shape."""
    size = 2 * radius + 1
    offsets = (np.arange(samples) + 0.5) / samples - 0.5
    ys, xs = np.meshgrid(np.arange(size) - radius, np.arange(size) - radius, indexing="ij")
    k = np.zeros((size, size))
    for dy in offsets:
        for dx in offsets:
            k += inside(xs + dx, ys + dy)
    return k / k.sum()


def disk_kernel(radius: float) -> np.ndarray:
    """Defocus: a uniform disk (pillbox) with anti-aliased edges."""
    r = int(math.ceil(radius))
    return _supersampled(r, lambda x, y: (x * x + y * y) <= radius * radius)


def motion_kernel(length: float, angle_deg: float) -> np.ndarray:
    """Linear motion: a streak of the given length and direction, one pixel wide."""
    r = int(math.ceil(length / 2)) + 1
    theta = math.radians(angle_deg)
    ux, uy = math.cos(theta), math.sin(theta)

    def inside(x, y):
        along = x * ux + y * uy
        across = -x * uy + y * ux
        return (np.abs(along) <= length / 2) & (np.abs(across) <= 0.5)

    return _supersampled(r, inside)


def kernel_for(spec: BlurSpec) -> Optional[np.ndarray]:
    if spec.kind == "gaussian":
        return gaussian_kernel(spec.amount)
    if spec.kind == "disk":
        return disk_kernel(spec.amount)
    if spec.kind == "motion":
        return motion_kernel(spec.amount, spec.angle)
    return None


def convolve(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """2-D convolution with reflected borders, through numpy's FFT, same size as the input."""
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(image, ((ph, ph), (pw, pw)), mode="reflect")
    shape = (padded.shape[0] + kh - 1, padded.shape[1] + kw - 1)
    result = np.fft.irfft2(np.fft.rfft2(padded, shape) * np.fft.rfft2(kernel, shape), shape)
    return result[kh - 1 : kh - 1 + image.shape[0], kw - 1 : kw - 1 + image.shape[1]]


def camera(image: np.ndarray, noise: float, jpeg_quality: Optional[int], rng: np.random.Generator,
           contrast: float = 1.0) -> np.ndarray:
    """Exposure contrast, sensor noise, 8-bit quantisation and (optionally) JPEG compression."""
    out = image.mean() + contrast * (image - image.mean()) if contrast != 1.0 else image
    out = out + rng.normal(0.0, noise, image.shape) if noise > 0 else out
    out = np.clip(np.round(out * 255.0), 0, 255).astype(np.uint8)
    if jpeg_quality is not None:
        buffer = io.BytesIO()
        Image.fromarray(out).save(buffer, format="JPEG", quality=jpeg_quality)
        out = np.asarray(Image.open(io.BytesIO(buffer.getvalue())).convert("L"))
    return out.astype(np.float64) / 255.0


def make_benchmark(
    sources: Optional[list[str]] = None,
    crops_per_source: int = 4,
    seed: int = 2024,
    min_size: int = 128,
    max_size: int = 768,
    vary_contrast: bool = False,
) -> Iterator[Sample]:
    """Yields the benchmark's samples, deterministically for a given seed.

    Each source gives ``crops_per_source`` crops of random size (so most
    dimensions are not powers of two) and every crop is blurred every way in
    BLURS. Motion streaks get a random direction. Each sample then gets a random
    camera: noise of 0, 0.003 or 0.008 and JPEG quality of none, 90 or 75. With
    ``vary_contrast``, each sample's contrast is also scaled by a random factor
    between 0.35 and 1, as dim, hazy or flat-lit scenes would be.
    """
    rng = np.random.default_rng(seed)
    margin = 24  # room for the largest kernel, cut away after blurring
    for name in sources or list(SOURCES):
        source = load_source(name)
        sh, sw = source.shape
        for _ in range(crops_per_source):
            room_h, room_w = sh - 2 * margin, sw - 2 * margin  # short sources get smaller crops
            h = int(rng.integers(min(min_size, room_h), min(max_size, room_h) + 1))
            w = int(rng.integers(min(min_size, room_w), min(max_size, room_w) + 1))
            top = int(rng.integers(0, sh - h - 2 * margin + 1))
            left = int(rng.integers(0, sw - w - 2 * margin + 1))
            region = source[top : top + h + 2 * margin, left : left + w + 2 * margin]
            for spec in BLURS:
                if spec.kind == "motion":
                    spec = BlurSpec("motion", spec.amount, float(rng.uniform(0, 180)))
                kernel = kernel_for(spec)
                blurred = convolve(region, kernel) if kernel is not None else region
                blurred = blurred[margin : margin + h, margin : margin + w]
                noise = float(rng.choice([0.0, 0.003, 0.008]))
                quality = [None, 90, 75][int(rng.integers(0, 3))]
                contrast = float(rng.uniform(0.35, 1.0)) if vary_contrast else 1.0
                yield Sample(
                    image=np.ascontiguousarray(camera(blurred, noise, quality, rng, contrast)),
                    source=name,
                    blur=spec,
                    noise=noise,
                    jpeg_quality=quality,
                    crop=(top + margin, left + margin, h, w),
                    meta={"contrast": contrast},
                )
