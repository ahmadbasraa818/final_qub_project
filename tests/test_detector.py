"""The detector on images with known blur."""

import numpy as np
import pytest

from blurfft import BlurDetector, default_model
from blurfft.dataset import convolve, gaussian_kernel, load_source, motion_kernel


@pytest.fixture(scope="module")
def photo():
    # A crop of a public-domain photograph (scikit-image's "astronaut").
    return load_source("astronaut")[60:380, 80:430]


@pytest.fixture(scope="module")
def detector():
    return BlurDetector("float64")


def test_sharp_photo_is_sharp(detector, photo):
    report = detector.analyse(photo)
    assert not report.blurry
    assert report.severity == "none" and report.blur_type is None


@pytest.mark.parametrize("sigma", [2.0, 3.0, 5.0])
def test_blurred_photo_is_blurred(detector, photo, sigma):
    report = detector.analyse(convolve(photo, gaussian_kernel(sigma)))
    assert report.blurry
    # The benchmark's median radius error is under 20%; one image may stray further.
    assert 0.5 * sigma < report.sigma < 2.0 * sigma
    if sigma >= 3.0:
        assert report.severity in ("moderate", "strong")


def test_probability_rises_with_blur(detector, photo):
    probabilities = [detector.analyse(convolve(photo, gaussian_kernel(s)) if s else photo).probability for s in (0, 1.0, 2.0, 4.0)]
    assert probabilities == sorted(probabilities)


def test_motion_blur_and_its_direction(detector, photo):
    report = detector.analyse(convolve(photo, motion_kernel(15, 30.0)))
    assert report.blurry and report.blur_type == "motion"
    assert min(abs(report.motion_angle - 30.0), 180 - abs(report.motion_angle - 30.0)) < 10


@pytest.mark.parametrize("precision", ["float32", "float16", "bfloat16", "floatx"])
def test_reduced_precision_gives_the_same_verdicts(photo, precision):
    blurred = convolve(photo, gaussian_kernel(2.5))
    for image in (photo, blurred):
        exact = BlurDetector("float64").analyse(image)
        reduced = BlurDetector(precision).analyse(image)
        assert reduced.blurry == exact.blurry
        assert abs(reduced.probability - exact.probability) < 0.02


def test_blur_map_finds_the_blurred_half(detector, photo):
    image = photo.copy()
    image[:, 175:] = convolve(photo, gaussian_kernel(3.0))[:, 175:]
    result = detector.map(image, tile=64, stride=32)
    columns = np.nanmean(result.probability, axis=0)
    left, right = columns[: len(columns) // 2 - 1], columns[len(columns) // 2 + 1 :]
    assert np.nanmean(right) > 0.7 and np.nanmean(left) < 0.4
    report = detector.analyse(image)
    assert report.verdict in ("partly blurred", "blurred") and 0.3 <= report.tile_share <= 0.8


def test_blur_among_dark_empty_areas_is_found(detector, photo):
    # A blurred subject in a mostly black frame: the black carries no detail, so
    # the tiles that can be judged decide.
    frame = np.zeros((480, 640))
    frame[120:440, 150:500] = convolve(photo, gaussian_kernel(3.0))
    report = detector.analyse(frame)
    assert report.verdict == "blurred" and report.tile_share >= 0.7


def test_report_is_serialisable(detector, photo):
    report = detector.analyse(photo)
    data = report.to_dict()
    assert {"blurry", "verdict", "probability", "tile_share", "severity", "sigma", "precision", "fft_ms", "metrics"} <= data.keys()
    assert "BLURRED" in detector.analyse(convolve(photo, gaussian_kernel(3))).summary()


def test_accepts_colour_and_8_bit_images(detector):
    rgb = (load_source("coffee")[:200, :300, None].repeat(3, axis=2) * 255).astype(np.uint8)
    assert detector.analyse(rgb).height == 200


def test_model_records_where_it_came_from():
    model = default_model()
    assert len(model.weights) == 2 * (len(model.band_edges) - 1)  # mean and weakest-direction power per band
    assert model.provenance["samples"] > 100
