"""The FFTs against numpy, at every precision, and the emulator against hardware."""

import numpy as np
import pytest

import blurfft
from blurfft import _core
from blurfft.precision import parse_precision

rng = np.random.default_rng(1)


def rel(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)


@pytest.mark.parametrize("n", [1, 2, 3, 5, 7, 8, 12, 16, 31, 64, 97, 100, 127, 128, 255, 256, 509, 1000, 1024, 1283])
def test_fft_matches_numpy_in_float64(n):
    x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    assert rel(blurfft.fft(x), np.fft.fft(x)) < 1e-12
    assert rel(blurfft.fft(x, algorithm="bluestein"), np.fft.fft(x)) < 1e-12


@pytest.mark.parametrize("n", [3, 100, 1000])
def test_inverse_is_the_unnormalised_inverse(n):
    x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    assert rel(blurfft.fft(x, inverse=True), np.fft.ifft(x) * n) < 1e-12


@pytest.mark.parametrize("shape", [(1, 1), (2, 3), (5, 8), (37, 53), (64, 64), (101, 99), (144, 233)])
def test_rfft2_matches_numpy(shape):
    image = rng.random(shape)
    assert rel(blurfft.rfft2(image), np.fft.rfft2(image)) < 1e-12


def test_plans_report_their_algorithm():
    assert _core.plan_info(1024)["bluestein"] is False
    info = _core.plan_info(1000)
    assert info["bluestein"] is True and info["inner_size"] == 2048  # >= 2n - 1
    assert _core.plan_info(1024, "bluestein")["bluestein"] is True


def test_error_tracks_unit_roundoff():
    image = rng.random((120, 175))
    reference = np.fft.rfft2(image)
    errors = {}
    for name in ("float32", "float16", "floatx", "bfloat16", "e8m4"):
        p = parse_precision(name)
        errors[name] = rel(blurfft.rfft2(image, name), reference)
        # A transform of n = 120 x 175 points: error within a small multiple of u * log2(n).
        assert errors[name] < 4 * p.unit_roundoff * np.log2(image.size), (name, errors[name])
    assert errors["float32"] < errors["floatx"] < errors["float16"] < errors["bfloat16"] < errors["e8m4"]


def test_emulated_formats_match_hardware_bit_for_bit():
    image = rng.random((60, 90))
    assert np.array_equal(blurfft.rfft2(image, "float32"), blurfft.rfft2(image, "emulated-float32"))
    if _core.has_native_half():
        assert np.array_equal(blurfft.rfft2(image, "float16"), blurfft.rfft2(image, "emulated-float16"))


def test_quantize_matches_numpy_casts():
    x = rng.standard_normal(200_000) * np.exp(rng.uniform(-30, 30, 200_000))
    assert np.array_equal(blurfft.quantize(x, "e8m23")[np.abs(x) < 3e38], x.astype(np.float32).astype(np.float64)[np.abs(x) < 3e38])
    small = rng.standard_normal(200_000) * np.exp(rng.uniform(-17, 11, 200_000))
    small = small[np.abs(small) < 65504]  # inside binary16's range, where numpy's cast is defined
    assert np.array_equal(blurfft.quantize(small, "e5m10"), small.astype(np.float16).astype(np.float64))


def test_half_precision_survives_a_1080p_frame():
    y, x = np.mgrid[0:1080, 0:1920]
    frame = 0.9 * x / 1920 + 0.05 * np.sin(0.3 * y)
    spectrum = blurfft.rfft2(frame - frame.mean(), "float16")
    assert np.all(np.isfinite(spectrum))
    assert rel(spectrum, np.fft.rfft2(frame - frame.mean())) < 5e-3


def test_original_transform_pads_to_powers_of_two():
    image = rng.random((300, 500))
    magnitude = _core.legacy_fft2_magnitude(image, "float64", 11, 52)
    assert magnitude.shape == (512, 512)
    padded = np.zeros((512, 512))
    padded[:300, :500] = image
    assert rel(magnitude, np.abs(np.fft.fft2(padded))) < 1e-10
