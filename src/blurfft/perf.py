"""Performance analysis: speed, memory, numerical accuracy and energy.

The plan asked for a tool that measures the system's efficiency, its resource
use and its speed. This module measures, for each frame size and precision:

* time per 2-D FFT and per full analysis (median of several runs), with one
  thread and with every core, next to numpy's pocketfft for reference;
* the cost of an exact transform: Bluestein at the true size against
  zero-padding to powers of two (the original project's approach);
* the memory the spectrum needs in each format;
* the spectrum's error against float64, for this project's transform and the
  original one at the same format;
* energy, where the operating system exposes counters (Linux RAPL). macOS
  needs root for its power metrics, so there it is reported as unavailable
  rather than estimated.
"""

from __future__ import annotations

import os
import platform
import statistics
import time
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

from . import _core
from .detector import measure
from .precision import parse_precision

__all__ = ["FRAME_SIZES", "time_call", "spectrum_bytes", "speed_table", "padding_table", "accuracy_table",
           "energy_joules", "machine", "run_performance"]

FRAME_SIZES = {
    "480p (640 x 480)": (480, 640),
    "720p (1280 x 720)": (720, 1280),
    "1080p (1920 x 1080)": (1080, 1920),
    "1024 x 1024": (1024, 1024),
    "1009 x 997 (primes)": (997, 1009),
}


def machine() -> dict:
    info = {"system": platform.system(), "machine": platform.machine(), "python": platform.python_version(),
            "cores": os.cpu_count(), "native_float16": bool(_core.has_native_half())}
    try:
        if info["system"] == "Darwin":
            import subprocess

            info["cpu"] = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
        else:
            info["cpu"] = platform.processor() or "unknown"
    except OSError:
        info["cpu"] = "unknown"
    return info


def time_call(fn: Callable[[], object], repeats: int = 7, warmup: int = 1) -> float:
    """Median wall time of fn() in seconds."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def spectrum_bytes(height: int, width: int, precision: str) -> int:
    """Bytes for the half-plane spectrum in a format, packed at its bit width.

    Emulated formats are computed in float64 here; the figure is what hardware
    with that format would store.
    """
    p = parse_precision(precision)
    return int(height * (width // 2 + 1) * 2 * p.bits / 8)


def test_frame(height: int, width: int, seed: int = 0) -> np.ndarray:
    """A deterministic natural-looking frame: smooth structure plus fine texture."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width] / max(height, width)
    frame = 0.5 + 0.2 * np.sin(9 * x + 4 * y) + 0.1 * np.cos(31 * x * y)
    frame += 0.05 * rng.standard_normal((height, width))
    return np.clip(frame, 0, 1)


def speed_table(precisions: Sequence[str] = ("float64", "float32", "float16"), sizes: Optional[dict] = None,
                threads: Sequence[int] = (1, 0), repeats: int = 7) -> list[dict]:
    rows = []
    for label, (h, w) in (sizes or FRAME_SIZES).items():
        frame = test_frame(h, w)
        prepared = np.ascontiguousarray(frame - frame.mean())
        reference = time_call(lambda prepared=prepared: np.fft.rfft2(prepared), repeats)
        rows.append({"size": label, "height": h, "width": w, "precision": "numpy pocketfft (float64)", "threads": 1,
                     "fft_ms": reference * 1000, "analysis_ms": None, "fps": 1 / reference})
        for name in precisions:
            p = parse_precision(name)
            for t in threads:
                fft = time_call(lambda prepared=prepared, p=p, t=t: _core.rfft2(prepared, *p.core_args(), threads=t), repeats)
                total = time_call(lambda frame=frame, p=p, t=t: measure(frame, p, threads=t), repeats)
                rows.append({"size": label, "height": h, "width": w, "precision": p.name,
                             "threads": t or os.cpu_count(), "fft_ms": fft * 1000, "analysis_ms": total * 1000,
                             "fps": 1 / total, "spectrum_mb": spectrum_bytes(h, w, p.name) / 1e6})
    return rows


def padding_table(sizes: Optional[dict] = None, threads: int = 0, repeats: int = 7) -> list[dict]:
    """Bluestein at the true size against zero-padding to the next powers of two."""
    rows = []
    for label, (h, w) in (sizes or FRAME_SIZES).items():
        frame = test_frame(h, w)
        ph, pw = 1 << (h - 1).bit_length(), 1 << (w - 1).bit_length()
        padded = np.zeros((ph, pw))
        padded[:h, :w] = frame
        exact = time_call(lambda frame=frame: _core.rfft2(frame, "float64", threads=threads), repeats)
        pad = time_call(lambda padded=padded: _core.rfft2(padded, "float64", threads=threads), repeats)
        rows.append({"size": label, "exact_ms": exact * 1000, "padded_ms": pad * 1000, "padded_shape": f"{pw} x {ph}",
                     "extra_pixels": ph * pw / (h * w) - 1, "bluestein_rows": bool(_core.plan_info(w)["bluestein"]),
                     "bluestein_columns": bool(_core.plan_info(h)["bluestein"])})
    return rows


def accuracy_table(images: Sequence[np.ndarray], precisions: Sequence[str]) -> list[dict]:
    """Median relative error of the spectrum against float64, new and original transforms."""
    rows = []
    references = [_core.rfft2(np.ascontiguousarray(im - im.mean()), "float64") for im in images]
    legacy_refs = [_core.legacy_fft2_magnitude(im * 255.0, "float64", 11, 52) for im in images]
    for name in precisions:
        p = parse_precision(name)
        errors, legacy_errors = [], []
        for im, ref, legacy_ref in zip(images, references, legacy_refs):
            got = _core.rfft2(np.ascontiguousarray(im - im.mean()), *p.core_args())
            errors.append(float(np.linalg.norm(got - ref) / np.linalg.norm(ref)))
            if p.exponent_bits >= 8:  # the original fed raw 0-255 pixels, which overflow narrower exponents
                legacy = _core.legacy_fft2_magnitude(im * 255.0, *p.core_args())
                legacy_errors.append(float(np.linalg.norm(legacy - legacy_ref) / np.linalg.norm(legacy_ref)))
        rows.append({"precision": p.name, "mantissa_bits": p.mantissa_bits, "exponent_bits": p.exponent_bits,
                     "unit_roundoff": p.unit_roundoff, "error": float(np.median(errors)),
                     "original_error": float(np.median(legacy_errors)) if legacy_errors else None})
    return rows


def _rapl_files() -> list[Path]:
    base = Path("/sys/class/powercap")
    if not base.exists():
        return []
    return [p / "energy_uj" for p in sorted(base.glob("intel-rapl:*")) if (p / "energy_uj").exists() and ":" not in p.name[11:]]


def energy_joules(fn: Callable[[], object], repeats: int = 20) -> Optional[float]:
    """Package energy per call from Linux RAPL counters, or None where unavailable."""
    files = _rapl_files()
    try:
        before = [int(f.read_text()) for f in files]
    except (OSError, ValueError):
        return None
    if not files:
        return None
    for _ in range(repeats):
        fn()
    after = [int(f.read_text()) for f in files]
    used = sum(max(0, b - a) for a, b in zip(before, after))
    return used / 1e6 / repeats


def run_performance(images: Sequence[np.ndarray], precisions_speed: Sequence[str], precisions_accuracy: Sequence[str],
                    repeats: int = 7, progress: Optional[Callable[[str], None]] = None) -> dict:
    say = progress or (lambda message: None)
    out = {"machine": machine()}
    say("speed")
    out["speed"] = speed_table(precisions_speed, repeats=repeats)
    say("padding")
    out["padding"] = padding_table(repeats=repeats)
    say("accuracy")
    out["accuracy"] = accuracy_table(images, precisions_accuracy)
    frame = test_frame(720, 1280)
    out["energy"] = {p: energy_joules(lambda p=p: measure(frame, p)) for p in precisions_speed}
    out["emulation"] = {
        "e8m12_720p_ms": time_call(lambda: measure(frame, "e8m12"), repeats=3) * 1000,
        "note": "emulated formats round every operation in software: a tool for accuracy studies, not for speed",
    }
    return out
