"""Precision formats: parse a name, describe it, pass it to the C++ core."""

from __future__ import annotations

import re
from dataclasses import dataclass

from . import _core

__all__ = ["Precision", "parse_precision", "STANDARD_FORMATS", "SWEEP_FORMATS"]


@dataclass(frozen=True)
class Precision:
    """Arithmetic for the FFT.

    ``kind`` is ``float64``, ``float32`` or ``float16`` for hardware arithmetic,
    or ``emulated`` for any format of ``exponent_bits`` and ``mantissa_bits``,
    rounded after every operation in software.
    """

    name: str
    kind: str
    exponent_bits: int
    mantissa_bits: int

    @property
    def bits(self) -> int:
        """Bits per real value: sign, exponent and mantissa."""
        return 1 + self.exponent_bits + self.mantissa_bits

    @property
    def native(self) -> bool:
        return self.kind != "emulated"

    @property
    def unit_roundoff(self) -> float:
        return 2.0 ** -(self.mantissa_bits + 1)

    def core_args(self) -> tuple[str, int, int]:
        return (self.kind, self.exponent_bits, self.mantissa_bits)

    def __str__(self) -> str:
        return self.name


_NATIVE = {
    "float64": ("float64", 11, 52),
    "float32": ("float32", 8, 23),
    "float16": ("float16", 5, 10),
}
_ALIASES = {
    "double": "float64", "f64": "float64", "fp64": "float64", "binary64": "float64",
    "single": "float32", "f32": "float32", "fp32": "float32", "binary32": "float32", "float": "float32",
    "half": "float16", "f16": "float16", "fp16": "float16", "binary16": "float16",
    "bfloat16": "e8m7", "bf16": "e8m7",
    "floatx": "e8m12",  # the original project's FloatX<8, 12>
}


def parse_precision(spec: "str | Precision") -> Precision:
    """Turns a name into a Precision.

    Accepted: ``float64``/``double``, ``float32``/``single``, ``float16``/``half``
    (hardware where available, otherwise emulated), ``bfloat16``, ``floatx``
    (the original project's 8-bit exponent, 12-bit mantissa) and ``eXmY`` for
    any emulated format, such as ``e5m10`` or ``e8m4``. Prefix a hardware name
    with ``emulated-`` to emulate it instead (useful for checking the emulator).
    """
    if isinstance(spec, Precision):
        return spec
    text = str(spec).strip().lower()
    emulate = text.startswith("emulated-")
    if emulate:
        text = text[len("emulated-"):]
    text = _ALIASES.get(text, text)
    if text in _NATIVE:
        kind, e, m = _NATIVE[text]
        if emulate or (kind == "float16" and not _core.has_native_half()):
            return Precision(f"e{e}m{m}", "emulated", e, m)
        return Precision(text, kind, e, m)
    match = re.fullmatch(r"e(\d+)m(\d+)", text)
    if match:
        e, m = int(match.group(1)), int(match.group(2))
        if not 2 <= e <= 11 or not 1 <= m <= 52:
            raise ValueError("emulated formats need 2-11 exponent bits and 1-52 mantissa bits")
        return Precision(f"e{e}m{m}", "emulated", e, m)
    raise ValueError(
        f"unknown precision {spec!r}: use float64, float32, float16, bfloat16, floatx or eXmY (for example e5m10)"
    )


#: Formats the experiments compare, widest first.
STANDARD_FORMATS = ("float64", "float32", "float16", "bfloat16", "floatx")

#: Formats for the accuracy sweep: an 8-bit exponent (wide range, like bfloat16
#: and FloatX) with the mantissa narrowed one bit at a time.
SWEEP_FORMATS = tuple(f"e8m{m}" for m in (23, 18, 14, 12, 10, 8, 7, 6, 5, 4, 3, 2))
