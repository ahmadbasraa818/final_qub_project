import pytest

from blurfft import _core
from blurfft.precision import parse_precision


@pytest.mark.parametrize(
    "spec, name, bits",
    [
        ("float64", "float64", 64), ("double", "float64", 64), ("float32", "float32", 32), ("single", "float32", 32),
        ("bfloat16", "e8m7", 16), ("floatx", "e8m12", 21), ("e5m10", "e5m10", 16), ("E8M4", "e8m4", 13),
        ("emulated-float32", "e8m23", 32),
    ],
)
def test_names(spec, name, bits):
    p = parse_precision(spec)
    assert p.name == name and p.bits == bits


def test_half_is_hardware_where_available():
    p = parse_precision("half")
    assert p.native == bool(_core.has_native_half())
    assert (p.exponent_bits, p.mantissa_bits) == (5, 10)


@pytest.mark.parametrize("spec", ["float128", "e1m3", "e12m3", "e8m0", "e8m53", "fast", ""])
def test_rejects_unknown_or_out_of_range(spec):
    with pytest.raises(ValueError):
        parse_precision(spec)


def test_unit_roundoff():
    assert parse_precision("float32").unit_roundoff == 2.0**-24
    assert parse_precision("e8m12").unit_roundoff == 2.0**-13
