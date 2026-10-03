"""blurfft: image blur detection with approximate FFTs.

Bluestein's FFT computes the exact DFT of an image of any size, in any
floating-point precision from float64 down to formats a few bits wide, and
the image's spectrum shows how much fine detail survives.
"""

from . import _core
from .detector import BlurDetector, BlurMapResult, BlurReport, measure
from .imageio import load_gray
from .model import BlurModel, default_model
from .precision import Precision, parse_precision

__version__ = "2.0.0"

__all__ = [
    "BlurDetector", "BlurMapResult", "BlurReport", "BlurModel", "Precision", "default_model", "fft", "load_gray",
    "measure", "parse_precision", "quantize", "rfft2",
]


def fft(x, precision="float64", inverse=False, algorithm="auto"):
    """DFT of a 1-D array computed in ``precision`` (returned in float64).

    ``inverse`` gives the unnormalised inverse, sum x_j e^(+2 pi i jk/n).
    """
    import numpy as np

    p = parse_precision(precision)
    return _core.fft(np.asarray(x, dtype=np.complex128), *p.core_args(), inverse=inverse, algorithm=algorithm)


def rfft2(image, precision="float64", algorithm="auto", threads=0):
    """Half-plane 2-D DFT of a real image, like numpy.fft.rfft2, computed in ``precision``."""
    import numpy as np

    p = parse_precision(precision)
    return _core.rfft2(np.asarray(image, dtype=np.float64), *p.core_args(), algorithm=algorithm, threads=threads)


def quantize(x, precision):
    """Rounds values to the nearest value of an emulated format, such as 'e8m12'."""
    import numpy as np

    p = parse_precision(precision)
    return _core.quantize(np.asarray(x, dtype=np.float64), p.exponent_bits, p.mantissa_bits)
