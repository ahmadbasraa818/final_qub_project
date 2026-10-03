# The original implementation

This folder keeps the project as it was first submitted, so the rebuild can be
compared with it. Nothing here is used by the current code.

- `NewFFT.cpp`: the C++ transform, using FloatX numbers with an 8-bit exponent
  and a 12-bit mantissa.
- `main.py`: the Python front end. It called `NewFFT` as a subprocess, read
  the spectrum back from a CSV file and computed a blurriness ratio.
- `tests/`: the original tests.
- `Notes.txt`: the supervisor's guidance on the project's scope.
- `README.original.md`, `Command to run.txt`, `For Document.txt` and
  `original-dependencies.md`: the original instructions, notes and pinned
  dependencies. The pins were a `requirements.txt`; they are now a table, so
  that nothing installs, or security-scans, them as if they were live.

## What the rebuild changed, and why

The project's plan asked for Bluestein's FFT for arbitrary lengths, reduced
precision that users can adjust, accurate blur detection checked against
ground truth, real-time speed and a performance analysis. The original
prototype fell short in these ways:

| Original | Problem | Rebuild |
|---|---|---|
| Zero-padded every row and column to a power of two | Not Bluestein. Padding computes the DFT of a different, larger image, so the spectrum changes with the image size. | Bluestein's algorithm: the exact DFT of any size (tested against a direct DFT for every length from 1 to 200 and primes up to 1283). |
| Twiddle factors built by repeated multiplication (`w = w * wn`) in 12-bit arithmetic, with the angle rounded first | Rounding error compounds across each stage. | Twiddles and chirps computed once in binary64 and rounded once. |
| Raw 0 to 255 pixels, no scaling | Would overflow any format with a 5-bit exponent, such as float16. | The mean is removed, and every stage scales by exact powers of two, so float16 is safe on 1080p frames. |
| "High frequency" taken as `x > N/5 && y > N/5` of the unshifted spectrum (C++), and quarter-bands at its edges (Python) | In an unshifted spectrum, the low frequencies sit at the corners, so these regions mix low and high frequencies. | Radial frequency in cycles per pixel, with mean and weakest-direction power in seven bands, and a model fitted on ground truth. |
| Inverse transform divided by 2 at every level and again by n * m | Normalised twice. | One unnormalised inverse with known scaling, tested by round trip. |
| Precision fixed at compile time (`floatx<8, 12>`) | The plan asked for adjustable precision. | Any format chosen at run time (`e8m12`, `bfloat16`, `e5m10`...), plus hardware float64, float32 and float16. |
| Subprocess plus a CSV of the whole spectrum, one console line per recursive call, blocking OpenCV windows | Slow, and could not run without a display. | pybind11 extension; a 720p frame is analysed in about 11 ms on 12 cores. |
| No ground truth and no comparison | The plan's success criteria could not be checked. | A labelled benchmark, cross-validated, compared with four other methods, including this original. |

The rebuild's results, including how the original measure scores on the same
benchmark, are in [`docs/results.md`](../docs/results.md).
