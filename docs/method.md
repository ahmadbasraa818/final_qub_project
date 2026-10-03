# Method

How blurfft computes an image's spectrum in reduced precision, and how it
decides from that spectrum whether, how much and in which direction the image
is blurred.

## 1. Blur in the frequency domain

A blurred image is the sharp image convolved with a point-spread function
(PSF), so its spectrum is the sharp spectrum multiplied by the PSF's transfer
function. Every common blur is a low-pass filter. A Gaussian of radius sigma
multiplies the power at spatial frequency f by exp(-4 pi^2 sigma^2 f^2). A
defocus disk does the same with ripples. Linear motion is a sinc along its
direction that leaves the perpendicular direction untouched.

Fine detail is high-frequency power, so blur shows up as missing power at high
frequency, relative to the low frequencies. Natural images have a typical
power spectrum (roughly 1/f^2), which is what makes "missing" measurable.

## 2. The FFT

### Why not pad to a power of two?

Zero-padding an image to the next power of two (the original prototype's
approach) computes the DFT of a different, larger image: the true spectrum
convolved with a sinc, sampled more finely. The result depends on how much
padding the size needed, and the jump to zero at the padded edge adds false
high-frequency energy. That is the opposite of what a blur detector needs.

### Bluestein's algorithm

For any length n, write jk = (j^2 + k^2 - (k - j)^2) / 2. Then

    X_k = sum_j x_j e^(-2 pi i jk/n) = w_k * sum_j (x_j w_j) * conj(w_(k-j)),   w_k = e^(-i pi k^2 / n)

The sum is a convolution of a_j = x_j w_j with the chirp b_m = conj(w_m). It
is computed exactly with power-of-two FFTs of length M >= 2n - 1: transform
a, multiply by the precomputed transform of b, transform back, and multiply by
w_k. The cost is O(n log n) for every n, primes included.

Implementation details that matter for accuracy:

- `k^2` is reduced modulo 2n in 64-bit integers before forming the angle, since
  w_k has period 2n in k^2. Without this, the angle pi k^2 / n loses precision
  once k^2 passes 2^53.
- The chirp and the kernel's transform are computed once per length in
  binary64 and rounded to the working format once. The original built its
  twiddles by repeated multiplication in 12-bit arithmetic, which compounds
  rounding error.
- Plans are built once per length and shared by every row, column and thread.

### Scaling: no overflow in narrow formats

A DFT can grow values by a factor of n. On a 1080p frame that would overflow
binary16, whose largest value is 65504, by orders of magnitude. Each radix-2
stage can halve its outputs. Halving is exact in binary floating point (unless
it underflows), so it adds no rounding error. The halvings are spread evenly
over the stages, so intermediate values stay bounded by about sqrt(n) times
the input.

In Bluestein's algorithm, the forward and inverse inner transforms each halve
on floor(L / 2) of their L stages, and the kernel is scaled by 2^-c so its
largest value is at most 1. The result is DFT(x) * 2^-(2 floor(L/2) + c - L),
which is close to the unitary scaling 1/sqrt(n). The scale is a known power of
two, undone exactly in binary64 afterwards. The image's mean is removed before
the transform, so the DC term does not dominate the range either. The tests
check that a binary16 transform of a 1080p frame with a strong gradient stays
finite and within 5e-3 of the exact spectrum.

### Real images: half the work

A real image's spectrum is conjugate-symmetric, F(-u, -v) = conj F(u, v), so
only the half-plane v = 0 .. W/2 is computed (numpy.fft.rfft2's layout). Two
real rows go through one complex FFT as z = a + ib and are separated
afterwards:

    A_k = (Z_k + conj Z_(W-k)) / 2,      B_k = (Z_k - conj Z_(W-k)) / 2i

Then a complex FFT runs down each of the W/2 + 1 columns. Rows and columns are
split across threads. In binary64 the result matches `numpy.fft.rfft2` to
about 1e-15.

## 3. Precision: approximate computing

Each format is IEEE 754-style: a sign bit, an e-bit exponent with the usual
bias, an m-bit fraction, subnormals, round-to-nearest-even, and overflow to
infinity.

- **Hardware formats** (`float64`, `float32`, `float16` via `_Float16` where
  the CPU has it, as Apple silicon and recent x86 do) measure real speed and
  memory.
- **Emulated formats** (any `eXmY`, chosen at run time) measure accuracy at
  formats the hardware lacks: bfloat16 (`e8m7`), the original project's FloatX
  (`e8m12`), down to a 2-bit mantissa. Every operation (add, subtract,
  multiply) is computed in binary64 and rounded to the format.

The rounding scales the value by a power of two so the bits to keep form an
integer, rounds once with `nearbyint`, and scales back. That gives the
correctly rounded value. Computing an operation in binary64 and then rounding
is itself correctly rounded for every format with at most 25 mantissa bits:
binary64's 53 bits are at least 2p + 2, which makes double rounding harmless
for +, -, * (Figueroa, 1995). The code is compiled with floating-point
contraction off, so no a*b + c is fused into a single rounding. As a result
the emulated binary32 and binary16 transforms are bit-for-bit identical to
hardware `float` and `_Float16`. The test suite checks this, and checks the
rounding against hardware casts on two million random values.

Reductions over the spectrum (band powers) are accumulated in binary64, as
mixed-precision hardware accumulates in a wider format. The arithmetic under
study is the FFT, which is where nearly all the work is.

## 4. From spectrum to verdict

The image is converted to greyscale (BT.601 luma), its mean is removed, and a
separable Hann window tapers it to zero at the edges. Without the window, the
DFT's implied periodicity puts a jump at the borders, which shows as a bright
cross of false high-frequency energy.

**Features.** For seven radial bands between 0.02 and 0.5 cycles per pixel
(0.5 is the Nyquist frequency):

- the log of the band's mean power spectral density;
- the log of its density in its weakest direction, taken over eight
  22.5-degree sectors. Motion blur removes detail along one direction only,
  which a radial mean would hide.

Bin power is divided by the window's energy (the sum of its squared weights).
That makes it a density per pixel, which does not depend on the image's size:
a 4000-pixel photograph and a 200-pixel crop of the same scene have the same
density. An earlier version used raw power, and every large photograph looked
sharper than it was.

Each density has the noise floor of 8-bit quantisation added first: white
noise of variance (1/255)^2 / 12, which every real photograph already carries.
Adding it keeps the features physical for images that never had it, such as
renders or images blurred in floating point.

**Decision.** A logistic regression on the 14 features, with classes weighted
equally, gives the probability that the image is blurred. In effect it learns
a high-pass filter from data. The variance of the Laplacian, the most common
blur check, is one fixed filter of this kind, weighting power roughly as f^4.

**Severity.** A ridge regression, quadratic in the standardised features,
estimates the blur radius as an equivalent Gaussian sigma. It is fitted on
blurred images. A defocus disk of radius r counts as sigma = r/2, and a motion
streak of length L as sigma = L/sqrt(12) along its direction. The quadratic
terms matter because strong blur drives the upper bands to the noise floor,
where a linear fit flattens out.

**Type and direction.** The power spectrum's structure tensor, whitened by f^2
over 0.05 to 0.35 cycles per pixel, has eigenvalues l1 >= l2. Its anisotropy,
(l1 - l2) / (l1 + l2), separates blur with one dominant direction (linear
motion) from blur with none. Blur with no dominant direction includes defocus,
and camera shake whose path curves through several directions. The dominant
eigenvector, turned 90 degrees, is the motion direction.

**Blur maps and the region rule.** The same features on overlapping square
tiles give a map of where an image is blurred, after Liu, Li and Jia (2008).
Tiles may be any size, since Bluestein handles lengths such as 96 exactly.
Tiles with too little detail to judge, such as sky, night or plain walls, are
left out.

The verdict uses both the whole image and its tiles. An image is blurred when
the whole-image probability passes 0.5, or when at least 70% of its judged
tiles (and at least four of them) are blurred. Between 30% and 70%, it is
partly blurred. The tile rule catches blurred subjects surrounded by
featureless areas, which dilute a whole-image average. The 70% threshold is
the lowest one that adds no false alarms on the benchmark's sharp images, so
the benchmark's accuracy is the same with or without it.

The model is small and readable, as the brief asked: FFT-based, with no neural
network. Its parameters are fitted on the benchmark described in
[results.md](results.md) and shipped in `src/blurfft/model.json`, with a
record of where they came from.

## 5. The benchmark

Twelve public-domain or CC0 photographs from scikit-image's sample data cover
faces, animals, objects, text, textures, astronomy and microscopy. Each gives
four random crops between 128 and 768 pixels a side, mostly sized neither as
powers of two nor as squares, so that size-invariance is tested too. Every
crop is blurred in 13 ways:

- none;
- Gaussian with sigma 0.5 to 4.5;
- defocus disks with radius 2 to 5;
- motion streaks of length 4 to 13 at random angles.

Then each sample passes through a random camera pipeline: sensor noise of
sigma 0 to 0.008, 8-bit quantisation and JPEG at quality 75 to 90 or none. In
the second condition, contrast is also varied from 0.35 to 1. Labels follow
the equivalent radius: sharp up to 0.5 px, blurred from 1.5 px. The band in
between is unlabelled, since people disagree there.

Evaluation is two-fold cross-validation split by photograph, so no test image
shares a photograph with a training image. Single-score methods, such as the
Laplacian, are thresholded where balanced accuracy on the training fold is
highest.

## References

- L. Bluestein, "A linear filtering approach to the computation of discrete Fourier transform", IEEE Trans. Audio and Electroacoustics, 1970.
- J. O. Smith III, "Bluestein's FFT Algorithm", *Mathematics of the Discrete Fourier Transform*, W3K Publishing.
- R. Liu, Z. Li and J. Jia, "Image partial blur detection and classification", IEEE CVPR 2008.
- J. L. Pech-Pacheco et al., "Diatom autofocusing in brightfield microscopy: a comparative study", ICPR 2000 (variance of the Laplacian).
- E. Krotkov, "Focusing", IJCV 1987 (Tenengrad).
- A. Rosebrock, "OpenCV Fast Fourier Transform (FFT) for blur detection", PyImageSearch, 2020.
- S. A. Figueroa, "When is double rounding innocuous?", ACM SIGNUM Newsletter, 1995.
- G. Tagliavini et al., "FloatX: a C++ library for customized floating-point arithmetic", ACM TACO, 2019.
- R. Fisher, S. Perkins, A. Walker and E. Wolfart, "Fourier Transform", HIPR2, University of Edinburgh.
