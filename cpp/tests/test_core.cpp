// Unit tests for the C++ core. Plain asserts, no framework: build and run with
//   cmake -S . -B build && cmake --build build && ctest --test-dir build
#include <blurfft/analysis.hpp>
#include <blurfft/fft.hpp>
#include <blurfft/fft2d.hpp>
#include <blurfft/metrics.hpp>
#include <blurfft/precision.hpp>

#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

using namespace blurfft;
using cd = std::complex<double>;

static int failures = 0;
static int checks = 0;

#define CHECK(condition, ...)                                   \
  do {                                                          \
    ++checks;                                                   \
    if (!(condition)) {                                         \
      ++failures;                                               \
      std::printf("FAIL %s:%d  %s  ", __FILE__, __LINE__, #condition); \
      std::printf(__VA_ARGS__);                                 \
      std::printf("\n");                                        \
    }                                                           \
  } while (0)

static std::vector<cd> naive_dft(const std::vector<cd>& x, int sign = -1) {
  const std::size_t n = x.size();
  std::vector<cd> out(n);
  for (std::size_t k = 0; k < n; ++k) {
    cd sum = 0;
    for (std::size_t j = 0; j < n; ++j) {
      const double angle = sign * 2.0 * kPi * static_cast<double>((j * k) % n) / static_cast<double>(n);
      sum += x[j] * cd(std::cos(angle), std::sin(angle));
    }
    out[k] = sum;
  }
  return out;
}

static double relative_error(const std::vector<cd>& got, const std::vector<cd>& want) {
  double num = 0, den = 0;
  for (std::size_t i = 0; i < want.size(); ++i) {
    num += std::norm(got[i] - want[i]);
    den += std::norm(want[i]);
  }
  return den == 0 ? std::sqrt(num) : std::sqrt(num / den);
}

static std::vector<cd> random_signal(std::size_t n, std::mt19937_64& rng) {
  std::normal_distribution<double> g(0.0, 1.0);
  std::vector<cd> x(n);
  for (auto& v : x) v = {g(rng), g(rng)};
  return x;
}

template <class A>
static std::vector<cd> run_fft(const std::vector<cd>& x, const A& arith, Algorithm algorithm, bool inverse = false) {
  using T = typename A::value_type;
  const ComplexOps<A> c{arith};
  Fft<A> plan(x.size(), arith, algorithm);
  std::vector<Complex<T>> buf(x.size());
  for (std::size_t i = 0; i < x.size(); ++i) buf[i] = c.from(x[i]);
  std::vector<Complex<T>> work;
  if (inverse) {
    plan.inverse(buf.data(), buf.data(), work);
  } else {
    plan.forward(buf.data(), buf.data(), work);
  }
  std::vector<cd> out(x.size());
  const double unscale = std::ldexp(1.0, plan.shift());
  for (std::size_t i = 0; i < x.size(); ++i) out[i] = unscale * c.to(buf[i]);
  return out;
}

static void test_quantize_matches_hardware() {
  std::mt19937_64 rng(1);
  std::uniform_real_distribution<double> mantissa(1.0, 2.0);
  std::uniform_int_distribution<int> exponent_f32(-149, 127), exponent_f16(-24, 15);
  std::bernoulli_distribution sign(0.5);
  int mismatches32 = 0, mismatches16 = 0;
  for (int i = 0; i < 2000000; ++i) {
    const double x32 = (sign(rng) ? -1 : 1) * std::ldexp(mantissa(rng), exponent_f32(rng));
    if (std::fabs(x32) < 3.4e38) {
      const double want = static_cast<double>(static_cast<float>(x32));
      if (quantize(x32, Format{8, 23}) != want) ++mismatches32;
    }
#if BLURFFT_HAS_FLOAT16
    const double x16 = (sign(rng) ? -1 : 1) * std::ldexp(mantissa(rng), exponent_f16(rng));
    if (std::fabs(x16) < 65504) {
      const double want = static_cast<double>(static_cast<half_t>(x16));
      if (quantize(x16, Format{5, 10}) != want) ++mismatches16;
    }
#endif
  }
  CHECK(mismatches32 == 0, "%d binary32 mismatches", mismatches32);
  CHECK(mismatches16 == 0, "%d binary16 mismatches", mismatches16);
  // Edge cases: ties to even, overflow, subnormals, signed zero.
  CHECK(quantize(1.0 + std::ldexp(1.0, -11), Format{5, 10}) == 1.0, "tie rounds to even (down)");
  CHECK(quantize(1.0 + 3 * std::ldexp(1.0, -11), Format{5, 10}) == 1.0 + std::ldexp(1.0, -9), "tie rounds to even (up)");
  CHECK(std::isinf(quantize(65520.0, Format{5, 10})), "rounds past the largest half to infinity");
  CHECK(quantize(65519.0, Format{5, 10}) == 65504.0, "just below the overflow threshold");
  CHECK(quantize(std::ldexp(1.0, -24), Format{5, 10}) == std::ldexp(1.0, -24), "smallest half subnormal");
  CHECK(quantize(std::ldexp(1.0, -26), Format{5, 10}) == 0.0, "below half the smallest subnormal");
  CHECK(std::signbit(quantize(-std::ldexp(1.0, -30), Format{5, 10})), "keeps the sign of an underflow");
  CHECK(quantize(0.1, Format{11, 52}) == 0.1, "binary64 is untouched");
}

static void test_fft_matches_dft() {
  std::mt19937_64 rng(2);
  double worst_auto = 0, worst_bluestein = 0;
  std::vector<std::size_t> lengths;
  for (std::size_t n = 1; n <= 200; ++n) lengths.push_back(n);
  for (std::size_t n : {251u, 256u, 509u, 512u, 1021u, 1024u, 1283u}) lengths.push_back(n);
  for (std::size_t n : lengths) {
    const std::vector<cd> x = random_signal(n, rng);
    const std::vector<cd> want = naive_dft(x);
    worst_auto = std::max(worst_auto, relative_error(run_fft(x, Native<double>{}, Algorithm::Auto), want));
    worst_bluestein = std::max(worst_bluestein, relative_error(run_fft(x, Native<double>{}, Algorithm::Bluestein), want));
    const std::vector<cd> back = run_fft(want, Native<double>{}, Algorithm::Auto, true);
    std::vector<cd> scaled(n);
    for (std::size_t i = 0; i < n; ++i) scaled[i] = back[i] / static_cast<double>(n);
    CHECK(relative_error(scaled, x) < 1e-11, "inverse round trip at n = %zu: %.3g", n, relative_error(scaled, x));
  }
  CHECK(worst_auto < 1e-12, "radix-2/Bluestein vs direct DFT: worst relative error %.3g", worst_auto);
  CHECK(worst_bluestein < 1e-12, "forced Bluestein vs direct DFT: worst relative error %.3g", worst_bluestein);
}

static void test_reduced_precision_error_scales() {
  std::mt19937_64 rng(3);
  const std::vector<cd> x = random_signal(1000, rng);
  const std::vector<cd> want = naive_dft(x);
  const double e32 = relative_error(run_fft(x, Native<float>{}, Algorithm::Auto), want);
  const double e12 = relative_error(run_fft(x, Emulated{Format{8, 12}}, Algorithm::Auto), want);
  const double e7 = relative_error(run_fft(x, Emulated{Format{8, 7}}, Algorithm::Auto), want);
  // Error tracks unit roundoff times a slowly growing factor of log(n).
  CHECK(e32 < 1e-5 && e32 > 1e-9, "float32 error %.3g", e32);
  CHECK(e12 < 5e-3 && e12 > e32, "e8m12 error %.3g", e12);
  CHECK(e7 < 1e-1 && e7 > e12, "bfloat16 error %.3g", e7);
}

static void test_emulation_is_bit_exact_with_hardware() {
  std::mt19937_64 rng(4);
  for (std::size_t n : {100u, 128u, 997u}) {
    const std::vector<cd> x = random_signal(n, rng);
    const std::vector<cd> native32 = run_fft(x, Native<float>{}, Algorithm::Auto);
    const std::vector<cd> emulated32 = run_fft(x, Emulated{Format{8, 23}}, Algorithm::Auto);
    CHECK(native32 == emulated32, "emulated e8m23 equals hardware float at n = %zu", n);
#if BLURFFT_HAS_FLOAT16
    std::vector<cd> small = x;
    for (auto& v : small) v *= 0.05;  // keep the inputs inside binary16's range
    const std::vector<cd> native16 = run_fft(small, Native<half_t>{}, Algorithm::Auto);
    const std::vector<cd> emulated16 = run_fft(small, Emulated{Format{5, 10}}, Algorithm::Auto);
    CHECK(native16 == emulated16, "emulated e5m10 equals hardware _Float16 at n = %zu", n);
#endif
  }
}

static std::vector<cd> naive_dft2d(const std::vector<double>& img, std::size_t h, std::size_t w) {
  std::vector<cd> out(h * w);
  for (std::size_t u = 0; u < h; ++u) {
    for (std::size_t v = 0; v < w; ++v) {
      cd sum = 0;
      for (std::size_t y = 0; y < h; ++y) {
        for (std::size_t x = 0; x < w; ++x) {
          const double angle = -2.0 * kPi * (static_cast<double>((u * y) % h) / h + static_cast<double>((v * x) % w) / w);
          sum += img[y * w + x] * cd(std::cos(angle), std::sin(angle));
        }
      }
      out[u * w + v] = sum;
    }
  }
  return out;
}

static void test_real_fft2d() {
  std::mt19937_64 rng(5);
  std::uniform_real_distribution<double> uniform(-1.0, 1.0);
  const std::size_t sizes[][2] = {{1, 1}, {2, 2}, {3, 5}, {7, 8}, {16, 9}, {31, 17}, {12, 33}, {64, 64}, {29, 47}};
  for (const auto& sz : sizes) {
    const std::size_t h = sz[0], w = sz[1];
    std::vector<double> img(h * w);
    for (auto& v : img) v = uniform(rng);
    const std::vector<cd> full = naive_dft2d(img, h, w);
    const HalfSpectrum s = real_fft2d(img.data(), h, w, Native<double>{}, Algorithm::Auto, 3);
    std::vector<cd> got, want;
    for (std::size_t u = 0; u < h; ++u) {
      for (std::size_t v = 0; v < s.columns(); ++v) {
        got.push_back(s.at(u, v));
        want.push_back(full[u * w + v]);
      }
    }
    CHECK(relative_error(got, want) < 1e-11, "2-D real FFT %zux%zu: %.3g", h, w, relative_error(got, want));
  }
}

static void test_half_precision_does_not_overflow() {
#if BLURFFT_HAS_FLOAT16
  // A 1080p frame with a strong gradient: unscaled, its low frequencies would
  // pass 65504 by orders of magnitude.
  const std::size_t h = 1080, w = 1920;
  std::vector<double> img(h * w);
  for (std::size_t y = 0; y < h; ++y) {
    for (std::size_t x = 0; x < w; ++x) img[y * w + x] = 0.9 * static_cast<double>(x) / w - 0.45 + 0.05 * std::sin(0.3 * y);
  }
  const HalfSpectrum s = real_fft2d(img.data(), h, w, Native<half_t>{}, Algorithm::Auto, 0);
  bool finite = true;
  for (const auto& z : s.values) finite = finite && std::isfinite(z.real()) && std::isfinite(z.imag());
  CHECK(finite, "binary16 spectrum of a 1080p frame stays finite");
#endif
}

// A random texture with a 1/f spectrum, roughly like a natural image.
static std::vector<double> texture(std::size_t h, std::size_t w, unsigned seed) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> g(0.0, 1.0);
  std::vector<double> img(h * w, 0.0);
  for (int octave = 0; octave < 6; ++octave) {
    const std::size_t step = std::size_t{1} << (5 - octave);
    const double amplitude = std::ldexp(1.0, -octave);
    const std::size_t gh = h / step + 2, gw = w / step + 2;
    std::vector<double> grid(gh * gw);
    for (auto& v : grid) v = g(rng);
    for (std::size_t y = 0; y < h; ++y) {
      for (std::size_t x = 0; x < w; ++x) {
        const double fy = static_cast<double>(y) / step, fx = static_cast<double>(x) / step;
        const std::size_t y0 = static_cast<std::size_t>(fy), x0 = static_cast<std::size_t>(fx);
        const double ty = fy - y0, tx = fx - x0;
        const double top = grid[y0 * gw + x0] * (1 - tx) + grid[y0 * gw + x0 + 1] * tx;
        const double bottom = grid[(y0 + 1) * gw + x0] * (1 - tx) + grid[(y0 + 1) * gw + x0 + 1] * tx;
        img[y * w + x] += amplitude * (top * (1 - ty) + bottom * ty);
      }
    }
  }
  std::uniform_real_distribution<double> noise(-0.02, 0.02);
  for (auto& v : img) v = 0.5 + 0.1 * v + noise(rng);
  return img;
}

static std::vector<double> convolve(const std::vector<double>& img, std::size_t h, std::size_t w, const std::vector<double>& k, int kh, int kw) {
  std::vector<double> out(h * w, 0.0);
  for (std::size_t y = 0; y < h; ++y) {
    for (std::size_t x = 0; x < w; ++x) {
      double sum = 0;
      for (int dy = 0; dy < kh; ++dy) {
        for (int dx = 0; dx < kw; ++dx) {
          const long sy = std::min<long>(std::max<long>(static_cast<long>(y) + dy - kh / 2, 0), static_cast<long>(h) - 1);
          const long sx = std::min<long>(std::max<long>(static_cast<long>(x) + dx - kw / 2, 0), static_cast<long>(w) - 1);
          sum += k[dy * kw + dx] * img[sy * w + sx];
        }
      }
      out[y * w + x] = sum;
    }
  }
  return out;
}

static std::vector<double> gaussian_kernel(double sigma, int& size) {
  const int radius = static_cast<int>(std::ceil(3 * sigma));
  size = 2 * radius + 1;
  std::vector<double> k(size * size);
  double total = 0;
  for (int y = 0; y < size; ++y) {
    for (int x = 0; x < size; ++x) {
      total += k[y * size + x] = std::exp(-((x - radius) * (x - radius) + (y - radius) * (y - radius)) / (2 * sigma * sigma));
    }
  }
  for (auto& v : k) v /= total;
  return k;
}

static void test_metrics_follow_blur() {
  const std::size_t h = 160, w = 210;
  const std::vector<double> sharp = texture(h, w, 7);
  AnalysisOptions opt;
  const SpectralMetrics base = analyse(sharp.data(), h, w, opt).metrics;
  double previous_ratio = base.high_frequency_ratio, previous_slope = base.slope;
  for (double sigma : {0.8, 1.5, 3.0}) {
    int size = 0;
    const std::vector<double> kernel = gaussian_kernel(sigma, size);
    const std::vector<double> blurred = convolve(sharp, h, w, kernel, size, size);
    const SpectralMetrics m = analyse(blurred.data(), h, w, opt).metrics;
    CHECK(m.high_frequency_ratio < previous_ratio, "sigma %.1f lowers the high-frequency share (%.4g >= %.4g)", sigma, m.high_frequency_ratio, previous_ratio);
    CHECK(m.slope > previous_slope, "sigma %.1f steepens the slope (%.3g <= %.3g)", sigma, m.slope, previous_slope);
    previous_ratio = m.high_frequency_ratio;
    previous_slope = m.slope;
  }
}

static void test_motion_blur_direction() {
  const std::size_t n = 192;
  const std::vector<double> sharp = texture(n, n, 9);
  for (double angle : {0.0, 45.0, 90.0, 135.0}) {
    // A 15-pixel line kernel at the given angle (image coordinates, y down).
    const int size = 15;
    std::vector<double> k(size * size, 0.0);
    const double rad = angle * kPi / 180.0;
    for (int t = -70; t <= 70; ++t) {
      const double s = t / 10.0;
      const int x = static_cast<int>(std::lround(7 + s * std::cos(rad)));
      const int y = static_cast<int>(std::lround(7 + s * std::sin(rad)));
      if (x >= 0 && x < size && y >= 0 && y < size) k[y * size + x] = 1.0;
    }
    double total = 0;
    for (double v : k) total += v;
    for (auto& v : k) v /= total;
    const std::vector<double> blurred = convolve(sharp, n, n, k, size, size);
    const SpectralMetrics m = analyse(blurred.data(), n, n, AnalysisOptions{}).metrics;
    double diff = std::fabs(m.orientation_deg - angle);
    diff = std::min(diff, 180.0 - diff);
    CHECK(diff < 12.0, "motion blur at %.0f deg reads as %.1f deg", angle, m.orientation_deg);
    CHECK(m.anisotropy > 0.3, "motion blur at %.0f deg is anisotropic (%.3g)", angle, m.anisotropy);
  }
}

static void test_legacy_matches_its_own_definition() {
  std::mt19937_64 rng(6);
  // On a power-of-two length in binary64 the legacy transform is a plain DFT.
  const std::vector<cd> x = random_signal(256, rng);
  std::vector<Complex<double>> buf(x.size());
  for (std::size_t i = 0; i < x.size(); ++i) buf[i] = {x[i].real(), x[i].imag()};
  const std::vector<Complex<double>> out = legacy_padded_fft(buf, Native<double>{});
  std::vector<cd> got(out.size());
  for (std::size_t i = 0; i < out.size(); ++i) got[i] = {out[i].re, out[i].im};
  CHECK(relative_error(got, naive_dft(x)) < 1e-10, "legacy radix-2 in binary64 is a DFT");
  // On other lengths it pads, so it is not the DFT of the input.
  CHECK(next_power_of_two(300) == 512, "300 pads to 512");
}

int main() {
  test_quantize_matches_hardware();
  test_fft_matches_dft();
  test_reduced_precision_error_scales();
  test_emulation_is_bit_exact_with_hardware();
  test_real_fft2d();
  test_half_precision_does_not_overflow();
  test_metrics_follow_blur();
  test_motion_blur_direction();
  test_legacy_matches_its_own_definition();
  std::printf("%d checks, %d failures (hardware float16: %s)\n", checks, failures, has_native_half() ? "yes" : "no");
  return failures == 0 ? 0 : 1;
}
