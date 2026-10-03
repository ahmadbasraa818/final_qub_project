// One-dimensional FFTs of any length, in any arithmetic.
//
// Lengths that are powers of two use an iterative radix-2 Cooley-Tukey
// transform. Every other length uses Bluestein's algorithm, which rewrites a
// length-n DFT as a convolution with a chirp and computes that convolution
// with power-of-two FFTs of length m >= 2n - 1. The result is the exact DFT of
// the n samples: nothing is padded, so no length changes the spectrum.
//
// Scaling. A DFT of length n can grow values by a factor of n, which would
// overflow narrow formats such as binary16 on a 1080p image. Each transform
// therefore divides by powers of two along the way (exact operations that add
// no rounding error) and returns DFT(x) * 2^-shift(), with shift() close to
// log2(n) / 2. Callers undo the scaling in binary64 when they need it.
#pragma once

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

#include "precision.hpp"

namespace blurfft {

template <class T>
struct Complex {
  T re{};
  T im{};
};

constexpr double kPi = 3.141592653589793238462643383279502884;

inline bool is_power_of_two(std::size_t n) { return n != 0 && (n & (n - 1)) == 0; }

inline std::size_t next_power_of_two(std::size_t n) {
  std::size_t p = 1;
  while (p < n) p <<= 1;
  return p;
}

inline int log2_of_power_of_two(std::size_t n) {
  int l = 0;
  while ((std::size_t{1} << l) < n) ++l;
  return l;
}

/// Complex operations in arithmetic A: each real operation is rounded separately.
template <class A>
struct ComplexOps {
  using T = typename A::value_type;
  const A& a;
  Complex<T> add(Complex<T> x, Complex<T> y) const { return {a.add(x.re, y.re), a.add(x.im, y.im)}; }
  Complex<T> sub(Complex<T> x, Complex<T> y) const { return {a.sub(x.re, y.re), a.sub(x.im, y.im)}; }
  Complex<T> mul(Complex<T> x, Complex<T> y) const {
    return {a.sub(a.mul(x.re, y.re), a.mul(x.im, y.im)), a.add(a.mul(x.re, y.im), a.mul(x.im, y.re))};
  }
  Complex<T> half(Complex<T> x) const { return {a.half(x.re), a.half(x.im)}; }
  Complex<T> conj(Complex<T> x) const { return {x.re, -x.im}; }
  Complex<T> from(std::complex<double> z) const { return {a.from(z.real()), a.from(z.imag())}; }
  std::complex<double> to(Complex<T> z) const { return {a.to(z.re), a.to(z.im)}; }
};

/// Which algorithm a length uses. Auto picks radix-2 for powers of two and
/// Bluestein for everything else; Bluestein forces it for every length, which
/// the experiments use to measure its cost and accuracy on its own.
enum class Algorithm { Auto, Bluestein };

/// In-place iterative radix-2 FFT of a power-of-two length.
template <class A>
class Radix2Plan {
 public:
  using T = typename A::value_type;

  Radix2Plan(std::size_t n, const A& arith) : n_(n), arith_(arith) {
    if (!is_power_of_two(n)) throw std::invalid_argument("radix-2 needs a power-of-two length");
    log2n_ = log2_of_power_of_two(n);
    reversed_.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
      std::size_t r = 0;
      for (int b = 0; b < log2n_; ++b) r |= ((i >> b) & 1u) << (log2n_ - 1 - b);
      reversed_[i] = static_cast<std::uint32_t>(r);
    }
    // Twiddles are constants: computed once in binary64 and rounded to the format,
    // never built up by repeated multiplication, which compounds rounding error.
    twiddles_.resize(n / 2);
    for (std::size_t k = 0; k < n / 2; ++k) {
      const double angle = 2.0 * kPi * static_cast<double>(k) / static_cast<double>(n);
      twiddles_[k] = {arith_.from(std::cos(angle)), arith_.from(-std::sin(angle))};
    }
  }

  std::size_t size() const { return n_; }
  int stages() const { return log2n_; }

  /// Transforms x in place: sum_j x_j e^(-2 pi i jk / n) for forward, e^(+...) for
  /// inverse, multiplied by 2^-halvings. The halvings are spread evenly over the
  /// stages so intermediate values stay in range.
  void execute(Complex<T>* x, bool inverse, int halvings) const {
    const ComplexOps<A> c{arith_};
    for (std::size_t i = 0; i < n_; ++i) {
      const std::size_t r = reversed_[i];
      if (i < r) std::swap(x[i], x[r]);
    }
    int stage = 0;
    for (std::size_t len = 2; len <= n_; len <<= 1, ++stage) {
      const bool halve = halvings > 0 && ((stage + 1) * halvings) / log2n_ != (stage * halvings) / log2n_;
      const std::size_t half = len / 2;
      const std::size_t step = n_ / len;
      for (std::size_t start = 0; start < n_; start += len) {
        for (std::size_t j = 0; j < half; ++j) {
          Complex<T> w = twiddles_[j * step];
          if (inverse) w = c.conj(w);
          Complex<T> u = x[start + j];
          Complex<T> v = j == 0 ? x[start + j + half] : c.mul(x[start + j + half], w);
          if (halve) {
            u = c.half(u);
            v = c.half(v);
          }
          x[start + j] = c.add(u, v);
          x[start + j + half] = c.sub(u, v);
        }
      }
    }
  }

 private:
  std::size_t n_;
  int log2n_ = 0;
  A arith_;
  std::vector<std::uint32_t> reversed_;
  std::vector<Complex<T>> twiddles_;
};

/// Bluestein's algorithm (the chirp-z transform) for any length n.
///
///   X_k = w_k * sum_j (x_j w_j) conj(w_(k-j)),  with w_k = e^(-i pi k^2 / n)
///
/// The sum is a convolution, computed with radix-2 FFTs of length m >= 2n - 1.
template <class A>
class BluesteinPlan {
 public:
  using T = typename A::value_type;

  BluesteinPlan(std::size_t n, const A& arith)
      : n_(n), m_(next_power_of_two(2 * n - 1)), arith_(arith), inner_(m_, arith) {
    const ComplexOps<A> c{arith_};
    const int stages = inner_.stages();
    halvings_ = stages / 2;
    // The chirp in binary64. k^2 is reduced modulo 2n in integers first, so the
    // angle stays small and exact even when k^2 is far beyond 2^53.
    std::vector<std::complex<double>> chirp(n);
    for (std::size_t k = 0; k < n; ++k) {
      const std::uint64_t k2 = (static_cast<std::uint64_t>(k) * k) % (2 * static_cast<std::uint64_t>(n));
      const double angle = kPi * static_cast<double>(k2) / static_cast<double>(n);
      chirp[k] = {std::cos(angle), -std::sin(angle)};
    }
    chirp_.resize(n);
    for (std::size_t k = 0; k < n; ++k) chirp_[k] = c.from(chirp[k]);
    // The convolution kernel b = conj(w) laid out circularly, and its DFT, both in
    // binary64; the DFT is scaled by 2^-c_ so its largest value is at most 1, then
    // rounded to the working format once.
    std::vector<std::complex<double>> b(m_);
    b[0] = std::conj(chirp[0]);
    for (std::size_t k = 1; k < n; ++k) b[k] = b[m_ - k] = std::conj(chirp[k]);
    std::vector<std::complex<double>> kernel = exact_fft(b);
    double largest = 0;
    for (const auto& z : kernel) largest = std::max(largest, std::abs(z));
    kernel_shift_ = static_cast<int>(std::ceil(std::log2(largest)));
    kernel_.resize(m_);
    for (std::size_t k = 0; k < m_; ++k) kernel_[k] = c.from(std::ldexp(1.0, -kernel_shift_) * kernel[k]);
    // DFT(x) = out * 2^shift_; see execute() for the derivation.
    shift_ = 2 * halvings_ + kernel_shift_ - stages;
  }

  std::size_t size() const { return n_; }
  std::size_t convolution_size() const { return m_; }
  int shift() const { return shift_; }

  /// out = DFT(in) * 2^-shift() (forward) or sum_j in_j e^(+2 pi i jk/n) * 2^-shift() (inverse).
  void execute(const Complex<T>* in, Complex<T>* out, bool inverse, std::vector<Complex<T>>& work) const {
    const ComplexOps<A> c{arith_};
    work.assign(m_, Complex<T>{arith_.from(0.0), arith_.from(0.0)});
    // The inverse DFT is the conjugate of the forward DFT of the conjugate.
    for (std::size_t k = 0; k < n_; ++k) work[k] = c.mul(inverse ? c.conj(in[k]) : in[k], chirp_[k]);
    inner_.execute(work.data(), false, halvings_);           // A = DFT(a) 2^-f
    for (std::size_t k = 0; k < m_; ++k) work[k] = c.mul(work[k], kernel_[k]);  // C = A B'
    inner_.execute(work.data(), true, halvings_);            // y = (sum C e^+) 2^-f = 2^(L-2f-c) conv
    for (std::size_t k = 0; k < n_; ++k) {
      const Complex<T> z = c.mul(work[k], chirp_[k]);       // X 2^-(2f+c-L)
      out[k] = inverse ? c.conj(z) : z;
    }
  }

 private:
  // A plain binary64 radix-2 FFT, used only to precompute the kernel.
  static std::vector<std::complex<double>> exact_fft(std::vector<std::complex<double>> x) {
    Radix2Plan<Native<double>> plan(x.size(), Native<double>{});
    std::vector<Complex<double>> v(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) v[i] = {x[i].real(), x[i].imag()};
    plan.execute(v.data(), false, 0);
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = {v[i].re, v[i].im};
    return x;
  }

  std::size_t n_;
  std::size_t m_;
  A arith_;
  Radix2Plan<A> inner_;
  int halvings_ = 0;
  int kernel_shift_ = 0;
  int shift_ = 0;
  std::vector<Complex<T>> chirp_;
  std::vector<Complex<T>> kernel_;
};

/// A length-n transform: radix-2 or Bluestein, chosen once at construction.
template <class A>
class Fft {
 public:
  using T = typename A::value_type;

  Fft(std::size_t n, const A& arith, Algorithm algorithm = Algorithm::Auto) : n_(n) {
    if (n == 0) throw std::invalid_argument("FFT length must be positive");
    if (n == 1) {
      shift_ = 0;
    } else if (is_power_of_two(n) && algorithm == Algorithm::Auto) {
      radix2_ = std::make_unique<Radix2Plan<A>>(n, arith);
      shift_ = radix2_->stages() / 2;
    } else {
      bluestein_ = std::make_unique<BluesteinPlan<A>>(n, arith);
      shift_ = bluestein_->shift();
    }
  }

  std::size_t size() const { return n_; }
  /// Results are the DFT times 2^-shift().
  int shift() const { return shift_; }
  bool uses_bluestein() const { return static_cast<bool>(bluestein_); }
  /// Length of the power-of-two FFTs used inside (n itself for radix-2).
  std::size_t inner_size() const { return bluestein_ ? bluestein_->convolution_size() : n_; }

  /// out = DFT(in) * 2^-shift(); in and out may be the same buffer.
  void forward(const Complex<T>* in, Complex<T>* out, std::vector<Complex<T>>& work) const { run(in, out, false, work); }
  /// out = sum_j in_j e^(+2 pi i jk/n) * 2^-shift() (an unnormalised inverse).
  void inverse(const Complex<T>* in, Complex<T>* out, std::vector<Complex<T>>& work) const { run(in, out, true, work); }

 private:
  void run(const Complex<T>* in, Complex<T>* out, bool inverse, std::vector<Complex<T>>& work) const {
    if (n_ == 1) {
      out[0] = in[0];
    } else if (radix2_) {
      if (out != in) std::copy(in, in + n_, out);
      radix2_->execute(out, inverse, shift_);
    } else {
      bluestein_->execute(in, out, inverse, work);
    }
  }

  std::size_t n_;
  int shift_ = 0;
  std::unique_ptr<Radix2Plan<A>> radix2_;
  std::unique_ptr<BluesteinPlan<A>> bluestein_;
};

/// Builds each length's plan once and shares it; plans are read-only, so threads can share them.
template <class A>
class PlanCache {
 public:
  explicit PlanCache(const A& arith, Algorithm algorithm = Algorithm::Auto) : arith_(arith), algorithm_(algorithm) {}

  const Fft<A>& get(std::size_t n) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto& slot = plans_[n];
    if (!slot) slot = std::make_unique<Fft<A>>(n, arith_, algorithm_);
    return *slot;
  }

 private:
  A arith_;
  Algorithm algorithm_;
  std::mutex mutex_;
  std::map<std::size_t, std::unique_ptr<Fft<A>>> plans_;
};

/// The original project's transform, kept to measure what the rebuild fixed: a
/// radix-2 FFT that zero-pads to the next power of two and builds each stage's
/// twiddle factors by repeated multiplication (w = w * wn), with the angle itself
/// rounded to the format first. Returns DFT(padded x) with no scaling.
template <class A>
std::vector<Complex<typename A::value_type>> legacy_padded_fft(std::vector<Complex<typename A::value_type>> x,
                                                               const A& arith) {
  using T = typename A::value_type;
  const ComplexOps<A> c{arith};
  const std::size_t n = next_power_of_two(x.size());
  x.resize(n, Complex<T>{arith.from(0.0), arith.from(0.0)});
  const int bits = log2_of_power_of_two(n);
  for (std::size_t i = 0; i < n; ++i) {
    std::size_t r = 0;
    for (int b = 0; b < bits; ++b) r |= ((i >> b) & 1u) << (bits - 1 - b);
    if (i < r) std::swap(x[i], x[r]);
  }
  for (std::size_t len = 2; len <= n; len <<= 1) {
    const double angle = arith.to(arith.from(-2.0 * kPi / static_cast<double>(len)));
    const Complex<T> wn{arith.from(std::cos(angle)), arith.from(std::sin(angle))};
    for (std::size_t start = 0; start < n; start += len) {
      Complex<T> w{arith.from(1.0), arith.from(0.0)};
      for (std::size_t j = 0; j < len / 2; ++j) {
        const Complex<T> u = x[start + j];
        const Complex<T> v = c.mul(x[start + j + len / 2], w);
        x[start + j] = c.add(u, v);
        x[start + j + len / 2] = c.sub(u, v);
        w = c.mul(w, wn);
      }
    }
  }
  return x;
}

}  // namespace blurfft
