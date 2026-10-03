// Floating-point formats and the arithmetic used inside the FFT.
//
// The project studies approximate computing through reduced precision. Two
// kinds of arithmetic are provided, both used through the same interface so
// every kernel is written once:
//
//   Native<T>   real hardware arithmetic: double, float and, where the compiler
//               supports it, _Float16. Use these to measure speed and memory.
//   Emulated    any IEEE 754-style format, chosen at run time by its exponent
//               and mantissa widths. Every operation is computed in binary64
//               and rounded to the target format, as FloatX does. Use this to
//               measure accuracy at formats the hardware does not have.
#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace blurfft {

/// An IEEE 754-style binary format: one sign bit, `exponent_bits` of biased
/// exponent and `mantissa_bits` of stored fraction, with subnormals,
/// round-to-nearest-even and overflow to infinity. binary64 is {11, 52},
/// binary32 {8, 23}, binary16 {5, 10} and bfloat16 {8, 7}.
struct Format {
  int exponent_bits = 11;
  int mantissa_bits = 52;

  int bias() const { return (1 << (exponent_bits - 1)) - 1; }
  /// Exponent of the smallest normal number.
  int min_exponent() const { return 1 - bias(); }
  /// Exponent of the largest finite number.
  int max_exponent() const { return bias(); }
  int total_bits() const { return 1 + exponent_bits + mantissa_bits; }
  double max_finite() const { return std::ldexp(2.0 - std::ldexp(1.0, -mantissa_bits), max_exponent()); }
  double min_normal() const { return std::ldexp(1.0, min_exponent()); }
  /// Distance from 1 to the next representable number; unit roundoff is half of it.
  double epsilon() const { return std::ldexp(1.0, -mantissa_bits); }
  bool is_binary64() const { return exponent_bits == 11 && mantissa_bits == 52; }
};

inline void validate(const Format& f) {
  if (f.exponent_bits < 2 || f.exponent_bits > 11 || f.mantissa_bits < 1 || f.mantissa_bits > 52) {
    throw std::invalid_argument("a format needs 2 to 11 exponent bits and 1 to 52 mantissa bits");
  }
}

/// Rounds a binary64 value to the nearest value of `f`, ties to even.
///
/// The value is scaled by a power of two so the bits to keep form an integer,
/// rounded once with nearbyint, and scaled back. Power-of-two scaling is exact,
/// so there is exactly one rounding: the result is the correctly rounded value
/// for every format up to binary64, including subnormals.
inline double quantize(double x, const Format& f) noexcept {
  if (f.is_binary64() || x == 0.0 || !std::isfinite(x)) return x;
  int e = 0;
  std::frexp(x, &e);  // x = m * 2^e with 0.5 <= |m| < 1, so the leading bit is worth 2^(e - 1)
  int exponent = e - 1;
  if (exponent < f.min_exponent()) exponent = f.min_exponent();  // gradual underflow: fixed quantum
  const double scaled = std::ldexp(x, f.mantissa_bits - exponent);
  const double rounded = std::ldexp(std::nearbyint(scaled), exponent - f.mantissa_bits);
  if (std::fabs(rounded) > f.max_finite()) return std::copysign(std::numeric_limits<double>::infinity(), x);
  return rounded;
}

/// Hardware arithmetic on T.
template <class T>
struct Native {
  using value_type = T;
  T from(double x) const { return static_cast<T>(x); }
  double to(T x) const { return static_cast<double>(x); }
  T add(T a, T b) const { return a + b; }
  T sub(T a, T b) const { return a - b; }
  T mul(T a, T b) const { return a * b; }
  /// Multiplication by a power of two; exact unless the result leaves the normal range.
  T scale(T a, int exponent) const { return static_cast<T>(std::ldexp(static_cast<double>(a), exponent)); }
  T half(T a) const { return a * static_cast<T>(0.5); }
  int bits_per_value() const { return static_cast<int>(sizeof(T) * 8); }
};

/// Software arithmetic for any Format: every result is rounded to the format.
struct Emulated {
  using value_type = double;
  Format format{};
  double from(double x) const { return quantize(x, format); }
  double to(double x) const { return x; }
  double add(double a, double b) const { return quantize(a + b, format); }
  double sub(double a, double b) const { return quantize(a - b, format); }
  double mul(double a, double b) const { return quantize(a * b, format); }
  double scale(double a, int exponent) const { return quantize(std::ldexp(a, exponent), format); }
  double half(double a) const { return quantize(a * 0.5, format); }
  int bits_per_value() const { return format.total_bits(); }
};

/// Whether this build has hardware half-precision arithmetic (_Float16).
#if defined(__FLT16_MANT_DIG__) || (defined(__clang__) && (defined(__aarch64__) || defined(__arm64__)))
#define BLURFFT_HAS_FLOAT16 1
using half_t = _Float16;
#else
#define BLURFFT_HAS_FLOAT16 0
#endif

constexpr bool has_native_half() { return BLURFFT_HAS_FLOAT16 == 1; }

/// How the FFT should do its arithmetic, as chosen by the caller at run time.
struct Precision {
  enum class Kind { NativeDouble, NativeFloat, NativeHalf, Emulated };
  Kind kind = Kind::NativeDouble;
  Format format{};  // the format computed in; for native kinds, the hardware format

  static Precision native_double() { return {Kind::NativeDouble, Format{11, 52}}; }
  static Precision native_float() { return {Kind::NativeFloat, Format{8, 23}}; }
  static Precision native_half() { return {Kind::NativeHalf, Format{5, 10}}; }
  static Precision emulated(int exponent_bits, int mantissa_bits) {
    Format f{exponent_bits, mantissa_bits};
    validate(f);
    return {Kind::Emulated, f};
  }

  std::string name() const {
    switch (kind) {
      case Kind::NativeDouble: return "float64";
      case Kind::NativeFloat: return "float32";
      case Kind::NativeHalf: return "float16";
      case Kind::Emulated: break;
    }
    return "e" + std::to_string(format.exponent_bits) + "m" + std::to_string(format.mantissa_bits);
  }
};

/// Calls fn with the arithmetic object for p, so kernels are instantiated once per kind.
template <class Fn>
decltype(auto) with_arithmetic(const Precision& p, Fn&& fn) {
  switch (p.kind) {
    case Precision::Kind::NativeDouble: return fn(Native<double>{});
    case Precision::Kind::NativeFloat: return fn(Native<float>{});
    case Precision::Kind::NativeHalf:
#if BLURFFT_HAS_FLOAT16
      return fn(Native<half_t>{});
#else
      throw std::invalid_argument("this build has no hardware float16; use emulated e5m10 instead");
#endif
    case Precision::Kind::Emulated: break;
  }
  return fn(Emulated{p.format});
}

}  // namespace blurfft
