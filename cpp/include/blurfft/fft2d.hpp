// The 2-D DFT of a real image, computed in any arithmetic.
//
// A real image's spectrum is conjugate-symmetric, F(-u, -v) = conj F(u, v), so
// only the half-plane v = 0 .. W/2 is computed (the layout numpy.fft.rfft2
// uses). Two real rows are transformed with one complex FFT (z = a + ib) and
// separated afterwards, and the column pass runs on W/2 + 1 columns: about
// half the work of a complex 2-D FFT.
#pragma once

#include <complex>
#include <cstddef>
#include <vector>

#include "fft.hpp"
#include "parallel.hpp"

namespace blurfft {

/// The half-plane spectrum of an H x W real image, row-major, H x (W/2 + 1),
/// returned in binary64 with the internal power-of-two scaling undone, so in
/// binary64 arithmetic it equals numpy.fft.rfft2(image).
struct HalfSpectrum {
  std::size_t height = 0;
  std::size_t width = 0;  // of the image; the spectrum has width / 2 + 1 columns
  std::vector<std::complex<double>> values;

  std::size_t columns() const { return width / 2 + 1; }
  const std::complex<double>& at(std::size_t u, std::size_t v) const { return values[u * columns() + v]; }
};

/// The transform with plans the caller already built (row_fft for the width,
/// col_fft for the height), so repeated transforms of one size share them.
template <class A>
HalfSpectrum real_fft2d(const double* image, std::size_t height, std::size_t width, const A& arith,
                        const Fft<A>& row_fft, const Fft<A>& col_fft, int threads = 0) {
  using T = typename A::value_type;
  if (height == 0 || width == 0) throw std::invalid_argument("image must not be empty");
  if (row_fft.size() != width || col_fft.size() != height) throw std::invalid_argument("plans do not match the image size");
  const ComplexOps<A> c{arith};
  const std::size_t cols = width / 2 + 1;
  std::vector<Complex<T>> spectrum(height * cols);

  // Rows, two at a time: z = row_a + i row_b, then
  //   A_k = (Z_k + conj Z_(W-k)) / 2,   B_k = (Z_k - conj Z_(W-k)) / 2i.
  const std::size_t pairs = (height + 1) / 2;
  parallel_for(pairs, threads, [&](std::size_t begin, std::size_t end, int) {
    std::vector<Complex<T>> z(width);
    std::vector<Complex<T>> work;
    for (std::size_t p = begin; p < end; ++p) {
      const std::size_t ra = 2 * p;
      const std::size_t rb = ra + 1;
      const bool paired = rb < height;
      for (std::size_t x = 0; x < width; ++x) {
        z[x] = {arith.from(image[ra * width + x]), arith.from(paired ? image[rb * width + x] : 0.0)};
      }
      row_fft.forward(z.data(), z.data(), work);
      for (std::size_t k = 0; k < cols; ++k) {
        const Complex<T> zk = z[k];
        if (!paired) {
          spectrum[ra * cols + k] = zk;
          continue;
        }
        const Complex<T> mirror = c.conj(z[(width - k) % width]);
        const Complex<T> sum = c.add(zk, mirror);
        const Complex<T> diff = c.sub(zk, mirror);
        spectrum[ra * cols + k] = c.half(sum);
        // (diff / 2i) = (diff.im - i diff.re) / 2
        spectrum[rb * cols + k] = {arith.half(diff.im), arith.half(-diff.re)};
      }
    }
  });

  // Columns: a complex FFT down each of the W/2 + 1 columns.
  parallel_for(cols, threads, [&](std::size_t begin, std::size_t end, int) {
    std::vector<Complex<T>> column(height);
    std::vector<Complex<T>> work;
    for (std::size_t v = begin; v < end; ++v) {
      for (std::size_t u = 0; u < height; ++u) column[u] = spectrum[u * cols + v];
      col_fft.forward(column.data(), column.data(), work);
      for (std::size_t u = 0; u < height; ++u) spectrum[u * cols + v] = column[u];
    }
  });

  HalfSpectrum out;
  out.height = height;
  out.width = width;
  out.values.resize(spectrum.size());
  const double unscale = std::ldexp(1.0, row_fft.shift() + col_fft.shift());
  for (std::size_t i = 0; i < spectrum.size(); ++i) out.values[i] = unscale * c.to(spectrum[i]);
  return out;
}

template <class A>
HalfSpectrum real_fft2d(const double* image, std::size_t height, std::size_t width, const A& arith,
                        Algorithm algorithm = Algorithm::Auto, int threads = 0) {
  PlanCache<A> cache(arith, algorithm);
  const Fft<A>& row_fft = cache.get(width);
  const Fft<A>& col_fft = cache.get(height);
  return real_fft2d(image, height, width, arith, row_fft, col_fft, threads);
}

/// The original project's 2-D transform: rows then columns of legacy_padded_fft,
/// so the image is zero-padded to powers of two in both directions. Returns the
/// full (padded) magnitude spectrum, row-major, with its padded size.
template <class A>
std::vector<double> legacy_fft2d_magnitude(const double* image, std::size_t height, std::size_t width, const A& arith,
                                           std::size_t& padded_height, std::size_t& padded_width) {
  using T = typename A::value_type;
  const ComplexOps<A> c{arith};
  padded_width = next_power_of_two(width);
  padded_height = next_power_of_two(height);
  std::vector<std::vector<Complex<T>>> rows(padded_height, std::vector<Complex<T>>(padded_width, Complex<T>{arith.from(0.0), arith.from(0.0)}));
  for (std::size_t y = 0; y < height; ++y) {
    std::vector<Complex<T>> row(width);
    for (std::size_t x = 0; x < width; ++x) row[x] = {arith.from(image[y * width + x]), arith.from(0.0)};
    rows[y] = legacy_padded_fft(row, arith);
  }
  std::vector<double> magnitude(padded_height * padded_width);
  for (std::size_t x = 0; x < padded_width; ++x) {
    std::vector<Complex<T>> column(padded_height);
    for (std::size_t y = 0; y < padded_height; ++y) column[y] = rows[y][x];
    column = legacy_padded_fft(column, arith);
    for (std::size_t y = 0; y < padded_height; ++y) magnitude[y * padded_width + x] = std::abs(c.to(column[y]));
  }
  return magnitude;
}

}  // namespace blurfft
