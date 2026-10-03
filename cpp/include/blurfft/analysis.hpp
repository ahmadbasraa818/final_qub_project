// From a greyscale image to blur measures: preparation, the 2-D FFT in the
// requested precision, and the spectral measures, for a whole image or tiles.
#pragma once

#include <chrono>
#include <type_traits>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "fft2d.hpp"
#include "metrics.hpp"
#include "precision.hpp"

namespace blurfft {

/// A window tapers the image to zero at its edges. Without one, the jump
/// between opposite edges (the DFT treats the image as periodic) adds a bright
/// cross of false high-frequency energy to every spectrum.
enum class Window { None, Hann };

struct AnalysisOptions {
  Precision precision = Precision::native_double();
  Algorithm algorithm = Algorithm::Auto;
  Window window = Window::Hann;
  MetricOptions metrics{};
  int threads = 0;  // 0: one per core
};

struct Analysis {
  SpectralMetrics metrics;
  /// Sum of the squared window over the image: white noise of variance s^2 adds
  /// s^2 * window_energy to every frequency bin's expected power.
  double window_energy = 0;
  double fft_seconds = 0;
  std::size_t height = 0;
  std::size_t width = 0;
};

inline std::vector<double> window_weights(std::size_t n, Window window) {
  std::vector<double> w(n, 1.0);
  if (window == Window::Hann && n > 1) {
    for (std::size_t i = 0; i < n; ++i) w[i] = 0.5 - 0.5 * std::cos(2.0 * kPi * static_cast<double>(i) / static_cast<double>(n - 1));
  }
  return w;
}

inline double window_energy(std::size_t height, std::size_t width, Window window) {
  double ey = 0, ex = 0;
  for (double v : window_weights(height, window)) ey += v * v;
  for (double v : window_weights(width, window)) ex += v * v;
  return ey * ex;
}

/// Subtracts the mean (so DC does not dominate the range of narrow formats) and
/// applies the window. Pixel values are expected in [0, 1].
inline std::vector<double> prepare(const double* gray, std::size_t height, std::size_t width, Window window) {
  if (height < 2 || width < 2) throw std::invalid_argument("images must be at least 2 x 2 pixels");
  double mean = 0;
  for (std::size_t i = 0; i < height * width; ++i) mean += gray[i];
  mean /= static_cast<double>(height * width);
  const std::vector<double> wy = window_weights(height, window), wx = window_weights(width, window);
  std::vector<double> out(height * width);
  for (std::size_t y = 0; y < height; ++y) {
    for (std::size_t x = 0; x < width; ++x) out[y * width + x] = (gray[y * width + x] - mean) * wy[y] * wx[x];
  }
  return out;
}

inline HalfSpectrum spectrum(const double* gray, std::size_t height, std::size_t width, const AnalysisOptions& opt) {
  const std::vector<double> prepared = prepare(gray, height, width, opt.window);
  return with_arithmetic(opt.precision, [&](const auto& arith) {
    return real_fft2d(prepared.data(), height, width, arith, opt.algorithm, opt.threads);
  });
}

inline Analysis analyse(const double* gray, std::size_t height, std::size_t width, const AnalysisOptions& opt) {
  const std::vector<double> prepared = prepare(gray, height, width, opt.window);
  const auto start = std::chrono::steady_clock::now();
  HalfSpectrum s = with_arithmetic(opt.precision, [&](const auto& arith) {
    return real_fft2d(prepared.data(), height, width, arith, opt.algorithm, opt.threads);
  });
  const auto stop = std::chrono::steady_clock::now();
  Analysis a;
  a.fft_seconds = std::chrono::duration<double>(stop - start).count();
  a.metrics = spectral_metrics(s, opt.metrics, opt.threads);
  a.window_energy = window_energy(height, width, opt.window);
  a.height = height;
  a.width = width;
  return a;
}

/// Blur measures for overlapping square tiles, for images that are only partly
/// blurred. Tiles may be any size: Bluestein handles lengths such as 96 exactly.
struct BlurMap {
  std::size_t rows = 0;
  std::size_t cols = 0;
  std::size_t tile = 0;
  std::size_t stride = 0;
  std::vector<double> high_frequency_ratio;
  std::vector<double> slope;
  std::vector<double> energy;  // per pixel, so flat tiles (sky, walls) can be told apart
  std::vector<double> band_power;      // rows x cols x bands
  std::vector<double> band_power_min;  // rows x cols x bands, weakest direction
  std::vector<double> anisotropy;
  std::vector<double> orientation_deg;
  std::size_t bands = 0;
};

inline BlurMap blur_map(const double* gray, std::size_t height, std::size_t width, std::size_t tile, std::size_t stride,
                        const AnalysisOptions& opt) {
  if (tile < 8 || tile > height || tile > width) throw std::invalid_argument("tile must be at least 8 and fit the image");
  if (stride == 0) throw std::invalid_argument("stride must be positive");
  BlurMap map;
  map.tile = tile;
  map.stride = stride;
  map.rows = (height - tile) / stride + 1;
  map.cols = (width - tile) / stride + 1;
  const std::size_t count = map.rows * map.cols;
  map.high_frequency_ratio.assign(count, 0.0);
  map.slope.assign(count, 0.0);
  map.energy.assign(count, 0.0);
  map.bands = opt.metrics.band_edges.size() > 1 ? opt.metrics.band_edges.size() - 1 : 0;
  map.band_power.assign(count * map.bands, 0.0);
  map.band_power_min.assign(count * map.bands, 0.0);
  map.anisotropy.assign(count, 0.0);
  map.orientation_deg.assign(count, 0.0);
  with_arithmetic(opt.precision, [&](const auto& arith) {
    using A = std::decay_t<decltype(arith)>;
    const Fft<A> plan(tile, arith, opt.algorithm);  // one plan, shared read-only by every tile
    parallel_for(count, opt.threads, [&](std::size_t begin, std::size_t end, int) {
      std::vector<double> patch(tile * tile);
      for (std::size_t i = begin; i < end; ++i) {
        const std::size_t r = i / map.cols, c = i % map.cols;
        for (std::size_t y = 0; y < tile; ++y) {
          for (std::size_t x = 0; x < tile; ++x) patch[y * tile + x] = gray[(r * stride + y) * width + c * stride + x];
        }
        const std::vector<double> prepared = prepare(patch.data(), tile, tile, opt.window);
        const HalfSpectrum s = real_fft2d(prepared.data(), tile, tile, arith, plan, plan, 1);
        const SpectralMetrics m = spectral_metrics(s, opt.metrics);
        map.high_frequency_ratio[i] = m.high_frequency_ratio;
        map.slope[i] = m.slope;
        map.energy[i] = m.total_energy / static_cast<double>(tile * tile * tile * tile);
        map.anisotropy[i] = m.anisotropy;
        map.orientation_deg[i] = m.orientation_deg;
        for (std::size_t b = 0; b < map.bands; ++b) {
          map.band_power[i * map.bands + b] = m.band_power[b];
          map.band_power_min[i * map.bands + b] = m.band_power_min[b];
        }
      }
    });
    return 0;
  });
  return map;
}

}  // namespace blurfft
