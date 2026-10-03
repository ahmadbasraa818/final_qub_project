// Blur measures computed from an image's spectrum.
//
// Blur is a low-pass filter: it removes fine detail, which lives at high
// spatial frequencies. Every measure here looks at how the image's energy is
// spread over frequency, with frequency in cycles per pixel so the measures
// mean the same thing at any image size.
//
//   band_power            mean power in each radial frequency band: the
//                         detector's features (its model learns how much each
//                         band matters, as a generalised high-pass filter)
//   band_power_min        the same, in the band's weakest direction (of eight
//                         22.5-degree sectors): motion blur removes detail in one
//                         direction only, which the mean alone would hide
//   high_frequency_ratio  share of the energy at radius >= cutoff
//   slope                 alpha in P(f) ~ f^-alpha: natural images sit near 2,
//                         blur makes the fall-off steeper (Liu, Li and Jia,
//                         "Image partial blur detection and classification", 2008)
//   anisotropy            how much the energy favours one direction, from the
//                         spectrum's structure tensor: motion blur is directional,
//                         defocus is not
//   orientation           the likely blur direction, in degrees from the x axis
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "fft2d.hpp"
#include "parallel.hpp"

namespace blurfft {

struct MetricOptions {
  double cutoff = 0.25;    // cycles per pixel
  double fit_low = 0.05;   // band for the slope and the tensor
  double fit_high = 0.35;
  int radial_bins = 64;    // over radii (0, 0.5]
  /// Edges of the radial bands for band_power, cycles per pixel, increasing.
  std::vector<double> band_edges{0.02, 0.05, 0.10, 0.18, 0.26, 0.34, 0.42, 0.50};
  /// Direction sectors over 180 degrees for band_power_min.
  int sectors = 8;
};

struct SpectralMetrics {
  double total_energy = 0;  // all bins except DC
  double high_frequency_ratio = 0;
  double slope = 0;
  double fit_r2 = 0;
  double anisotropy = 0;
  double orientation_deg = 0;
  std::vector<double> band_power;        // mean power per band of MetricOptions::band_edges
  std::vector<double> band_power_min;    // mean power in each band's weakest direction
  std::vector<double> radial_frequency;  // bin centres, cycles per pixel
  std::vector<double> radial_power;      // mean power per bin (0 where a bin is empty)
};

/// Signed frequency of row u of an n-row spectrum, in cycles per sample.
inline double signed_frequency(std::size_t u, std::size_t n) {
  const double k = u <= n / 2 ? static_cast<double>(u) : static_cast<double>(u) - static_cast<double>(n);
  return k / static_cast<double>(n);
}

namespace detail {

/// Sums over part of the spectrum; one per thread, added together at the end.
struct SpectrumSums {
  double total = 0, high = 0, txx = 0, tyy = 0, txy = 0;
  std::vector<double> bin_power, bin_weight, band_sum, band_weight, sector_sum, sector_weight;

  SpectrumSums(int bins, std::size_t bands, int sectors)
      : bin_power(bins, 0.0), bin_weight(bins, 0.0), band_sum(bands, 0.0), band_weight(bands, 0.0),
        sector_sum(bands * sectors, 0.0), sector_weight(bands * sectors, 0.0) {}

  void add(const SpectrumSums& o) {
    total += o.total;
    high += o.high;
    txx += o.txx;
    tyy += o.tyy;
    txy += o.txy;
    auto merge = [](std::vector<double>& a, const std::vector<double>& b) {
      for (std::size_t i = 0; i < a.size(); ++i) a[i] += b[i];
    };
    merge(bin_power, o.bin_power);
    merge(bin_weight, o.bin_weight);
    merge(band_sum, o.band_sum);
    merge(band_weight, o.band_weight);
    merge(sector_sum, o.sector_sum);
    merge(sector_weight, o.sector_weight);
  }
};

}  // namespace detail

inline SpectralMetrics spectral_metrics(const HalfSpectrum& s, const MetricOptions& opt = {}, int threads = 1) {
  SpectralMetrics m;
  const std::size_t cols = s.columns();
  const int bins = std::max(4, opt.radial_bins);
  const std::size_t bands = opt.band_edges.size() > 1 ? opt.band_edges.size() - 1 : 0;
  const int sectors = std::max(1, opt.sectors);
  const int workers = std::max(1, std::min(resolve_threads(threads), static_cast<int>(s.height)));
  std::vector<detail::SpectrumSums> partial(workers, detail::SpectrumSums(bins, bands, sectors));
  parallel_for(s.height, workers, [&](std::size_t begin, std::size_t end, int worker) {
    detail::SpectrumSums& acc = partial[worker];
    for (std::size_t u = begin; u < end; ++u) {
      const double fy = signed_frequency(u, s.height);
      for (std::size_t v = 0; v < cols; ++v) {
        if (u == 0 && v == 0) continue;  // DC carries brightness, not detail
        const double fx = static_cast<double>(v) / static_cast<double>(s.width);
        // Columns 1 .. W/2 - 1 stand for themselves and their mirror image.
        const bool unpaired = v == 0 || (s.width % 2 == 0 && v == s.width / 2);
        const double weight = unpaired ? 1.0 : 2.0;
        const double power = std::norm(s.at(u, v));
        const double radius = std::sqrt(fx * fx + fy * fy);
        acc.total += weight * power;
        if (radius >= opt.cutoff) acc.high += weight * power;
        if (radius <= 0.5) {
          const int b = std::min(bins - 1, static_cast<int>(radius / 0.5 * bins));
          acc.bin_power[b] += weight * power;
          acc.bin_weight[b] += weight;
        }
        for (std::size_t b = 0; b < bands; ++b) {
          if (radius >= opt.band_edges[b] && radius < opt.band_edges[b + 1]) {
            acc.band_sum[b] += weight * power;
            acc.band_weight[b] += weight;
            // A bin and its mirror share a direction modulo 180 degrees.
            double angle = std::atan2(fy, fx) * 180.0 / kPi;
            if (angle < 0) angle += 180.0;
            const int sector = std::min(sectors - 1, static_cast<int>(angle / (180.0 / sectors)));
            acc.sector_sum[b * sectors + sector] += weight * power;
            acc.sector_weight[b * sectors + sector] += weight;
            break;
          }
        }
        if (radius >= opt.fit_low && radius <= opt.fit_high) {
          // Whitened by radius^2 (natural images fall off as 1/f^2), so every
          // frequency in the band counts, not just the lowest.
          const double w = weight * power;  // power * r^2 * (unit direction)^2 = power * f^2
          acc.txx += w * fx * fx;
          acc.tyy += w * fy * fy;
          acc.txy += w * fx * fy;
        }
      }
    }
  });
  detail::SpectrumSums sum = partial[0];
  for (int w = 1; w < workers; ++w) sum.add(partial[w]);
  m.total_energy = sum.total;
  const double high = sum.high, txx = sum.txx, tyy = sum.tyy, txy = sum.txy;
  const std::vector<double>&bin_power = sum.bin_power, &bin_weight = sum.bin_weight;
  const std::vector<double>&band_sum = sum.band_sum, &band_weight = sum.band_weight;
  const std::vector<double>&sector_sum = sum.sector_sum, &sector_weight = sum.sector_weight;
  m.high_frequency_ratio = m.total_energy > 0 ? high / m.total_energy : 0.0;
  m.band_power.resize(bands);
  m.band_power_min.resize(bands);
  for (std::size_t b = 0; b < bands; ++b) {
    m.band_power[b] = band_weight[b] > 0 ? band_sum[b] / band_weight[b] : 0.0;
    double weakest = -1;
    for (int k = 0; k < sectors; ++k) {
      const double w = sector_weight[b * sectors + k];
      if (w <= 0) continue;
      const double mean = sector_sum[b * sectors + k] / w;
      if (weakest < 0 || mean < weakest) weakest = mean;
    }
    m.band_power_min[b] = weakest < 0 ? m.band_power[b] : weakest;
  }

  m.radial_frequency.resize(bins);
  m.radial_power.resize(bins);
  double sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
  int n = 0;
  for (int b = 0; b < bins; ++b) {
    const double f = (b + 0.5) / bins * 0.5;
    const double p = bin_weight[b] > 0 ? bin_power[b] / bin_weight[b] : 0.0;
    m.radial_frequency[b] = f;
    m.radial_power[b] = p;
    if (f >= opt.fit_low && f <= opt.fit_high && p > 0) {
      const double x = std::log(f), y = std::log(p);
      sx += x;
      sy += y;
      sxx += x * x;
      syy += y * y;
      sxy += x * y;
      ++n;
    }
  }
  if (n >= 3) {
    const double vx = sxx - sx * sx / n, vy = syy - sy * sy / n, cxy = sxy - sx * sy / n;
    if (vx > 0) {
      m.slope = -cxy / vx;
      m.fit_r2 = vy > 0 ? (cxy * cxy) / (vx * vy) : 1.0;
    }
  }

  const double trace = txx + tyy;
  if (trace > 0) {
    const double gap = std::sqrt((txx - tyy) * (txx - tyy) + 4 * txy * txy);
    m.anisotropy = gap / trace;
    // The spectrum's dominant direction, in image coordinates (x right, y down);
    // blur smears along the perpendicular direction.
    const double spectral = 0.5 * std::atan2(2 * txy, txx - tyy);
    double blur = spectral * 180.0 / kPi + 90.0;
    blur = std::fmod(std::fmod(blur, 180.0) + 180.0, 180.0);
    m.orientation_deg = blur;
  }
  return m;
}

}  // namespace blurfft
