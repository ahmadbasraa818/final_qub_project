// Python bindings for the C++ core (module blurfft._core).
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <blurfft/analysis.hpp>
#include <blurfft/fft.hpp>
#include <blurfft/fft2d.hpp>
#include <blurfft/precision.hpp>

#include <complex>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;
using namespace blurfft;

namespace {

using Image = py::array_t<double, py::array::c_style | py::array::forcecast>;

Precision make_precision(const std::string& kind, int exponent_bits, int mantissa_bits) {
  if (kind == "float64") return Precision::native_double();
  if (kind == "float32") return Precision::native_float();
  if (kind == "float16") return Precision::native_half();
  if (kind == "emulated") return Precision::emulated(exponent_bits, mantissa_bits);
  throw std::invalid_argument("precision kind must be float64, float32, float16 or emulated, not " + kind);
}

Algorithm make_algorithm(const std::string& name) {
  if (name == "auto") return Algorithm::Auto;
  if (name == "bluestein") return Algorithm::Bluestein;
  throw std::invalid_argument("algorithm must be auto or bluestein");
}

Window make_window(const std::string& name) {
  if (name == "hann") return Window::Hann;
  if (name == "none") return Window::None;
  throw std::invalid_argument("window must be hann or none");
}

std::pair<std::size_t, std::size_t> shape_of(const Image& image) {
  if (image.ndim() != 2) throw std::invalid_argument("expected a 2-D greyscale image");
  return {static_cast<std::size_t>(image.shape(0)), static_cast<std::size_t>(image.shape(1))};
}

AnalysisOptions make_options(const std::string& kind, int exponent_bits, int mantissa_bits, const std::string& window,
                             double cutoff, double fit_low, double fit_high, int radial_bins, const std::string& algorithm,
                             int threads, const std::vector<double>& band_edges = MetricOptions{}.band_edges) {
  AnalysisOptions opt;
  opt.precision = make_precision(kind, exponent_bits, mantissa_bits);
  opt.window = make_window(window);
  opt.algorithm = make_algorithm(algorithm);
  opt.metrics.cutoff = cutoff;
  opt.metrics.fit_low = fit_low;
  opt.metrics.fit_high = fit_high;
  opt.metrics.radial_bins = radial_bins;
  for (std::size_t i = 1; i < band_edges.size(); ++i) {
    if (!(band_edges[i] > band_edges[i - 1])) throw std::invalid_argument("band edges must increase");
  }
  opt.metrics.band_edges = band_edges;
  opt.threads = threads;
  return opt;
}

py::dict metrics_to_dict(const SpectralMetrics& m) {
  py::dict d;
  d["high_frequency_ratio"] = m.high_frequency_ratio;
  d["slope"] = m.slope;
  d["fit_r2"] = m.fit_r2;
  d["anisotropy"] = m.anisotropy;
  d["orientation_deg"] = m.orientation_deg;
  d["total_energy"] = m.total_energy;
  d["band_power"] = py::array_t<double>(m.band_power.size(), m.band_power.data());
  d["band_power_min"] = py::array_t<double>(m.band_power_min.size(), m.band_power_min.data());
  d["radial_frequency"] = py::array_t<double>(m.radial_frequency.size(), m.radial_frequency.data());
  d["radial_power"] = py::array_t<double>(m.radial_power.size(), m.radial_power.data());
  return d;
}

py::array_t<std::complex<double>> to_array(const std::vector<std::complex<double>>& v, std::vector<py::ssize_t> shape) {
  py::array_t<std::complex<double>> out(shape);
  std::copy(v.begin(), v.end(), out.mutable_data());
  return out;
}

}  // namespace

PYBIND11_MODULE(_core, m) {
  m.doc() = "C++ core of blurfft: Bluestein FFTs in any floating-point precision, and spectral blur measures.";

  m.def("has_native_half", &has_native_half, "Whether this build has hardware float16 (_Float16) arithmetic.");

  m.def(
      "quantize",
      [](py::array_t<double, py::array::c_style | py::array::forcecast> x, int exponent_bits, int mantissa_bits) {
        Format f{exponent_bits, mantissa_bits};
        validate(f);
        py::array_t<double> out(x.request().shape);
        const double* in = x.data();
        double* o = out.mutable_data();
        for (py::ssize_t i = 0; i < x.size(); ++i) o[i] = quantize(in[i], f);
        return out;
      },
      py::arg("x"), py::arg("exponent_bits"), py::arg("mantissa_bits"),
      "Rounds every value to the nearest value of the format (ties to even).");

  m.def(
      "fft",
      [](py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> x, const std::string& kind,
         int exponent_bits, int mantissa_bits, bool inverse, const std::string& algorithm) {
        if (x.ndim() != 1) throw std::invalid_argument("expected a 1-D array");
        const std::vector<std::complex<double>> input(x.data(), x.data() + x.size());
        const Precision precision = make_precision(kind, exponent_bits, mantissa_bits);
        const Algorithm algo = make_algorithm(algorithm);
        std::vector<std::complex<double>> result;
        {
          py::gil_scoped_release release;
          result = with_arithmetic(precision, [&](const auto& arith) {
            using A = std::decay_t<decltype(arith)>;
            using T = typename A::value_type;
            const ComplexOps<A> c{arith};
            Fft<A> plan(input.size(), arith, algo);
            std::vector<Complex<T>> buf(input.size()), work;
            for (std::size_t i = 0; i < input.size(); ++i) buf[i] = c.from(input[i]);
            if (inverse) {
              plan.inverse(buf.data(), buf.data(), work);
            } else {
              plan.forward(buf.data(), buf.data(), work);
            }
            const double unscale = std::ldexp(1.0, plan.shift());
            std::vector<std::complex<double>> out(input.size());
            for (std::size_t i = 0; i < input.size(); ++i) out[i] = unscale * c.to(buf[i]);
            return out;
          });
        }
        return to_array(result, {static_cast<py::ssize_t>(result.size())});
      },
      py::arg("x"), py::arg("kind") = "float64", py::arg("exponent_bits") = 11, py::arg("mantissa_bits") = 52,
      py::arg("inverse") = false, py::arg("algorithm") = "auto",
      "DFT of a 1-D array (inverse: the unnormalised sum with e^+). Computed in the given precision, returned in float64.");

  m.def(
      "plan_info",
      [](std::size_t n, const std::string& algorithm) {
        Fft<Native<double>> plan(n, Native<double>{}, make_algorithm(algorithm));
        py::dict d;
        d["length"] = n;
        d["bluestein"] = plan.uses_bluestein();
        d["inner_size"] = plan.inner_size();
        d["shift"] = plan.shift();
        return d;
      },
      py::arg("n"), py::arg("algorithm") = "auto", "How a length is transformed: radix-2 or Bluestein, and the inner FFT size.");

  m.def(
      "rfft2",
      [](Image image, const std::string& kind, int exponent_bits, int mantissa_bits, const std::string& algorithm, int threads) {
        const auto [h, w] = shape_of(image);
        const Precision precision = make_precision(kind, exponent_bits, mantissa_bits);
        const Algorithm algo = make_algorithm(algorithm);
        HalfSpectrum s;
        {
          py::gil_scoped_release release;
          s = with_arithmetic(precision, [&](const auto& arith) { return real_fft2d(image.data(), h, w, arith, algo, threads); });
        }
        return to_array(s.values, {static_cast<py::ssize_t>(h), static_cast<py::ssize_t>(s.columns())});
      },
      py::arg("image"), py::arg("kind") = "float64", py::arg("exponent_bits") = 11, py::arg("mantissa_bits") = 52,
      py::arg("algorithm") = "auto", py::arg("threads") = 0,
      "Half-plane 2-D DFT of a real image, like numpy.fft.rfft2, computed in the given precision.");

  m.def(
      "spectrum",
      [](Image image, const std::string& kind, int exponent_bits, int mantissa_bits, const std::string& window,
         const std::string& algorithm, int threads) {
        const auto [h, w] = shape_of(image);
        AnalysisOptions opt = make_options(kind, exponent_bits, mantissa_bits, window, 0.25, 0.05, 0.35, 64, algorithm, threads);
        HalfSpectrum s;
        {
          py::gil_scoped_release release;
          s = spectrum(image.data(), h, w, opt);
        }
        return to_array(s.values, {static_cast<py::ssize_t>(h), static_cast<py::ssize_t>(s.columns())});
      },
      py::arg("image"), py::arg("kind") = "float64", py::arg("exponent_bits") = 11, py::arg("mantissa_bits") = 52,
      py::arg("window") = "hann", py::arg("algorithm") = "auto", py::arg("threads") = 0,
      "The prepared (mean removed, windowed) image's half-plane spectrum, as the measures see it.");

  m.def(
      "analyse",
      [](Image image, const std::string& kind, int exponent_bits, int mantissa_bits, const std::string& window, double cutoff,
         double fit_low, double fit_high, int radial_bins, const std::string& algorithm, int threads,
         const std::vector<double>& band_edges) {
        const auto [h, w] = shape_of(image);
        const AnalysisOptions opt = make_options(kind, exponent_bits, mantissa_bits, window, cutoff, fit_low, fit_high,
                                                 radial_bins, algorithm, threads, band_edges);
        Analysis a;
        {
          py::gil_scoped_release release;
          a = analyse(image.data(), h, w, opt);
        }
        py::dict d = metrics_to_dict(a.metrics);
        d["fft_seconds"] = a.fft_seconds;
        d["window_energy"] = a.window_energy;
        d["height"] = a.height;
        d["width"] = a.width;
        return d;
      },
      py::arg("image"), py::arg("kind") = "float64", py::arg("exponent_bits") = 11, py::arg("mantissa_bits") = 52,
      py::arg("window") = "hann", py::arg("cutoff") = 0.25, py::arg("fit_low") = 0.05, py::arg("fit_high") = 0.35,
      py::arg("radial_bins") = 64, py::arg("algorithm") = "auto", py::arg("threads") = 0,
      py::arg("band_edges") = MetricOptions{}.band_edges,
      "Spectral blur measures of a greyscale image with values in [0, 1].");

  m.def(
      "blur_map",
      [](Image image, std::size_t tile, std::size_t stride, const std::string& kind, int exponent_bits, int mantissa_bits,
         const std::string& window, double cutoff, double fit_low, double fit_high, int radial_bins, const std::string& algorithm,
         int threads, const std::vector<double>& band_edges) {
        const auto [h, w] = shape_of(image);
        const AnalysisOptions opt = make_options(kind, exponent_bits, mantissa_bits, window, cutoff, fit_low, fit_high,
                                                 radial_bins, algorithm, threads, band_edges);
        BlurMap map;
        {
          py::gil_scoped_release release;
          map = blur_map(image.data(), h, w, tile, stride, opt);
        }
        const std::vector<py::ssize_t> shape{static_cast<py::ssize_t>(map.rows), static_cast<py::ssize_t>(map.cols)};
        auto grid = [&](const std::vector<double>& v) {
          py::array_t<double> a(shape);
          std::copy(v.begin(), v.end(), a.mutable_data());
          return a;
        };
        py::dict d;
        d["high_frequency_ratio"] = grid(map.high_frequency_ratio);
        d["slope"] = grid(map.slope);
        d["energy"] = grid(map.energy);
        d["anisotropy"] = grid(map.anisotropy);
        d["orientation_deg"] = grid(map.orientation_deg);
        auto per_band = [&](const std::vector<double>& v) {
          py::array_t<double> a({shape[0], shape[1], static_cast<py::ssize_t>(map.bands)});
          std::copy(v.begin(), v.end(), a.mutable_data());
          return a;
        };
        d["band_power"] = per_band(map.band_power);
        d["band_power_min"] = per_band(map.band_power_min);
        d["window_energy"] = window_energy(map.tile, map.tile, opt.window);
        d["tile"] = map.tile;
        d["stride"] = map.stride;
        return d;
      },
      py::arg("image"), py::arg("tile"), py::arg("stride"), py::arg("kind") = "float64", py::arg("exponent_bits") = 11,
      py::arg("mantissa_bits") = 52, py::arg("window") = "hann", py::arg("cutoff") = 0.25, py::arg("fit_low") = 0.05,
      py::arg("fit_high") = 0.35, py::arg("radial_bins") = 32, py::arg("algorithm") = "auto", py::arg("threads") = 0,
      py::arg("band_edges") = MetricOptions{}.band_edges, "Blur measures for overlapping square tiles.");

  m.def(
      "legacy_fft2_magnitude",
      [](Image image, const std::string& kind, int exponent_bits, int mantissa_bits) {
        const auto [h, w] = shape_of(image);
        const Precision precision = make_precision(kind, exponent_bits, mantissa_bits);
        std::size_t ph = 0, pw = 0;
        std::vector<double> mag;
        {
          py::gil_scoped_release release;
          mag = with_arithmetic(precision, [&](const auto& arith) { return legacy_fft2d_magnitude(image.data(), h, w, arith, ph, pw); });
        }
        py::array_t<double> out({static_cast<py::ssize_t>(ph), static_cast<py::ssize_t>(pw)});
        std::copy(mag.begin(), mag.end(), out.mutable_data());
        return out;
      },
      py::arg("image"), py::arg("kind") = "emulated", py::arg("exponent_bits") = 8, py::arg("mantissa_bits") = 12,
      "The original project's transform: zero-padded radix-2 with twiddles built by repeated multiplication.");
}
