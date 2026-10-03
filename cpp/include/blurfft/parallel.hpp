// A minimal parallel-for: splits [0, count) into contiguous chunks, one per thread.
#pragma once

#include <algorithm>
#include <cstddef>
#include <exception>
#include <thread>
#include <vector>

namespace blurfft {

inline int resolve_threads(int requested) {
  if (requested > 0) return requested;
  const unsigned hw = std::thread::hardware_concurrency();
  return hw == 0 ? 1 : static_cast<int>(hw);
}

/// Calls body(begin, end, worker) on disjoint chunks covering [0, count).
/// Exceptions thrown by any worker are rethrown on the calling thread.
template <class Body>
void parallel_for(std::size_t count, int threads, Body&& body) {
  const std::size_t workers = std::min<std::size_t>(static_cast<std::size_t>(resolve_threads(threads)), std::max<std::size_t>(count, 1));
  if (workers <= 1 || count < 2) {
    body(std::size_t{0}, count, 0);
    return;
  }
  std::vector<std::thread> pool;
  std::vector<std::exception_ptr> errors(workers);
  const std::size_t chunk = (count + workers - 1) / workers;
  for (std::size_t w = 0; w < workers; ++w) {
    const std::size_t begin = w * chunk;
    const std::size_t end = std::min(count, begin + chunk);
    if (begin >= end) break;
    pool.emplace_back([&, begin, end, w] {
      try {
        body(begin, end, static_cast<int>(w));
      } catch (...) {
        errors[w] = std::current_exception();
      }
    });
  }
  for (auto& t : pool) t.join();
  for (const auto& e : errors) {
    if (e) std::rethrow_exception(e);
  }
}

}  // namespace blurfft
