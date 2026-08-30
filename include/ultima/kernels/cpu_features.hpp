#pragma once

#include <cstddef>
#include <string>

namespace ultima::kernels {

struct CpuFeatures {
    bool has_sse2   = false;
    bool has_avx    = false;
    bool has_avx2   = false;
    bool has_fma    = false;
    bool has_avx512f= false;

    std::size_t logical_cores  = 0;
    std::size_t physical_cores = 0;   // best-effort; falls back to logical on failure

    std::string cpu_brand;
};

// Detect once and cache. Thread-safe (init inside a call_once).
const CpuFeatures& cpu_features();

// Fatal check for our v0.1 required baseline. Prints a clear message to
// stderr and calls std::exit(1) if AVX2 or FMA is missing.
void require_v01_baseline_or_die();

} // namespace ultima::kernels
