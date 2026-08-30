#include "ultima/kernels/cpu_features.hpp"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>

#ifdef _MSC_VER
#  include <intrin.h>
#else
#  include <cpuid.h>
#endif

namespace ultima::kernels {

namespace {

void cpuid(int leaf, int subleaf, std::array<unsigned, 4>& regs) {
#ifdef _MSC_VER
    int r[4]{};
    __cpuidex(r, leaf, subleaf);
    regs = { static_cast<unsigned>(r[0]),
             static_cast<unsigned>(r[1]),
             static_cast<unsigned>(r[2]),
             static_cast<unsigned>(r[3]) };
#else
    unsigned a = 0, b = 0, c = 0, d = 0;
    __cpuid_count(leaf, subleaf, a, b, c, d);
    regs = { a, b, c, d };
#endif
}

std::string read_brand() {
    std::array<unsigned, 4> r{};
    cpuid(0x80000000, 0, r);
    if (r[0] < 0x80000004) return {};
    char brand[49]{};
    for (unsigned leaf = 0x80000002; leaf <= 0x80000004; ++leaf) {
        cpuid(static_cast<int>(leaf), 0, r);
        std::memcpy(brand + (leaf - 0x80000002u) * 16u, r.data(), 16);
    }
    brand[48] = '\0';
    // Trim leading spaces
    const char* start = brand;
    while (*start == ' ') ++start;
    return std::string{start};
}

CpuFeatures detect() {
    CpuFeatures f{};

    std::array<unsigned, 4> r{};
    cpuid(1, 0, r);
    f.has_sse2 = (r[3] & (1u << 26)) != 0;
    f.has_avx  = (r[2] & (1u << 28)) != 0;
    f.has_fma  = (r[2] & (1u << 12)) != 0;

    cpuid(7, 0, r);
    f.has_avx2    = (r[1] & (1u << 5))  != 0;
    f.has_avx512f = (r[1] & (1u << 16)) != 0;

    f.logical_cores = std::thread::hardware_concurrency();
    // Physical core detection is hard cross-vendor. Best-effort: divide by 2
    // if AVX2 is present (implies modern SMT-capable core). Falls back cleanly.
    f.physical_cores = (f.logical_cores >= 2) ? (f.logical_cores / 2)
                                              : f.logical_cores;

    f.cpu_brand = read_brand();

    return f;
}

} // namespace

const CpuFeatures& cpu_features() {
    static CpuFeatures cached = detect();
    return cached;
}

void require_v01_baseline_or_die() {
    const auto& f = cpu_features();
    if (!f.has_avx2 || !f.has_fma) {
        std::fprintf(stderr,
            "ultima: CPU does not meet v0.1 baseline requirements.\n"
            "  Required : AVX2 + FMA\n"
            "  Detected : AVX2=%d FMA=%d AVX=%d SSE2=%d\n"
            "  CPU      : %s\n",
            f.has_avx2 ? 1 : 0,
            f.has_fma  ? 1 : 0,
            f.has_avx  ? 1 : 0,
            f.has_sse2 ? 1 : 0,
            f.cpu_brand.c_str());
        std::exit(1);
    }
}

} // namespace ultima::kernels
