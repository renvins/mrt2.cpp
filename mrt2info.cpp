// mrt2info.cpp - list the tensors in a .safetensors file, and optionally
// check that every weight is a real number.
//
//   mrt2info [--verify] model.safetensors [name-substring]
//
// Listing reads only the header, so it is fast even on multi-gigabyte
// files. --verify reads every F32 value once and counts NaN/Inf: a corrupt
// download or broken checkpoint shows up here instead of as bad audio.
#include "weights.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

// Number of NaN/Inf values among the n floats at p.
static uint64_t count_non_finite(const float *p, uint64_t n) {
    uint64_t bad = 0;
    for (uint64_t i = 0; i < n; i++) bad += !std::isfinite(p[i]);
    return bad;
}

int main(int argc, char **argv) {
    // Flags first, then the file, then an optional name filter.
    bool verify = false;
    int a = 1;
    if (a < argc && std::strcmp(argv[a], "--verify") == 0) { verify = true; a++; }
    if (a >= argc) {
        std::fprintf(stderr, "usage: %s [--verify] model.safetensors [name-substring]\n", argv[0]);
        return 1;
    }
    const char *path = argv[a];
    // Optional filter: only tensors whose name contains this substring.
    const char *filter = a + 1 < argc ? argv[a + 1] : nullptr;

    mrt2::Weights w;
    std::string err;
    if (!w.open(path, err)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }
    const mrt2::SafeTensorsHeader &st = w.header();

    uint64_t total_params = 0;
    uint64_t total_bytes = 0;
    size_t shown = 0;
    uint64_t checked_values = 0, non_finite = 0;
    size_t checked = 0, skipped = 0, bad_tensors = 0;
    auto t0 = std::chrono::steady_clock::now();

    for (const auto &kv : st.tensors) {
        const mrt2::TensorInfo &t = kv.second;
        total_params += t.numel();
        total_bytes += t.nbytes;
        if (filter && t.name.find(filter) == std::string::npos) continue;
        shown++;

        if (!verify) {
            std::printf("%-6s [", t.dtype.c_str());
            for (size_t i = 0; i < t.shape.size(); i++)
                std::printf("%s%lld", i ? ", " : "", (long long)t.shape[i]);
            std::printf("]  %s\n", t.name.c_str());
            continue;
        }

        // Only F32 is checked; other dtypes would need their own decoder.
        if (t.dtype != "F32") { skipped++; continue; }
        const float *p = reinterpret_cast<const float *>(w.raw(t));
        uint64_t bad = count_non_finite(p, t.numel());
        checked++;
        checked_values += t.numel();
        if (bad) {
            bad_tensors++;
            non_finite += bad;
            std::printf("BAD  %llu non-finite values in %s\n",
                        (unsigned long long)bad, t.name.c_str());
        }
    }

    std::printf("tensors=%zu  params=%llu  bytes=%llu  data_start=%llu\n",
                st.tensors.size(),
                (unsigned long long)total_params,
                (unsigned long long)total_bytes,
                (unsigned long long)st.data_start);
    if (filter) std::printf("shown=%zu for \"%s\"\n", shown, filter);

    if (verify) {
        double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("verify: %zu tensors, %llu values, %llu non-finite in %zu tensors, %zu skipped (not F32), %.2f s\n",
                    checked, (unsigned long long)checked_values,
                    (unsigned long long)non_finite, bad_tensors, skipped, secs);
        if (non_finite) return 1;
    }
    return 0;
}
