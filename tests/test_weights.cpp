// test_weights.cpp - open a model and read one tensor through the mapping.
//
// The expected numbers come from Python's safetensors + numpy reading the
// same file, so a match proves we read the real weights, with zero copies.
#include "../weights.h"
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <type_traits>
#include <utility>

static int failures = 0;

static void check(bool ok, const char *what) {
    std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) failures++;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s mrt2_small.safetensors\n", argv[0]);
        return 1;
    }

    mrt2::Weights w;
    std::string err;
    if (!w.open(argv[1], err)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }

    const char *name = "params/depthformer/decoder/decoder_embedding/embedding/embedding";
    const float *emb = nullptr;
    try {
        emb = w.f32(name, {12294, 1024});
    } catch (const std::runtime_error &e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }

    // Row-major: element [r][c] lives at emb[r * 1024 + c].
    const float row0[4] = {0.08007417f, -0.0020667f, 0.06358631f, 0.00326423f};
    const float row1[4] = {0.02102244f, 0.00825383f, -0.02933007f, -0.02572102f};
    bool rows_ok = true;
    for (int i = 0; i < 4; i++) {
        rows_ok &= std::fabs(emb[i] - row0[i]) < 1e-7f;
        rows_ok &= std::fabs(emb[1024 + i] - row1[i]) < 1e-7f;
    }
    check(rows_ok, "rows 0 and 1 match Python");

    // Touching all 12.6M floats pulls ~50 MB of pages in from disk.
    double sum = 0;
    for (size_t i = 0; i < (size_t)12294 * 1024; i++) sum += emb[i];
    std::printf("      sum = %.6f (Python: 267.479301)\n", sum);
    check(std::fabs(sum - 267.4793005513534) < 1e-4, "sum matches Python");

    // Lookup of a name that isn't there.
    check(w.find("no/such/tensor") == nullptr, "find() returns nullptr for a missing name");

    // Each wrong request must throw, never hand back a bad pointer.
    auto throws = [&](const char *n, std::initializer_list<int64_t> s) {
        try { w.f32(n, s); } catch (const std::runtime_error &) { return true; }
        return false;
    };
    check(throws(name, {1024, 12294}), "wrong shape is rejected");
    check(throws(name, {12294}), "wrong rank is rejected");
    check(throws("no/such/tensor", {1}), "missing tensor is rejected");

    // Weights is movable because its members are: moving it keeps every
    // pointer we got from f32() valid, since the mapping does not move.
    static_assert(!std::is_copy_constructible_v<mrt2::Weights>);
    static_assert(std::is_move_constructible_v<mrt2::Weights>);
    mrt2::Weights moved(std::move(w));
    check(moved.find(name) != nullptr && emb[0] == row0[0],
          "after moving Weights, lookups and old pointers still work");

    std::printf("%s\n", failures ? "SOME TESTS FAILED" : "all tests passed");
    return failures ? 1 : 0;
}
