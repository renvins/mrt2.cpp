// test_conv.cpp - the fast conv kernels against the plain reference loops,
// on random data in shapes chosen to hit every edge case: odd channel
// counts, a single frequency bin, partial blocks of 4, every stride the
// codec uses, and the few-output-channels path. No model file needed.
#include "../conv.h"
#include "../ops.h"
#include "test.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

using namespace mrt2;

struct Case {
    bool transposed;
    int kt, kf, cin, cout, t, f;
    int a, b; // conv: freq pads (pl, pr); transposed: strides (st, sf)
};

// Random weights/bias owned by the test (Conv only points at them).
struct Owned {
    std::vector<float> k, b;
    Conv c;
};

static Owned make_conv(const Case &cs, Rng &rng) {
    Owned o;
    o.k.resize((size_t)cs.kt * cs.kf * cs.cin * cs.cout);
    o.b.resize(cs.cout);
    for (float &v : o.k) v = (float)rng.uniform() - 0.5f;
    for (float &v : o.b) v = (float)rng.uniform() - 0.5f;
    o.c.k = o.k.data();
    o.c.b = o.b.data();
    o.c.kt = cs.kt;
    o.c.kf = cs.kf;
    o.c.cin = cs.cin;
    o.c.cout = cs.cout;
    o.c.pack();
    return o;
}

TEST(fast_matches_reference) {
    const Case cases[] = {
        // regular convs: kt, kf, cin, cout, t, f, pl, pr
        {false, 1, 1, 256, 2560, 3, 1, 0, 0},  // the input layer: 1 freq bin
        {false, 3, 3, 64, 64, 5, 7, 1, 1},
        {false, 3, 3, 32, 48, 4, 10, 1, 1},    // 3 tiles of 16
        {false, 3, 3, 17, 16, 2, 3, 1, 1},     // odd cin, partial block
        {false, 1, 1, 33, 32, 1, 5, 0, 0},
        {false, 3, 3, 16, 16, 3, 1, 1, 1},     // 3x3 on a single bin: most taps are padding
        {false, 7, 7, 64, 2, 4, 9, 3, 3},      // the output layer: 2 channels
        // transposed: kt, kf, cin, cout, t, f, st, sf
        {true, 4, 3, 32, 64, 3, 5, 2, 1},      // like decoder_0
        {true, 4, 4, 48, 32, 2, 5, 2, 2},      // like decoder_1
        {true, 3, 4, 32, 32, 4, 10, 1, 2},     // like decoder_2,3,5,6
        {true, 3, 6, 32, 16, 4, 7, 1, 3},      // like decoder_4
        {true, 3, 4, 16, 16, 1, 3, 1, 2},
        {true, 3, 4, 16, 2, 2, 3, 1, 2},       // few-channels path with a stride
        {true, 4, 4, 16, 16, 1, 1, 2, 2},      // a single input position
    };
    Rng rng(5);
    int ok = 0, n = 0;
    double worst = 0;
    for (const Case &cs : cases) {
        Owned o = make_conv(cs, rng);
        Tensor x(cs.t, cs.f, cs.cin);
        for (float &v : x.v) v = (float)rng.uniform() - 0.5f;
        const Tensor want = cs.transposed ? conv_transpose2d_ref(x, o.c, cs.a, cs.b) : conv2d_ref(x, o.c, cs.a, cs.b);
        const Tensor got = cs.transposed ? conv_transpose2d(x, o.c, cs.a, cs.b) : conv2d(x, o.c, cs.a, cs.b);
        double err = 0, mag = 0;
        bool same_shape = got.t == want.t && got.f == want.f && got.c == want.c;
        for (size_t i = 0; same_shape && i < want.v.size(); i++) {
            err = std::max(err, (double)std::fabs(got.v[i] - want.v[i]));
            mag = std::max(mag, (double)std::fabs(want.v[i]));
        }
        const double rel = err / (mag + 1e-30);
        worst = std::max(worst, rel);
        const bool pass = same_shape && rel < 1e-5;
        if (!pass)
            std::printf("    case %d (%s %dx%d, %d->%d, %dx%d): %s, relative error %.1e\n", n,
                        cs.transposed ? "convT" : "conv", cs.kt, cs.kf, cs.cin, cs.cout, cs.t, cs.f,
                        same_shape ? "values differ" : "shape differs", rel);
        ok += pass;
        n++;
    }
    std::printf("      %d / %d shapes match the reference (worst relative error %.1e)\n", ok, n, worst);
    CHECK(ok == n);
}

TEST(winograd_matches_reference) {
    // Winograd F(2x2, 3x3) against the plain loops: odd and single
    // frequency counts (half-empty last tile), a streaming call (skip = 2),
    // and codec-sized values (inputs up to millions).
    struct W { int cin, cout, t, f, skip; float scale; };
    const W cases[] = {
        {16, 16, 4, 1, 0, 1.0f},   // one frequency bin
        {32, 48, 6, 5, 0, 1.0f},   // odd bins: the last tile has one column
        {64, 64, 8, 20, 0, 1.0f},
        {16, 32, 6, 7, 2, 1.0f},   // streaming: 2 remembered rows + 4 new
        {64, 64, 6, 10, 2, 1e6f},  // codec-sized values
    };
    Rng rng(11);
    int ok = 0;
    double worst = 0;
    bool stream_same = true;
    for (const W &w : cases) {
        Owned o = make_conv({false, 3, 3, w.cin, w.cout, w.t, w.f, 1, 1}, rng);
        Conv cw = o.c; // same weights, plus their Winograd form
        cw.pack_winograd();
        Tensor x(w.t, w.f, w.cin);
        for (float &v : x.v) v = w.scale * ((float)rng.uniform() - 0.5f);
        const Tensor want = conv2d_ref(x, o.c, 1, 1); // all rows
        const Tensor got = conv2d(x, cw, 1, 1, "conv", w.skip);
        double err = 0, mag = 0;
        const bool shape = got.t == w.t - w.skip && got.f == w.f && got.c == w.cout;
        for (int t = 0; shape && t < got.t; t++)
            for (int f = 0; f < w.f; f++)
                for (int c = 0; c < w.cout; c++) {
                    err = std::max(err, (double)std::fabs(got.at(t, f)[c] - want.at(t + w.skip, f)[c]));
                    mag = std::max(mag, (double)std::fabs(want.at(t + w.skip, f)[c]));
                }
        const double rel = err / (mag + 1e-30);
        worst = std::max(worst, rel);
        ok += shape && rel < 1e-5;
        if (w.skip) { // the same rows from one call over everything: same bits
            const Tensor full = conv2d(x, cw, 1, 1, "conv", 0);
            stream_same &= std::memcmp(full.at(w.skip, 0), got.v.data(), got.v.size() * 4) == 0;
        }
    }
    std::printf("      %d / 5 Winograd shapes match the reference (worst relative error %.1e);"
                " streaming call = full call: %s\n", ok, worst, stream_same ? "bit for bit" : "NO");
    CHECK(ok == 5);
    CHECK(stream_same);
}

int main() {
    RUN(fast_matches_reference);
    RUN(winograd_matches_reference);
    return test::summary("test_conv");
}
