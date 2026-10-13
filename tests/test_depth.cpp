// test_depth.cpp - the depth decoder, and full token generation, against
// Google's numbers.
//
// Usage: test_depth MODEL.safetensors REF.safetensors
#include "../depth.h"
#include "../encoder.h"
#include "../temporal.h"
#include "test.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

using namespace mrt2;

static const Weights *W = nullptr; // the model
static const Weights *R = nullptr; // the reference dump (50 frames)
static const int FRAMES = 50;

struct Err {
    double err = 0, mag = 0;
    void add(const float *a, const float *b, int n) {
        for (int i = 0; i < n; i++) {
            err = std::max(err, (double)std::fabs(a[i] - b[i]));
            mag = std::max(mag, (double)std::fabs(b[i]));
        }
    }
    double rel() const { return err / (mag + 1e-30); }
};

TEST(depth_matches_google) {
    // Google's T_t in, greedy: compare every step's intermediates.
    const DepthWeights dw = load_depth(*W);
    const float *t = R->f32("temporal/out", {FRAMES, kModel});
    const float *g_adapter = R->f32("depth/adapter", {FRAMES, kLevels, kDepthModel});
    const float *g_ln = R->f32("depth/final_ln", {FRAMES, kLevels, kDepthModel});
    const float *g_logits = R->f32("depth/logits", {FRAMES, kLevels, kCodebook});
    const int32_t *g_tokens = R->i32("tokens", {FRAMES, kLevels});
    DepthState st;
    Rng rng(0);
    const Sampling greedy{0.0f, 40};
    Err e_adapter, e_ln, e_logits;
    int same = 0;
    std::vector<float> adapter(kLevels * kDepthModel), ln(kLevels * kDepthModel), logits(kLevels * kCodebook);
    for (int f = 0; f < FRAMES; f++) {
        DepthDebug dbg;
        dbg.adapter = adapter.data();
        dbg.final_ln = ln.data();
        dbg.logits = logits.data();
        int32_t tok[kLevels];
        depth_frame(dw, st, t + f * kModel, greedy, rng, tok, &dbg);
        e_adapter.add(adapter.data(), g_adapter + f * kLevels * kDepthModel, kLevels * kDepthModel);
        e_ln.add(ln.data(), g_ln + f * kLevels * kDepthModel, kLevels * kDepthModel);
        e_logits.add(logits.data(), g_logits + f * kLevels * kCodebook, kLevels * kCodebook);
        for (int k = 0; k < kLevels; k++) same += tok[k] == g_tokens[f * kLevels + k];
    }
    std::printf("      adapter (12 steps x 50 frames): max error %.2e (relative %.1e)\n", e_adapter.err, e_adapter.rel());
    std::printf("      final LayerNorm:                max error %.2e (relative %.1e)\n", e_ln.err, e_ln.rel());
    std::printf("      logits before soft-cap:         max error %.2e (relative %.1e, values up to %.1f)\n",
                e_logits.err, e_logits.rel(), e_logits.mag);
    std::printf("      tokens identical to Google's:   %d / %d\n", same, FRAMES * kLevels);
    CHECK(e_adapter.rel() < 1e-5);
    CHECK(e_ln.rel() < 1e-5);
    CHECK(e_logits.rel() < 1e-5);
    CHECK(same == FRAMES * kLevels);
}

TEST(generation_matches_google) {
    // The whole model, autoregressively, all our code: each frame's tokens
    // become the next frame's input, exactly as in real generation. One
    // different token and everything after it diverges.
    const EncoderWeights ew = load_encoder(*W);
    const TemporalWeights tw = load_temporal(*W);
    const DepthWeights dw = load_depth(*W);
    const int32_t *cond = R->i32("cond", {FRAMES, kCondTokens});
    const int32_t *g_tokens = R->i32("tokens", {FRAMES, kLevels});
    const float *g_logits = R->f32("depth/logits", {FRAMES, kLevels, kCodebook});
    TemporalState ts;
    DepthState ds;
    Rng rng(0);
    const Sampling greedy{0.0f, 40};
    int32_t prev[kLevels] = {}; // SOS
    int same = 0, first_diff = -1;
    float e[256], x[kModel], t[kModel];
    auto t0 = std::chrono::steady_clock::now();
    for (int f = 0; f < FRAMES; f++) {
        encode(ew, cond + f * kCondTokens, e);
        temporal_input(tw, prev, x);
        temporal_step(tw, ts, x, e, t);
        int32_t tok[kLevels];
        depth_frame(dw, ds, t, greedy, rng, tok);
        for (int k = 0; k < kLevels; k++) {
            const bool ok = tok[k] == g_tokens[f * kLevels + k];
            same += ok;
            if (!ok && first_diff < 0) {
                first_diff = f * kLevels + k;
                // Was it a near-tie? Gap between Google's two best logits.
                const float *l = g_logits + (f * kLevels + k) * kCodebook;
                std::vector<float> s(l, l + kCodebook);
                std::sort(s.begin(), s.end(), std::greater<float>());
                std::printf("      first difference: frame %d level %d (ours %d, Google %d), Google's top-2 gap %.2e\n",
                            f, k, tok[k], g_tokens[f * kLevels + k], s[0] - s[1]);
            }
        }
        std::copy(tok, tok + kLevels, prev);
    }
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::printf("      generated tokens identical to Google's: %d / %d\n", same, FRAMES * kLevels);
    std::printf("      (plain scalar code, 1 thread: %.1f ms per frame, codec not included)\n", ms / FRAMES);
    CHECK(same == FRAMES * kLevels);
}

TEST(sampling_follows_the_rules) {
    // Temperature 1.3, top-k 40 (Google's defaults): every token must be in
    // its level's range and among that level's 40 highest (soft-capped)
    // logits; the same seed must give the same tokens.
    const DepthWeights dw = load_depth(*W);
    const float *t = R->f32("temporal/out", {FRAMES, kModel});
    DepthState st;
    const Sampling s{1.3f, 40};
    int in_range = 0, in_top_k = 0, total = 0;
    std::vector<float> logits(kLevels * kCodebook);
    int32_t a[kLevels], b[kLevels];
    for (int f = 0; f < 10; f++) {
        Rng r1(1000 + f), r2(1000 + f);
        DepthDebug dbg;
        dbg.logits = logits.data();
        depth_frame(dw, st, t + f * kModel, s, r1, a, &dbg);
        depth_frame(dw, st, t + f * kModel, s, r2, b);
        CHECK(std::equal(a, a + kLevels, b));
        for (int k = 0; k < kLevels; k++) {
            const int code = a[k] - kReserved - k * kCodebook;
            in_range += code >= 0 && code < kCodebook;
            float l[kCodebook];
            std::copy(&logits[k * kCodebook], &logits[(k + 1) * kCodebook], l);
            softcap(l, kCodebook, 30.0f);
            std::vector<float> sorted(l, l + kCodebook);
            std::sort(sorted.begin(), sorted.end(), std::greater<float>());
            in_top_k += code >= 0 && code < kCodebook && l[code] >= sorted[s.top_k - 1];
            total++;
        }
    }
    std::printf("      sampled tokens in range: %d / %d, inside the top %d: %d / %d\n", in_range, total, s.top_k,
                in_top_k, total);
    CHECK(in_range == total);
    CHECK(in_top_k == total);
}

TEST(sampling_distribution_matches_formula) {
    // Sampling at temperature 1.3 must draw code i with probability
    // softmax(softcap(logit) / 1.3) over the top 40. Greedy tests can't see
    // the soft-cap (it keeps the order), so check the frequencies directly,
    // on the real level with the largest logits (where the cap matters most).
    const float *g = R->f32("depth/logits", {FRAMES, kLevels, kCodebook});
    int best = 0;
    for (int i = 0; i < FRAMES * kLevels; i++)
        if (*std::max_element(g + i * kCodebook, g + (i + 1) * kCodebook) >
            *std::max_element(g + best * kCodebook, g + (best + 1) * kCodebook)) best = i;
    const float *raw = g + best * kCodebook;
    const Sampling s{1.3f, 40};

    // Expected probabilities, with and without the soft-cap.
    auto expected = [&](bool cap) {
        std::vector<float> l(raw, raw + kCodebook);
        if (cap) softcap(l.data(), kCodebook, 30.0f);
        std::vector<float> sorted = l;
        std::sort(sorted.begin(), sorted.end(), std::greater<float>());
        std::vector<double> p(kCodebook, 0.0);
        double z = 0;
        for (int i = 0; i < kCodebook; i++)
            if (l[i] >= sorted[s.top_k - 1]) z += p[i] = std::exp((l[i] - sorted[0]) / s.temperature);
        for (double &v : p) v /= z;
        return p;
    };
    const std::vector<double> want = expected(true), uncapped = expected(false);

    const int n = 20000;
    std::vector<double> got(kCodebook, 0.0);
    Rng rng(77);
    float l[kCodebook];
    for (int i = 0; i < n; i++) {
        std::copy(raw, raw + kCodebook, l);
        got[sample_code(l, s, rng)] += 1.0 / n;
    }
    // Total variation distance: half the sum of |p - q|; 0 = identical.
    auto tv = [](const std::vector<double> &a, const std::vector<double> &b) {
        double d = 0;
        for (size_t i = 0; i < a.size(); i++) d += std::fabs(a[i] - b[i]);
        return d / 2;
    };
    std::printf("      level with largest logit (%.1f): distance to formula %.3f; "
                "formula with vs without soft-cap differ by %.3f\n",
                *std::max_element(raw, raw + kCodebook), tv(got, want), tv(want, uncapped));
    CHECK(tv(want, uncapped) > 0.1); // the test can tell the two apart
    CHECK(tv(got, want) < 0.02);     // and we match the soft-capped one
}

int main(int argc, char **argv) {
    if (argc < 3) {
        std::printf("usage: %s MODEL.safetensors REF.safetensors\n", argv[0]);
        return 1;
    }
    Weights model, ref;
    std::string err;
    if (!model.open(argv[1], err) || !ref.open(argv[2], err)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }
    W = &model;
    R = &ref;
    try {
        RUN(depth_matches_google);
        RUN(generation_matches_google);
        RUN(sampling_follows_the_rules);
        RUN(sampling_distribution_matches_formula);
    } catch (const std::runtime_error &e) { // a missing/mis-shaped tensor
        std::printf("    FAIL %s\n", e.what());
        test::failures++;
    }
    return test::summary("test_depth");
}
