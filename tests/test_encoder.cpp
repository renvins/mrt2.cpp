// test_encoder.cpp - the conditioning encoder against Google's numbers.
//
// Usage: test_encoder MODEL.safetensors REF.safetensors
#include "../encoder.h"
#include "test.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>

using namespace mrt2;

static const Weights *W = nullptr; // the model
static const Weights *R = nullptr; // the reference dump (50 frames)
static const int FRAMES = 50;
// "jazz piano trio", the style the reference was made with.
static const int JAZZ[12] = {555, 402, 72, 424, 482, 932, 954, 153, 33, 374, 519, 779};

TEST(discretize_cfg_matches_google) {
    // Values and results from Google's discretize_cfg (Python). The .5
    // cases (0.1 -> 5.5, 0.3 -> 6.5, ...) check round-half-to-even.
    const double v[16] = {-2, -1, -0.9, 0, 0.1, 0.3, 0.5, 0.7, 1, 1.5, 2.9, 3, 3.1, 6.9, 7, 8};
    const int bins02[16] = {0, 0, 0, 5, 6, 6, 8, 8, 10, 12, 20, 20, 20, 40, 40, 40};
    const int bins10[16] = {0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 4, 4, 4, 8, 8, 8};
    int ok02 = 0, ok10 = 0;
    for (int i = 0; i < 16; i++) {
        ok02 += discretize_cfg(v[i], 0.2, 40) == bins02[i];
        ok10 += discretize_cfg(v[i], 1.0, 8) == bins10[i];
    }
    CHECK(ok02 == 16);
    CHECK(ok10 == 16);
    // Non-finite input behaves like Google's Python: NaN -> max bin,
    // +inf -> max bin, -inf -> 0 (and never undefined behaviour).
    CHECK(discretize_cfg(std::nan(""), 0.2, 40) == 40);
    CHECK(discretize_cfg(std::nan(""), 1.0, 8) == 8);
    CHECK(discretize_cfg(INFINITY, 0.2, 40) == 40);
    CHECK(discretize_cfg(-INFINITY, 0.2, 40) == 0);
}

TEST(tokens_match_google) {
    // The reference used: jazz style, no notes, no drums, default CFG.
    Conditioning c;
    std::copy(JAZZ, JAZZ + 12, c.style);
    int32_t tok[kCondTokens];
    const char *err = nullptr;
    CHECK(encoder_tokens(c, tok, &err));
    const int32_t *want = R->i32("cond", {FRAMES, kCondTokens});
    int same = 0;
    for (int f = 0; f < FRAMES; f++)
        same += std::equal(tok, tok + kCondTokens, want + f * kCondTokens);
    std::printf("      tokens identical to Google's in %d / %d frames\n", same, FRAMES);
    CHECK(same == FRAMES);
}

TEST(tokens_reject_bad_values) {
    int32_t tok[kCondTokens];
    const char *err = nullptr;
    Conditioning a; a.style[3] = 1024;  CHECK(!encoder_tokens(a, tok, &err));
    Conditioning b; b.pitch[60] = 4;    CHECK(!encoder_tokens(b, tok, &err));
    Conditioning c; c.pitch[0] = -2;    CHECK(!encoder_tokens(c, tok, &err));
    Conditioning d; d.drums = 2;        CHECK(!encoder_tokens(d, tok, &err));
    Conditioning e; e.pitch[127] = 3; e.drums = 1; e.style[0] = 1023;
    CHECK(encoder_tokens(e, tok, &err)); // the largest valid values pass
}

// Max |a - b| over n floats.
static double maxdiff(const float *a, const float *b, int n) {
    double m = 0;
    for (int i = 0; i < n; i++) m = std::max(m, (double)std::fabs(a[i] - b[i]));
    return m;
}

TEST(encode_matches_google) {
    const EncoderWeights e = load_encoder(*W);
    const int32_t *cond = R->i32("cond", {FRAMES, kCondTokens});
    const float *g_reg = R->f32("enc/regular", {FRAMES, kEncDim});
    const float *g_comb = R->f32("enc/combined", {FRAMES, kEncDim});
    const float *g_out = R->f32("enc/out", {FRAMES, kEncDim});
    float out[kEncDim], reg[kEncDim], comb[kEncDim];
    double e_reg = 0, e_comb = 0, e_out = 0;
    for (int f = 0; f < FRAMES; f++) {
        EncoderDebug dbg;
        dbg.regular = reg;
        dbg.combined = comb;
        encode(e, cond + f * kCondTokens, out, &dbg);
        e_reg = std::max(e_reg, maxdiff(reg, g_reg + f * kEncDim, kEncDim));
        e_comb = std::max(e_comb, maxdiff(comb, g_comb + f * kEncDim, kEncDim));
        e_out = std::max(e_out, maxdiff(out, g_out + f * kEncDim, kEncDim));
    }
    std::printf("      vs Google, 50 frames: regular %.2e, combined %.2e, output %.2e\n",
                e_reg, e_comb, e_out);
    CHECK(e_reg < 1e-5);
    CHECK(e_comb < 1e-5);
    CHECK(e_out < 1e-5);
}

int main(int argc, char **argv) {
    RUN(discretize_cfg_matches_google);
    RUN(tokens_reject_bad_values);
    if (argc < 3) {
        std::printf("(real-data tests skipped: pass MODEL and REF paths)\n");
        return test::summary("test_encoder");
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
        RUN(tokens_match_google);
        RUN(encode_matches_google);
    } catch (const std::runtime_error &e) { // a missing/mis-shaped tensor
        std::printf("    FAIL %s\n", e.what());
        test::failures++;
    }
    return test::summary("test_encoder");
}
