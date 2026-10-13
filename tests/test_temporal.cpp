// test_temporal.cpp - the temporal decoder against Google's numbers.
//
// Usage: test_temporal MODEL.safetensors REF.safetensors
#include "../encoder.h"
#include "../temporal.h"
#include "test.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

using namespace mrt2;

static const Weights *W = nullptr; // the model
static const Weights *R = nullptr; // the reference dump (50 frames)
static const int FRAMES = 50;

// Largest |a - b| and largest |b| over n floats, accumulated across calls.
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

// The previous frame's tokens (twelve 0s = SOS before the first frame).
static void prev_tokens(const int32_t *tokens, int f, int32_t prev[kLevels]) {
    for (int k = 0; k < kLevels; k++) prev[k] = f == 0 ? 0 : tokens[(f - 1) * kLevels + k];
}

TEST(input_matches_google) {
    const TemporalWeights tw = load_temporal(*W);
    const int32_t *tokens = R->i32("tokens", {FRAMES, kLevels});
    const float *want = R->f32("temporal/in", {FRAMES, kModel});
    Err e;
    float x[kModel];
    for (int f = 0; f < FRAMES; f++) {
        int32_t prev[kLevels];
        prev_tokens(tokens, f, prev);
        temporal_input(tw, prev, x);
        e.add(x, want + f * kModel, kModel);
    }
    std::printf("      temporal input vs Google, 50 frames: max error %.2e (values up to %.2f)\n", e.err, e.mag);
    CHECK(e.err < 1e-5);
}

TEST(step_matches_google) {
    // Feed Google's own inputs (temporal/in, enc/out), so this tests the
    // temporal decoder alone.
    const TemporalWeights tw = load_temporal(*W);
    const float *xin = R->f32("temporal/in", {FRAMES, kModel});
    const float *enc = R->f32("enc/out", {FRAMES, 256});
    const float *g_self = R->f32("temporal/l0_self", {FRAMES, kModel});
    const float *g_cross = R->f32("temporal/l0_cross", {FRAMES, kModel});
    const float *g_layers = R->f32("temporal/layers", {FRAMES, kTemporalLayers, kModel});
    const float *g_out = R->f32("temporal/out", {FRAMES, kModel});

    TemporalState st;
    Err e_self, e_cross, e_layer[kTemporalLayers], e_out, e_early, e_late;
    std::vector<float> layers(kTemporalLayers * kModel);
    float self[kModel], cross[kModel], out[kModel];
    for (int f = 0; f < FRAMES; f++) {
        TemporalDebug dbg;
        dbg.l0_self = self;
        dbg.l0_cross = cross;
        dbg.layers = layers.data();
        temporal_step(tw, st, xin + f * kModel, enc + f * 256, out, &dbg);
        e_self.add(self, g_self + f * kModel, kModel);
        e_cross.add(cross, g_cross + f * kModel, kModel);
        for (int l = 0; l < kTemporalLayers; l++)
            e_layer[l].add(&layers[l * kModel], g_layers + (f * kTemporalLayers + l) * kModel, kModel);
        e_out.add(out, g_out + f * kModel, kModel);
        // Frames 0..41 fill the ring buffer; from frame 42 on, each new
        // frame overwrites the oldest slot.
        (f < kSlots ? e_early : e_late).add(out, g_out + f * kModel, kModel);
    }
    std::printf("      layer 0 after self-attention:  max error %.2e (relative %.1e)\n", e_self.err, e_self.rel());
    std::printf("      layer 0 after cross-attention: max error %.2e (relative %.1e)\n", e_cross.err, e_cross.rel());
    for (int l = 0; l < kTemporalLayers; l++)
        std::printf("      layer %2d output:              max error %.2e (relative %.1e, values up to %.1f)\n", l,
                    e_layer[l].err, e_layer[l].rel(), e_layer[l].mag);
    std::printf("      output, frames 0-41 (ring filling):  max error %.2e\n", e_early.err);
    std::printf("      output, frames 42-49 (ring sliding): max error %.2e\n", e_late.err);
    CHECK(e_self.rel() < 1e-5);
    CHECK(e_cross.rel() < 1e-5);
    for (int l = 0; l < kTemporalLayers; l++) CHECK(e_layer[l].rel() < 1e-5);
    CHECK(e_out.rel() < 1e-5);
}

TEST(full_chain_matches_google) {
    // Our encoder + our temporal input + our temporal step, nothing from
    // Google except the conditioning tokens and the previous frame's tokens.
    const EncoderWeights ew = load_encoder(*W);
    const TemporalWeights tw = load_temporal(*W);
    const int32_t *cond = R->i32("cond", {FRAMES, kCondTokens});
    const int32_t *tokens = R->i32("tokens", {FRAMES, kLevels});
    const float *g_out = R->f32("temporal/out", {FRAMES, kModel});
    TemporalState st;
    Err e;
    float enc[256], x[kModel], out[kModel];
    for (int f = 0; f < FRAMES; f++) {
        encode(ew, cond + f * kCondTokens, enc);
        int32_t prev[kLevels];
        prev_tokens(tokens, f, prev);
        temporal_input(tw, prev, x);
        temporal_step(tw, st, x, enc, out);
        e.add(out, g_out + f * kModel, kModel);
    }
    std::printf("      encoder -> temporal, all ours, 50 frames: max error %.2e (relative %.1e)\n", e.err, e.rel());
    CHECK(e.rel() < 1e-5);
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
        RUN(input_matches_google);
        RUN(step_matches_google);
        RUN(full_chain_matches_google);
    } catch (const std::runtime_error &e) { // a missing/mis-shaped tensor
        std::printf("    FAIL %s\n", e.what());
        test::failures++;
    }
    return test::summary("test_temporal");
}
