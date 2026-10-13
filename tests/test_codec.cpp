// test_codec.cpp - the SpectroStream decoder against Google's numbers, and
// the whole model end to end: our tokens -> our audio vs Google's audio.
//
// Usage: test_codec MODEL.safetensors REF.safetensors
#include "../codec.h"
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
static const Weights *R = nullptr; // the reference dump
static const int FRAMES = 50;      // tokens and audio are saved for 50 frames
static const int CODEC_FRAMES = 10; // codec blocks only for the first 10

struct Err {
    double err = 0, mag = 0;
    void add(double a, double b) {
        err = std::max(err, std::fabs(a - b));
        mag = std::max(mag, std::fabs(b));
    }
    double rel() const { return err / (mag + 1e-30); }
};

TEST(dequantize_matches_google) {
    const CodecWeights cw = load_codec(*W);
    const int32_t *tokens = R->i32("tokens", {FRAMES, 12});
    const float *want = R->f32("codec/z", {FRAMES, 256});
    Err e;
    float z[256];
    for (int f = 0; f < FRAMES; f++) {
        dequantize(cw, tokens + f * 12, z);
        for (int i = 0; i < 256; i++) e.add(z[i], want[f * 256 + i]);
    }
    std::printf("      z vs Google, 50 frames: max error %.2e\n", e.err);
    CHECK(e.err < 1e-5);
}

TEST(decoder_blocks_match_google) {
    // Google saved each block's output per streaming frame; ours runs over
    // the whole clip at once. A frame's s-th time step is our time
    // frame * steps + s, so the two line up element by element.
    const CodecWeights cw = load_codec(*W);
    const float *z = R->f32("codec/z", {FRAMES, 256});
    CodecDebug dbg;
    decode(cw, z, CODEC_FRAMES, &dbg);

    auto check_block = [&](const char *name, const Tensor &ours, const float *ref, int steps) {
        Err e;
        // ref layout: [frame][step][freq][channel]; ours: [frame*steps + step][freq][channel]
        for (size_t i = 0; i < ours.v.size(); i++) e.add(ours.v[i], ref[i]);
        (void)steps;
        std::printf("      %-18s max error %.2e (relative %.1e, values up to %.0f)\n", name, e.err, e.rel(), e.mag);
        CHECK(e.rel() < 1e-5);
    };
    check_block("input layer", dbg.input_layer, R->f32("codec/input_layer", {CODEC_FRAMES, 1, 1, 2560}), 1);
    check_block("input residual", dbg.input_res, R->f32("codec/input_res", {CODEC_FRAMES, 1, 5, 512}), 1);
    check_block("decoder_0", dbg.decoder0, R->f32("codec/decoder_0", {CODEC_FRAMES, 2, 5, 1024}), 2);

    // decoder_1..6: ref [frame][group][4 steps][F][C], ours per group [frame*4 + step][F][C].
    const int F[6] = {10, 20, 40, 120, 240, 480}, C[6] = {256, 256, 256, 128, 128, 64};
    for (int b = 0; b < 6; b++) {
        const std::string name = "codec/decoder_" + std::to_string(b + 1);
        const float *ref = R->f32(name, {CODEC_FRAMES, 2, 4, F[b], C[b]});
        const size_t step = (size_t)F[b] * C[b];
        for (int g = 0; g < 2; g++) {
            Err e;
            const Tensor &ours = dbg.block[b][g];
            for (int f = 0; f < CODEC_FRAMES; f++)
                for (int s = 0; s < 4; s++)
                    for (size_t i = 0; i < step; i++)
                        e.add(ours.v[(size_t)(f * 4 + s) * step + i], ref[(((size_t)f * 2 + g) * 4 + s) * step + i]);
            std::printf("      decoder_%d %-8s max error %.2e (relative %.1e, values up to %.0f)\n", b + 1,
                        g ? "right" : "left", e.err, e.rel(), e.mag);
            CHECK(e.rel() < 1e-5);
        }
    }
    check_block("spectrum", dbg.spectrum, R->f32("codec/spectrum", {CODEC_FRAMES, 4, 480, 4}), 4);
}

TEST(audio_from_our_tokens_matches_google) {
    // Milestone 1, checked: the whole model in C++. Our encoder, temporal
    // and depth decoders generate the tokens (greedy, feeding each frame
    // back in), our codec turns them into audio, and the result must be
    // Google's audio, sample by sample.
    const EncoderWeights ew = load_encoder(*W);
    const TemporalWeights tw = load_temporal(*W);
    const DepthWeights dw = load_depth(*W);
    const CodecWeights cw = load_codec(*W);
    const int32_t *cond = R->i32("cond", {FRAMES, kCondTokens});
    const int32_t *g_audio = R->i32("audio/int16", {FRAMES, kSamplesPerFrame, 2});

    auto t0 = std::chrono::steady_clock::now();
    TemporalState ts;
    DepthState ds;
    Rng rng(0);
    const Sampling greedy{0.0f, 40};
    std::vector<float> z((size_t)FRAMES * 256);
    int32_t prev[kLevels] = {}; // SOS
    float e[256], x[kModel], t[kModel];
    for (int f = 0; f < FRAMES; f++) {
        encode(ew, cond + f * kCondTokens, e);
        temporal_input(tw, prev, x);
        temporal_step(tw, ts, x, e, t);
        depth_frame(dw, ds, t, greedy, rng, prev); // this frame's tokens become next frame's input
        dequantize(cw, prev, &z[(size_t)f * 256]);
    }
    auto t1 = std::chrono::steady_clock::now();
    const std::vector<float> audio = inverse_stft(decode(cw, z.data(), FRAMES));
    auto t2 = std::chrono::steady_clock::now();

    int maxdiff = 0, off_by_one = 0, worse = 0;
    for (size_t i = 0; i < audio.size(); i++) {
        const int d = std::abs((int)to_int16(audio[i]) - g_audio[i]);
        maxdiff = std::max(maxdiff, d);
        off_by_one += d == 1;
        worse += d > 1;
    }
    auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    std::printf("      %zu samples (2 s stereo): max |int16 difference| %d, off by 1: %d, worse: %d\n",
                audio.size(), maxdiff, off_by_one, worse);
    std::printf("      plain code, 1 thread, per frame: tokens %.1f ms + codec %.1f ms\n", ms(t0, t1) / FRAMES,
                ms(t1, t2) / FRAMES);
    CHECK(worse == 0);                       // never more than rounding
    CHECK(off_by_one < (int)audio.size() / 100); // and rounding differences rare
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
        RUN(dequantize_matches_google);
        RUN(decoder_blocks_match_google);
        RUN(audio_from_our_tokens_matches_google);
    } catch (const std::runtime_error &e) { // a missing/mis-shaped tensor
        std::printf("    FAIL %s\n", e.what());
        test::failures++;
    }
    return test::summary("test_codec");
}
