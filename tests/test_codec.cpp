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
#include <cstring>
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

TEST(streaming_equals_offline) {
    // The streaming decoder, one frame at a time with its remembered conv
    // rows, against the offline decoder on the whole clip: same samples.
    // Google's z for 50 frames (2 s), so this also checks against Google.
    const CodecWeights cw = load_codec(*W);
    const float *z = R->f32("codec/z", {FRAMES, 256});
    const int32_t *g_audio = R->i32("audio/int16", {FRAMES, kSamplesPerFrame, 2});
    const std::vector<float> offline = inverse_stft(decode(cw, z, FRAMES));
    std::vector<float> stream((size_t)FRAMES * kSamplesPerFrame * 2);
    CodecStream st;
    for (int f = 0; f < FRAMES; f++) decode_frame(cw, st, z + f * 256, &stream[(size_t)f * kSamplesPerFrame * 2]);

    double maxdiff = 0;
    int vs_google = 0;
    for (size_t i = 0; i < stream.size(); i++) {
        maxdiff = std::max(maxdiff, (double)std::fabs(stream[i] - offline[i]));
        vs_google = std::max(vs_google, std::abs((int)to_int16(stream[i]) - g_audio[i]));
    }
    const bool same = std::memcmp(stream.data(), offline.data(), stream.size() * 4) == 0;
    std::printf("      %d frames streamed: %s offline (max difference %.3g); vs Google max |int16 diff| %d\n",
                FRAMES, same ? "bit-identical to" : "NOT identical to", maxdiff, vs_google);
    CHECK(same);
    CHECK(vs_google <= 1);
}

TEST(istft_fft_matches_direct_sum) {
    // inverse_stft uses an FFT; here is the same thing written the slow,
    // obvious way (the formula, summed directly), on a random spectrum.
    const int steps = 8; // 2 frames: the first frame is silent (lookahead)
    Tensor spec(steps, 480, 4);
    Rng rng(42);
    for (float &v : spec.v) v = (float)(2.0 * rng.uniform() - 1.0);
    const std::vector<float> got = inverse_stft(spec);

    const double pi = 3.14159265358979323846;
    const int N = 960, H = 480;
    std::vector<double> win(N), frames((size_t)steps * 2 * N, 0.0);
    for (int n = 0; n < N; n++) {
        auto hann = [&](int i) { return 0.5 - 0.5 * std::cos(2 * pi * i / N); };
        const int m = n % H;
        win[n] = hann(n) / (hann(m) * hann(m) + hann(m + H) * hann(m + H));
    }
    for (int t = 4; t < steps; t++)
        for (int ch = 0; ch < 2; ch++)
            for (int n = 0; n < N; n++) {
                double x = spec.at(t, 0)[2 * ch]; // bin 0: real part only; bin 480: 0
                for (int k = 1; k < 480; k++)
                    x += 2.0 * (spec.at(t, k)[2 * ch] * std::cos(2 * pi * k * n / N) -
                                spec.at(t, k)[2 * ch + 1] * std::sin(2 * pi * k * n / N));
                frames[((size_t)t * 2 + ch) * N + n] = x / N * win[n];
            }
    double err = 0, mag = 0;
    for (int t = 0; t < steps; t++)
        for (int ch = 0; ch < 2; ch++)
            for (int j = 0; j < H; j++) {
                const double want = frames[((size_t)t * 2 + ch) * N + j] +
                                    (t > 0 ? frames[((size_t)(t - 1) * 2 + ch) * N + H + j] : 0.0);
                err = std::max(err, std::fabs(got[((size_t)t * H + j) * 2 + ch] - want));
                mag = std::max(mag, std::fabs(want));
            }
    std::printf("      FFT vs direct sum, random spectrum: max difference %.2e (values up to %.3f)\n", err, mag);
    CHECK(mag > 0.01);         // the test really produced sound
    CHECK(err <= 1e-6 * mag);  // only float rounding of the result
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
        RUN(istft_fft_matches_direct_sum);
        RUN(dequantize_matches_google);
        RUN(decoder_blocks_match_google);
        RUN(audio_from_our_tokens_matches_google);
        RUN(streaming_equals_offline);
    } catch (const std::runtime_error &e) { // a missing/mis-shaped tensor
        std::printf("    FAIL %s\n", e.what());
        test::failures++;
    }
    return test::summary("test_codec");
}
