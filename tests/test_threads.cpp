// test_threads.cpp - the thread pool, and the promise that the music does
// not depend on the number of threads.
//
// Usage: test_threads MODEL.safetensors [REF.safetensors, ignored]
#include "../codec.h"
#include "../depth.h"
#include "../encoder.h"
#include "../temporal.h"
#include "../threads.h"
#include "test.h"
#include <atomic>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace mrt2;

static const Weights *W = nullptr;

TEST(every_chunk_runs_exactly_once) {
    // Many chunk counts on pools of several sizes, 5000 runs in total: a
    // missed or doubled chunk, or a hang, would show up here.
    const int sizes[] = {1, 2, 4, (int)std::thread::hardware_concurrency()};
    bool all_ok = true;
    long runs = 0;
    for (int n : sizes) {
        ThreadPool p(n);
        for (int rep = 0; rep < 1250; rep++) {
            const int chunks = 1 + rep % 97;
            std::vector<std::atomic<int>> hits(chunks);
            for (auto &h : hits) h.store(0);
            p.run(chunks, [&](int i) { hits[i].fetch_add(1); });
            for (auto &h : hits) all_ok &= h.load() == 1;
            runs++;
        }
    }
    std::printf("      %ld runs on pools of 1, 2, 4 and %u threads: every chunk exactly once: %s\n", runs,
                std::thread::hardware_concurrency(), all_ok ? "yes" : "NO");
    CHECK(all_ok);
}

TEST(nested_run_executes_inline) {
    // A run() started inside a chunk must not deadlock waiting for workers
    // that are busy with the outer run: it executes inline.
    ThreadPool p((int)std::thread::hardware_concurrency());
    std::atomic<int> inner{0};
    p.run(8, [&](int) { p.run(4, [&](int) { inner.fetch_add(1); }); });
    CHECK(inner.load() == 32);
}

// 10 frames of tokens and their audio, on the global pool.
struct Result {
    std::vector<int32_t> tokens;
    std::vector<float> audio;
};

static Result generate(float temperature, bool int8) {
    const EncoderWeights ew = load_encoder(*W);
    TemporalWeights tw = load_temporal(*W);
    DepthWeights dw = load_depth(*W);
    const CodecWeights cw = load_codec(*W);
    if (int8) {
        make_int8(tw);
        make_int8(dw);
    }
    Conditioning cond;
    const int jazz[12] = {555, 402, 72, 424, 482, 932, 954, 153, 33, 374, 519, 779};
    for (int k = 0; k < 12; k++) cond.style[k] = jazz[k];
    int32_t ctok[kCondTokens];
    encoder_tokens(cond, ctok, nullptr);
    TemporalState ts;
    DepthState ds;
    Rng rng(123);
    const Sampling s{temperature, 40};
    const int frames = 10;
    Result r;
    std::vector<float> z((size_t)frames * 256);
    int32_t prev[kLevels] = {};
    float e[256], x[kModel], t[kModel];
    encode(ew, ctok, e);
    for (int f = 0; f < frames; f++) {
        temporal_input(tw, prev, x);
        temporal_step(tw, ts, x, e, t);
        depth_frame(dw, ds, t, s, rng, prev);
        r.tokens.insert(r.tokens.end(), prev, prev + kLevels);
        dequantize(cw, prev, &z[(size_t)f * 256]);
    }
    r.audio = inverse_stft(decode(cw, z.data(), frames));
    return r;
}

TEST(same_music_on_any_thread_count) {
    // Bit for bit: 1 thread vs all threads, greedy and with real sampling,
    // with float32 and with int8 transformer weights.
    const int hw = (int)std::thread::hardware_concurrency();
    for (bool int8 : {false, true})
    for (float temperature : {0.0f, 1.3f}) {
        set_threads(1);
        const Result one = generate(temperature, int8);
        set_threads(hw);
        const Result many = generate(temperature, int8);
        const bool same_tokens = one.tokens == many.tokens;
        const bool same_audio = one.audio.size() == many.audio.size() &&
                                std::memcmp(one.audio.data(), many.audio.data(), one.audio.size() * 4) == 0;
        std::printf("      %s, temperature %.1f, 1 vs %d threads: tokens %s, audio %s\n",
                    int8 ? "int8   " : "float32", temperature, hw,
                    same_tokens ? "identical" : "DIFFERENT", same_audio ? "bit-identical" : "DIFFERENT");
        CHECK(same_tokens);
        CHECK(same_audio);
    }
}

int main(int argc, char **argv) {
    RUN(every_chunk_runs_exactly_once);
    RUN(nested_run_executes_inline);
    if (argc < 2) {
        std::printf("(model tests skipped: pass MODEL path)\n");
        return test::summary("test_threads");
    }
    Weights model;
    std::string err;
    if (!model.open(argv[1], err)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }
    W = &model;
    try {
        RUN(same_music_on_any_thread_count);
    } catch (const std::runtime_error &e) {
        std::printf("    FAIL %s\n", e.what());
        test::failures++;
    }
    return test::summary("test_threads");
}
