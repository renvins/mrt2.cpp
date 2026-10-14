// bench.cpp - where does the time go, and what limits each part?
//
// Step 8 of the plan: the starting point of every optimization. It
//   1. measures this machine's limits: peak arithmetic of one core
//      (GFLOP/s) and memory bandwidth (GB/s, 1 thread and all threads);
//   2. runs the real model (style preset, real sampling) and times every
//      operation through the profiler (profile.h), with its FLOPs and bytes;
//   3. prints a "roofline" table: per operation, its share of the time, the
//      FLOP/s and GB/s it achieves, whether arithmetic or memory limits it,
//      and how close it gets to that limit.
//
// Roofline in one paragraph: an operation doing F flops on B bytes has
// "arithmetic intensity" F/B. A core can do at most P flops/s and read at
// most W bytes/s, so the operation can't beat min(P, F/B * W). Below the
// "ridge" P/W it is memory-bound (only reading fewer bytes helps); above,
// compute-bound (SIMD, register blocking, fewer flops help). "% of limit"
// says how much headroom is left before the hardware stops you.
//
// Built with -DMRT2_PROFILE. Usage:  tools/bench MODEL.safetensors [frames] [threads] [int8|fp32]
#include "../codec.h"
#include "../cpu.h"
#include "../depth.h"
#include "../encoder.h"
#include "../profile.h"
#include "../temporal.h"
#include "../threads.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif
#if defined(__APPLE__)
#include <pthread/qos.h>
#endif

using namespace mrt2;
using Clock = std::chrono::steady_clock;

static double secs(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

// ---- machine limits ------------------------------------------------------------

// Peak single-core arithmetic: 16 independent vector multiply-adds per
// iteration. A multiply-add takes ~4 cycles before its result can be
// reused and the M4 core has 4 FMA units, so 16 independent chains keep
// them all busy. Written with NEON intrinsics: plain C++ loops are not
// vectorized well enough by the compiler to measure the peak (17 GFLOP/s
// instead of ~110). The empty asm statement hides the values from the
// optimizer, so it can't simplify the loop away.
#if defined(MRT2_X86)
// The x86 version, with AVX2: 12 independent chains of 8-wide FMAs. An
// x86 core has 2 FMA units with a 4-5 cycle latency, so it needs about 10
// chains in flight to keep both busy.
MRT2_AVX2 static double peak_gflops_avx2() {
    volatile float va = 0.9999999f, vb = 1e-7f; // not compile-time constants
    const long iters = 40'000'000;
    volatile float sink = 0;
    __m256 acc[12];
    for (int i = 0; i < 12; i++) acc[i] = _mm256_set1_ps(1.0f + i);
    const __m256 a = _mm256_set1_ps(va), b = _mm256_set1_ps(vb);
    auto t0 = Clock::now();
    for (long it = 0; it < iters; it++)
        for (int i = 0; i < 12; i++) acc[i] = _mm256_fmadd_ps(acc[i], a, b);
    __m256 t = acc[0];
    for (int i = 1; i < 12; i++) t = _mm256_add_ps(t, acc[i]);
    float lanes[8];
    _mm256_storeu_ps(lanes, t);
    sink = lanes[0] + lanes[7]; // stored before the clock is read again
    const double s = secs(t0, Clock::now());
    (void)sink;
    return 2.0 * 8 * 12 * iters / s / 1e9;
}
#endif

static double peak_gflops() {
#if defined(MRT2_X86)
    if (cpu_avx2()) return peak_gflops_avx2();
#endif
    volatile float va = 0.9999999f, vb = 1e-7f; // not compile-time constants
    const long iters = 40'000'000;
    volatile float sink = 0;
#if defined(__ARM_NEON)
    float32x4_t acc[16];
    for (int i = 0; i < 16; i++) acc[i] = vdupq_n_f32(1.0f + i);
    const float32x4_t a = vdupq_n_f32(va), b = vdupq_n_f32(vb);
    auto t0 = Clock::now();
    for (long it = 0; it < iters; it++) {
        for (int i = 0; i < 16; i++) acc[i] = vfmaq_f32(b, acc[i], a);
        __asm__ volatile("" : "+w"(acc[0]), "+w"(acc[1]), "+w"(acc[2]), "+w"(acc[3]));
    }
    float32x4_t t = acc[0];
    for (int i = 1; i < 16; i++) t = vaddq_f32(t, acc[i]);
    sink = vaddvq_f32(t); // stored before the clock is read again
#else
    // No NEON: a portable estimate (an AVX2 version comes with x86 support).
    float acc[64];
    for (int i = 0; i < 64; i++) acc[i] = 1.0f + i * 1e-3f;
    const float a = va, b = vb;
    auto t0 = Clock::now();
    for (long it = 0; it < iters; it++)
        for (int i = 0; i < 64; i++) acc[i] = acc[i] * a + b;
    float total = 0;
    for (float v : acc) total += v;
    sink = total;
#endif
    const double s = secs(t0, Clock::now());
    (void)sink;
    return 2.0 * 64 * iters / s / 1e9;
}

// Read bandwidth: `threads` threads summing disjoint parts of 1 GB.
static double bandwidth_gbs(int threads) {
    const size_t n = (size_t)1 << 28; // 1 GB of floats
    static std::vector<float> big(n, 1.0f);
    double best = 0;
    for (int r = 0; r < 3; r++) {
        std::atomic<float> sink{0};
        auto t0 = Clock::now();
        std::vector<std::thread> pool;
        for (int t = 0; t < threads; t++)
            pool.emplace_back([&, t] {
                float s[8] = {};
                for (size_t i = n * t / threads; i + 8 <= n * (t + 1) / threads; i += 8)
                    for (int j = 0; j < 8; j++) s[j] += big[i + j];
                float tot = 0;
                for (float v : s) tot += v;
                sink.fetch_add(tot);
            });
        for (auto &th : pool) th.join();
        best = std::max(best, n * 4.0 / secs(t0, Clock::now()) / 1e9);
    }
    return best;
}

// ---- the run --------------------------------------------------------------------

int main(int argc, char **argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s mrt2_small.safetensors [frames] [threads] [int8|fp32]\n", argv[0]);
        return 1;
    }
    const int frames = argc >= 3 ? std::atoi(argv[2]) : 50;
    const int hw0 = (int)std::thread::hardware_concurrency();
    const int threads = argc >= 4 ? std::atoi(argv[3]) : hw0;
    const bool int8 = argc < 5 || std::string(argv[4]) != "fp32";
    // Apple chips mix performance and efficiency cores, and macOS may run a
    // thread started from a script on an efficiency core (~3x slower),
    // making results swing between runs. Ask for the performance cores, as
    // the engine itself does.
    prefer_performance_cores();

    std::printf("=== machine: %s, %d hardware threads%s ===\n", cpu_name().empty() ? "unknown CPU" : cpu_name().c_str(),
                hw0, cpu_avx2() ? ", AVX2 kernels" : "");
    std::printf("=== machine limits ===\n");
    // Best of 5: the first runs can catch the core still raising its clock
    // speed (or briefly on an efficiency core); a limit is the best it can do.
    double P = 0;
    for (int r = 0; r < 5; r++) P = std::max(P, peak_gflops());
    // All cores at once: every thread runs the peak loop, and we add them up.
    double PN = 0;
    for (int r = 0; r < 3; r++) {
        std::vector<double> got(threads);
        std::vector<std::thread> ts;
        for (int i = 0; i < threads; i++)
            ts.emplace_back([&, i] { prefer_performance_cores(); got[i] = peak_gflops(); });
        for (auto &th : ts) th.join();
        double sum = 0;
        for (double g : got) sum += g;
        PN = std::max(PN, sum);
    }
    const double W1 = bandwidth_gbs(1), WN = bandwidth_gbs(threads);
    std::printf("peak arithmetic: 1 core %6.1f GFLOP/s | %d threads %6.1f GFLOP/s\n", P, threads, PN);
    std::printf("memory bandwidth: 1 thread %5.1f GB/s | %d threads %5.1f GB/s\n", W1, threads, WN);
    set_threads(threads);
    // The limits that apply to this run: one core, or all the threads used.
    const double Plim = threads > 1 ? PN : P, Wlim = threads > 1 ? WN : W1;
    std::printf("ridge point: %.1f flops per byte; below it an operation is memory-bound\n", Plim / Wlim);

    Weights w;
    std::string err;
    if (!w.open(argv[1], err)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }
    const EncoderWeights ew = load_encoder(w);
    TemporalWeights tw = load_temporal(w);
    DepthWeights dw = load_depth(w);
    const CodecWeights cw = load_codec(w);
    if (int8) {
        make_int8(tw);
        make_int8(dw);
    }

    // Jazz preset, Google's default sampling: a realistic run.
    Conditioning cond;
    const int jazz[12] = {555, 402, 72, 424, 482, 932, 954, 153, 33, 374, 519, 779};
    for (int k = 0; k < 12; k++) cond.style[k] = jazz[k];
    int32_t ctok[kCondTokens];
    encoder_tokens(cond, ctok, nullptr);

    // Warm-up frame (pages the weights in from disk), not measured.
    {
        TemporalState ts;
        DepthState ds;
        Rng rng(1);
        int32_t prev[kLevels] = {};
        float e[256], x[kModel], t[kModel];
        encode(ew, ctok, e);
        temporal_input(tw, prev, x);
        temporal_step(tw, ts, x, e, t);
        depth_frame(dw, ds, t, Sampling{}, rng, prev);
        prof::reset();
    }

    // The engine's loop: tokens, then the streaming codec, per frame.
    TemporalState ts;
    DepthState ds;
    CodecStream cs;
    Rng rng(0);
    std::vector<float> frame_ms(frames);
    int32_t prev[kLevels] = {};
    float e[256], x[kModel], t[kModel], z[256], audio[2 * kSamplesPerFrame];
    double tok = 0, codec = 0;
    auto t0 = Clock::now();
    for (int f = 0; f < frames; f++) {
        auto a = Clock::now();
        encode(ew, ctok, e);
        temporal_input(tw, prev, x);
        temporal_step(tw, ts, x, e, t);
        depth_frame(dw, ds, t, Sampling{}, rng, prev);
        auto b = Clock::now();
        dequantize(cw, prev, z);
        decode_frame(cw, cs, z, audio);
        auto c = Clock::now();
        tok += secs(a, b);
        codec += secs(b, c);
        frame_ms[f] = (float)(secs(a, c) * 1e3);
    }
    auto t2 = Clock::now();
    const double tok_ms = tok * 1e3 / frames, codec_ms = codec * 1e3 / frames;

    std::vector<float> sorted = frame_ms;
    std::sort(sorted.begin(), sorted.end());
    std::printf("\n=== the run: %d frames (%.1f s of audio), %d threads, %s transformer weights ===\n", frames,
                frames * 0.04, threads, int8 ? "int8" : "float32");
    std::printf("tokens (encoder+temporal+depth): %6.1f ms/frame\n", tok_ms);
    std::printf("codec (streaming):               %6.1f ms/frame\n", codec_ms);
    std::printf("total: %.1f ms per 40 ms frame -> %.2fx real time  (frame p50 %.1f, p99 %.1f, max %.1f ms)\n",
                tok_ms + codec_ms, 40.0 / (tok_ms + codec_ms), sorted[frames / 2],
                sorted[std::min(frames - 1, frames * 99 / 100)], sorted.back());

    // ---- the roofline table ----
    std::vector<prof::Entry> es = prof::entries();
    std::sort(es.begin(), es.end(), [](const prof::Entry &a, const prof::Entry &b) { return a.seconds > b.seconds; });
    double total = 0;
    for (const auto &e2 : es) total += e2.seconds;
    std::printf("\n=== where the time goes (sorted; limits are for the %d threads used) ===\n", threads);
    std::printf("%-8s %-10s %8s %6s %6s %8s %7s %7s %-7s %7s\n", "stage", "op", "ms/frame", "share", "cum.",
                "GFLOP/s", "GB/s", "flop/B", "limit", "% limit");
    double cum = 0;
    for (const auto &e2 : es) {
        cum += e2.seconds;
        const double gf = e2.flops / e2.seconds / 1e9, gb = e2.bytes / e2.seconds / 1e9;
        const double ai = e2.flops / e2.bytes;
        const bool mem = ai < Plim / Wlim;
        const double roof = mem ? ai * Wlim : Plim;
        std::printf("%-8s %-10s %8.2f %5.1f%% %5.1f%% %8.1f %7.1f %7.2f %-7s %6.0f%%\n", e2.stage, e2.op,
                    e2.seconds * 1e3 / frames, 100 * e2.seconds / total, 100 * cum / total, gf, gb, ai,
                    mem ? "memory" : "compute", 100 * gf / roof);
    }
    const double wall = secs(t0, t2);
    std::printf("profiled operations cover %.1f%% of the wall time\n", 100 * total / wall);
    return 0;
}
