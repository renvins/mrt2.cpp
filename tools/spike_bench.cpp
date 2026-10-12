// spike_bench.cpp - can a CPU run MRT2 in real time? Measure before step 8.
//
// One frame = 40 ms of audio, so all the work of a frame must finish in
// < 40 ms. This times both halves of that work at their exact per-frame
// sizes, with two implementations:
//
//   naive  - plain loops, roughly what Phase A writes first,
//   kernel - a register-blocked micro-kernel on packed weights (ARM NEON
//            intrinsics when available, plain C++ otherwise): the core of
//            what Phase B will write,
//   each on 1 thread or on a small persistent thread pool.
//
//  1. SpectroStream decoder (~84% of the arithmetic): every conv layer at
//     its real shape (channels, kernel, stride, frequency bins, the two
//     stereo groups), with random data: speed depends on sizes, not values.
//     naive and kernel are checked to give the same output.
//  2. Transformer: the real matrices of one frame (12 temporal layers +
//     12 depth steps) with our ops::vecmat/matvec on Google's real weights,
//     on 1 thread and on the pool.
//
// A measurement tool, not part of the engine. Usage:
//   tools/spike_bench MODEL.safetensors
#include "../ops.h"
#include "../weights.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <thread>
#include <vector>
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

using namespace mrt2;
using Clock = std::chrono::steady_clock;

static double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// ---------------------------------------------------------------------------
// A minimal persistent thread pool. Threads are created once and wait for
// work; run(n, fn) calls fn(0..n-1) spread over all threads. Chunks are
// handed out one at a time from a shared counter, so faster cores (the
// M4's performance cores) simply take more chunks than slower ones.
// ---------------------------------------------------------------------------
class Pool {
public:
    explicit Pool(int threads) : n_(threads) {
        for (int i = 1; i < n_; i++)
            workers_.emplace_back([this] {
                int seen = 0;
                for (;;) {
                    while (gen_.load(std::memory_order_acquire) == seen) std::this_thread::yield();
                    seen = gen_.load(std::memory_order_acquire);
                    if (quit_.load()) return;
                    work();
                    done_.fetch_add(1, std::memory_order_release);
                }
            });
    }
    ~Pool() {
        quit_ = true;
        gen_.fetch_add(1, std::memory_order_release);
        for (auto &t : workers_) t.join();
    }
    int size() const { return n_; }
    void run(int chunks, const std::function<void(int)> &fn) {
        if (n_ == 1 || chunks == 1) { for (int i = 0; i < chunks; i++) fn(i); return; }
        fn_ = &fn;
        chunks_ = chunks;
        next_.store(0);
        done_.store(0);
        gen_.fetch_add(1, std::memory_order_release); // wake the workers
        work();                                        // the caller helps too
        while (done_.load(std::memory_order_acquire) != n_ - 1) std::this_thread::yield();
    }

private:
    void work() {
        for (int i; (i = next_.fetch_add(1)) < chunks_;) (*fn_)(i);
    }
    int n_;
    std::vector<std::thread> workers_;
    std::atomic<int> gen_{0}, done_{0}, next_{0};
    std::atomic<bool> quit_{false};
    const std::function<void(int)> *fn_ = nullptr;
    int chunks_ = 0;
};

// ---------------------------------------------------------------------------
// Convolution over (time, freq, channel) activations, channels last, like
// JAX. Weights [KT][KF][Cin][Cout], the checkpoint's layout.
//
// Conv and transposed conv are both written in "gather" form: every output
// position sums over the kernel taps that land on it. A transposed conv
// with stride s then visits only the taps of its phase (out % s): exactly
// the real work, with no multiplies by the zeros that "insert zeros, then
// convolve" would waste.
//
// Inputs carry a zero margin, so every tap reads valid memory and loops
// need no boundary checks; the cost equals the real layer's.
// ---------------------------------------------------------------------------

struct Layer {
    const char *name;
    bool transposed;
    int cin, cout, kt, kf, st, sf; // channels, kernel (time, freq), stride
    int tin, fin;                  // new input positions per frame
    int groups;                    // 2 = runs once per stereo channel
    int tout() const { return transposed ? tin * st : tin; }
    int fout() const { return transposed ? fin * sf : fin; }
    double macs() const { // multiply-adds per frame, all groups
        const int taps = transposed ? (kt / st) * (kf / sf) : kt * kf;
        return (double)groups * tout() * fout() * taps * cin * cout;
    }
};

// Per-frame shapes of the decoder, from the verified config (plan 2.6).
static const Layer kCodec[] = {
    // shared part (both stereo channels together)
    {"input 1x1 a",     false, 256, 2560, 1, 1, 1, 1, 1, 1, 1},
    {"input 1x1 b1",    false, 256, 2560, 1, 1, 1, 1, 1, 1, 1},
    {"input 1x1 b2",    false, 2560, 2560, 1, 1, 1, 1, 1, 1, 1},
    {"input_res 3x3 a", false, 512, 512, 3, 3, 1, 1, 1, 5, 1},
    {"input_res 3x3",   false, 512, 512, 3, 3, 1, 1, 1, 5, 1},
    {"d0 convT 4x3",    true, 512, 1024, 4, 3, 2, 1, 1, 5, 1},
    {"d0 3x3",          false, 1024, 1024, 3, 3, 1, 1, 2, 5, 1},
    {"d0 shortcut",     false, 512, 1024, 1, 1, 1, 1, 1, 5, 1},
    // per stereo channel (groups = 2, same weights)
    {"d1 convT 4x4",    true, 512, 256, 4, 4, 2, 2, 2, 5, 2},
    {"d1 3x3",          false, 256, 256, 3, 3, 1, 1, 4, 10, 2},
    {"d1 shortcut",     false, 512, 256, 1, 1, 1, 1, 2, 5, 2},
    {"d2 convT 3x4",    true, 256, 256, 3, 4, 1, 2, 4, 10, 2},
    {"d2 3x3",          false, 256, 256, 3, 3, 1, 1, 4, 20, 2},
    {"d3 convT 3x4",    true, 256, 256, 3, 4, 1, 2, 4, 20, 2},
    {"d3 3x3",          false, 256, 256, 3, 3, 1, 1, 4, 40, 2},
    {"d4 convT 3x6",    true, 256, 128, 3, 6, 1, 3, 4, 40, 2},
    {"d4 3x3",          false, 128, 128, 3, 3, 1, 1, 4, 120, 2},
    {"d4 shortcut",     false, 256, 128, 1, 1, 1, 1, 4, 40, 2},
    {"d5 convT 3x4",    true, 128, 128, 3, 4, 1, 2, 4, 120, 2},
    {"d5 3x3",          false, 128, 128, 3, 3, 1, 1, 4, 240, 2},
    {"d6 convT 3x4",    true, 128, 64, 3, 4, 1, 2, 4, 240, 2},
    {"d6 3x3",          false, 64, 64, 3, 3, 1, 1, 4, 480, 2},
    {"d6 shortcut",     false, 128, 64, 1, 1, 1, 1, 4, 240, 2},
    {"output 7x7",      false, 64, 2, 7, 7, 1, 1, 4, 480, 2},
};

constexpr int P = 4;   // output positions per kernel call
constexpr int CT = 16; // output channels per kernel call

struct Buffers {
    int fi;                         // padded input width (freq)
    std::vector<float> in, w, packed, out;
};

static Buffers make_buffers(const Layer &L, Rng &rng) {
    Buffers b;
    const int ti = L.tin + L.kt;
    b.fi = L.fin + 2 * L.kf;
    b.in.resize((size_t)ti * b.fi * L.cin);
    b.w.resize((size_t)L.kt * L.kf * L.cin * L.cout);
    b.out.resize((size_t)L.tout() * L.fout() * L.cout);
    for (float &v : b.in) v = (float)rng.uniform() - 0.5f;
    for (float &v : b.w) v = ((float)rng.uniform() - 0.5f) * 0.05f;
    // Packed weights [Cout/CT][KT*KF][Cin][CT]: the CT columns one kernel
    // call needs become contiguous instead of a whole row apart. A real
    // engine does this once, at load time. (Cout not a multiple of CT:
    // the kernel falls back to naive, nothing to pack.)
    if (L.cout % CT == 0) {
        b.packed.resize(b.w.size());
        size_t i = 0;
        for (int c0 = 0; c0 < L.cout; c0 += CT)
            for (int k = 0; k < L.kt * L.kf; k++)
                for (int ci = 0; ci < L.cin; ci++)
                    for (int j = 0; j < CT; j++)
                        b.packed[i++] = b.w[((size_t)k * L.cin + ci) * L.cout + c0 + j];
    }
    return b;
}

static inline const float *in_at(const Layer &L, const Buffers &b, int ti, int fi) {
    return b.in.data() + ((size_t)ti * b.fi + fi) * L.cin;
}

// For output (to, fo) and tap (kt, kf): the input position it reads, or
// false if this tap does not land on this output (transposed-conv phase).
static inline bool tap_source(const Layer &L, int to, int fo, int kt, int kf, int &ti, int &fi) {
    if (!L.transposed) {
        ti = to + kt;
        fi = fo + kf;
        return true;
    }
    const int a = to + L.kt - 1 - kt, c = fo + L.kf - 1 - kf; // shifted to stay >= 0
    if (a % L.st || c % L.sf) return false;
    ti = a / L.st;
    fi = c / L.sf;
    return true;
}

// A piece of one layer's output: freq [f0, f1), channels [c0, c1).
struct Chunk { int f0, f1, c0, c1; };

// Naive: one output position at a time; the innermost loop runs over the
// output channels (contiguous in both W and out), so the compiler
// vectorizes it, but every weight loaded feeds only one multiply-add.
static void conv_naive(const Layer &L, Buffers &b, Chunk k) {
    for (int to = 0; to < L.tout(); to++)
        for (int fo = k.f0; fo < k.f1; fo++) {
            float *y = b.out.data() + ((size_t)to * L.fout() + fo) * L.cout;
            std::fill(y + k.c0, y + k.c1, 0.0f);
            for (int kt = 0; kt < L.kt; kt++)
                for (int kf = 0; kf < L.kf; kf++) {
                    int ti, fi;
                    if (!tap_source(L, to, fo, kt, kf, ti, fi)) continue;
                    const float *x = in_at(L, b, ti, fi);
                    const float *w = b.w.data() + (size_t)(kt * L.kf + kf) * L.cin * L.cout;
                    for (int ci = 0; ci < L.cin; ci++) {
                        const float xv = x[ci];
                        const float *wr = w + (size_t)ci * L.cout;
                        for (int co = k.c0; co < k.c1; co++) y[co] += xv * wr[co];
                    }
                }
        }
}

// Micro-kernel: acc[p][0..15] += sum_ci x[p][ci] * w[ci][0..15] for P = 4
// positions. The 4 x 16 = 64 running sums stay in 16 vector registers for
// the whole loop, and each weight vector loaded feeds 4 fused multiply-adds.
static inline void micro_4x16(const float *const x[P], const float *w, int cin, float acc[P][CT]) {
#if defined(__ARM_NEON)
    float32x4_t a[P][4];
    for (int p = 0; p < P; p++)
        for (int j = 0; j < 4; j++) a[p][j] = vld1q_f32(acc[p] + 4 * j);
    for (int ci = 0; ci < cin; ci++, w += CT) {
        const float32x4_t w0 = vld1q_f32(w), w1 = vld1q_f32(w + 4);
        const float32x4_t w2 = vld1q_f32(w + 8), w3 = vld1q_f32(w + 12);
        for (int p = 0; p < P; p++) {
            const float xv = x[p][ci];
            a[p][0] = vfmaq_n_f32(a[p][0], w0, xv);
            a[p][1] = vfmaq_n_f32(a[p][1], w1, xv);
            a[p][2] = vfmaq_n_f32(a[p][2], w2, xv);
            a[p][3] = vfmaq_n_f32(a[p][3], w3, xv);
        }
    }
    for (int p = 0; p < P; p++)
        for (int j = 0; j < 4; j++) vst1q_f32(acc[p] + 4 * j, a[p][j]);
#else
    for (int ci = 0; ci < cin; ci++, w += CT)
        for (int p = 0; p < P; p++) {
            const float xv = x[p][ci];
            for (int j = 0; j < CT; j++) acc[p][j] += xv * w[j];
        }
#endif
}

// Kernel version: blocks of P same-phase output positions x CT channels.
static void conv_kernel(const Layer &L, Buffers &b, Chunk k) {
    if (b.packed.empty()) { conv_naive(L, b, k); return; } // e.g. Cout = 2
    const int sf = L.transposed ? L.sf : 1;
    const size_t tap_stride = (size_t)L.cin * CT;
    for (int to = 0; to < L.tout(); to++)
        for (int phase = 0; phase < sf; phase++) {
            const int first = k.f0 + ((phase - k.f0) % sf + sf) % sf;
            for (int fo0 = first; fo0 < k.f1; fo0 += P * sf) {
                const int np = std::min(P, (k.f1 - fo0 + sf - 1) / sf);
                for (int c0 = k.c0; c0 < k.c1; c0 += CT) {
                    float acc[P][CT] = {};
                    const float *wt = b.packed.data() + (size_t)(c0 / CT) * L.kt * L.kf * tap_stride;
                    for (int kt = 0; kt < L.kt; kt++)
                        for (int kf = 0; kf < L.kf; kf++) {
                            int ti, fi;
                            if (!tap_source(L, to, fo0, kt, kf, ti, fi)) continue;
                            const float *x[P];
                            for (int p = 0; p < P; p++) { // past the row end: repeat the last
                                tap_source(L, to, fo0 + std::min(p, np - 1) * sf, kt, kf, ti, fi);
                                x[p] = in_at(L, b, ti, fi);
                            }
                            micro_4x16(x, wt + (size_t)(kt * L.kf + kf) * tap_stride, L.cin, acc);
                        }
                    for (int p = 0; p < np; p++) {
                        float *y = b.out.data() + ((size_t)to * L.fout() + fo0 + p * sf) * L.cout + c0;
                        std::copy(acc[p], acc[p] + CT, y);
                    }
                }
            }
        }
}

// Split a layer into chunks: along frequency when there are many bins,
// along output channels when there are few (the first layers have 1-10
// bins, so a frequency split alone would leave most threads idle).
static std::vector<Chunk> chunks_for(const Layer &L, int threads) {
    const int F = L.fout(), want = threads * 4;
    if (threads == 1) return {{0, F, 0, L.cout}};
    std::vector<Chunk> c;
    if (F >= want * P || L.cout % CT) { // few channels (output layer: 2): split frequency
        for (int i = 0; i < want; i++) c.push_back({F * i / want, F * (i + 1) / want, 0, L.cout});
    } else {
        const int tiles = L.cout / CT, n = std::min(want, tiles);
        for (int i = 0; i < n; i++) c.push_back({0, F, tiles * i / n * CT, tiles * (i + 1) / n * CT});
    }
    return c;
}

template <class Fn>
static double time_layer(const Layer &L, Buffers &b, Pool &pool, Fn fn, int reps) {
    const std::vector<Chunk> chunks = chunks_for(L, pool.size());
    const std::function<void(int)> job = [&](int i) { fn(L, b, chunks[i]); };
    double best = 1e9;
    for (int r = 0; r < reps; r++) {
        auto t0 = Clock::now();
        for (int g = 0; g < L.groups; g++) pool.run((int)chunks.size(), job);
        best = std::min(best, ms_since(t0));
    }
    return best;
}

static double bench_codec(Pool &p1, Pool &p4, Pool &pall) {
    const std::string all = "pool-" + std::to_string(pall.size());
    std::printf("\n=== 1. SpectroStream decoder, ms per frame ===\n");
    std::printf("%-16s %7s | %7s %7s | %7s %7s %7s\n", "layer", "MMAC", "naive", "naive", "kernel", "kernel",
                "kernel");
    std::printf("%-16s %7s | %7s %7s | %7s %7s %7s\n", "", "", "1T", "pool-4", "1T", "pool-4", all.c_str());
    double mac = 0, t[5] = {0, 0, 0, 0, 0}, worst = 0;
    Rng rng(1);
    for (const Layer &L : kCodec) {
        Buffers b = make_buffers(L, rng);
        // Correctness first: both versions must compute the same output.
        conv_naive(L, b, {0, L.fout(), 0, L.cout});
        const std::vector<float> ref = b.out;
        std::fill(b.out.begin(), b.out.end(), 0.0f);
        for (const Chunk &k : chunks_for(L, p4.size())) conv_kernel(L, b, k);
        double err = 0, mag = 0;
        for (size_t i = 0; i < ref.size(); i++) {
            err = std::max(err, (double)std::fabs(ref[i] - b.out[i]));
            mag = std::max(mag, (double)std::fabs(ref[i]));
        }
        worst = std::max(worst, err / (mag + 1e-30));

        const int reps = L.macs() > 5e7 ? 3 : 10;
        const double r[5] = {time_layer(L, b, p1, conv_naive, reps), time_layer(L, b, p4, conv_naive, reps),
                             time_layer(L, b, p1, conv_kernel, reps), time_layer(L, b, p4, conv_kernel, reps),
                             time_layer(L, b, pall, conv_kernel, reps)};
        std::printf("%-16s %7.1f | %7.2f %7.2f | %7.2f %7.2f %7.2f\n", L.name, L.macs() / 1e6, r[0], r[1], r[2],
                    r[3], r[4]);
        mac += L.macs();
        for (int i = 0; i < 5; i++) t[i] += r[i];
    }
    std::printf("%-16s %7.0f | %7.1f %7.1f | %7.1f %7.1f %7.1f\n", "TOTAL ms", mac / 1e6, t[0], t[1], t[2], t[3],
                t[4]);
    std::printf("%-16s %7s | %7.0f %7.0f | %7.0f %7.0f %7.0f\n", "GFLOP/s", "", 2 * mac / t[0] / 1e6,
                2 * mac / t[1] / 1e6, 2 * mac / t[2] / 1e6, 2 * mac / t[3] / 1e6, 2 * mac / t[4] / 1e6);
    std::printf("naive vs kernel outputs: max relative difference %.1e (must be tiny)\n", worst);
    return std::min(t[3], t[4]);
}

// ---------------------------------------------------------------------------
// Transformer: the real matrices of one frame, read from the mapped model.
// ---------------------------------------------------------------------------

// One matrix product: vecmat (W is [in][ld]) or matvec (W is [out][in]).
// Pointers are resolved once, before timing, as a real engine does.
struct Job {
    const float *W, *b;
    size_t in, ld, cols;
    bool out_in; // true: matvec layout
};

static std::vector<Job> frame_jobs(const Weights &w) {
    const std::string T = "params/depthformer/decoder/temporal_body/transformer/x_layers_";
    const std::string D = "params/depthformer/decoder/depth_body/";
    std::vector<Job> jobs;
    auto vm = [&](const std::string &n, std::initializer_list<int64_t> shape, size_t in, size_t cols,
                  const std::string &bias = "", size_t col0 = 0, size_t ld = 0) {
        const size_t L = ld ? ld : cols;
        const float *b = bias.empty() ? nullptr : w.f32(bias, {(int64_t)L}) + col0;
        jobs.push_back({w.f32(n, shape) + col0, b, in, L, cols, false});
    };
    auto mv = [&](const std::string &n, std::initializer_list<int64_t> shape, size_t rows, size_t in) {
        jobs.push_back({w.f32(n, shape), nullptr, in, in, rows, true});
    };
    // Temporal decoder: 12 layers, each self-attention q,k,v,o + cross-
    // attention q,o and k,v of the new 256-d encoder frame + FFN.
    for (int l = 0; l < 12; l++) {
        const std::string p = T + std::to_string(l) + "/";
        for (const char *n : {"query_projection", "key_projection", "value_projection"})
            vm(p + "self_attention/attention/" + n + "/kernel", {1024, 8, 128}, 1024, 1024);
        mv(p + "self_attention/output_projection/kernel", {1024, 8, 128}, 1024, 1024);
        vm(p + "cross_attention/attention/query_projection/kernel", {1024, 8, 128}, 1024, 1024);
        vm(p + "cross_attention/attention/key_projection/kernel", {256, 8, 128}, 256, 1024);
        vm(p + "cross_attention/attention/value_projection/kernel", {256, 8, 128}, 256, 1024);
        mv(p + "cross_attention/output_projection/kernel", {1024, 8, 128}, 1024, 1024);
        vm(p + "ffn/ffn_layer1/kernel", {1024, 4096}, 1024, 4096, p + "ffn/ffn_layer1/bias");
        vm(p + "ffn/ffn_layer2/kernel", {4096, 1024}, 4096, 1024, p + "ffn/ffn_layer2/bias");
    }
    // Depth decoder: 12 sequential steps, each the adapter + 2 layers +
    // this level's 1024-column slice of the 12294-column output layer.
    for (int k = 0; k < 12; k++) {
        vm(D + "depth_input_adapter/kernel", {1024, 768}, 1024, 768);
        for (int l = 0; l < 2; l++) {
            const std::string p = D + "transformer/x_layers_" + std::to_string(l) + "/";
            for (const char *n : {"query_projection", "key_projection", "value_projection"})
                vm(p + "self_attention/attention/" + n + "/kernel", {768, 6, 128}, 768, 768);
            mv(p + "self_attention/output_projection/kernel", {768, 6, 128}, 768, 768);
            vm(p + "ffn/ffn_layer1/kernel", {768, 3072}, 768, 3072, p + "ffn/ffn_layer1/bias");
            vm(p + "ffn/ffn_layer2/kernel", {3072, 768}, 3072, 768, p + "ffn/ffn_layer2/bias");
        }
        vm(D + "to_logits/kernel", {768, 12294}, 768, 1024, D + "to_logits/bias", 6 + 1024 * k, 12294);
    }
    return jobs;
}

static double bench_transformer(const Weights &w, Pool &p1, Pool &p4, Pool &pall) {
    const std::vector<Job> jobs = frame_jobs(w);
    double bytes = 0, macs = 0;
    for (const Job &j : jobs) { bytes += 4.0 * j.in * j.cols; macs += (double)j.in * j.cols; }
    std::vector<float> x(4096, 0.01f), y(4096);

    // One frame on a pool: every product is split by output columns (rows
    // for matvec) into chunks; vecmat's `ld` argument makes a column slice
    // a plain call on W + first_column.
    auto frame = [&](Pool &pool) {
        const int n = pool.size() == 1 ? 1 : pool.size() * 2;
        for (const Job &j : jobs) {
            const std::function<void(int)> part = [&](int i) {
                const size_t a = j.cols * i / n, e = j.cols * (i + 1) / n;
                if (j.out_in) matvec(j.W + a * j.in, e - a, j.in, x.data(), y.data() + a);
                else vecmat(x.data(), j.in, j.W + a, j.ld, e - a, j.b ? j.b + a : nullptr, y.data() + a);
            };
            pool.run(n, part);
        }
    };
    std::printf("\n=== 2. Transformer (temporal + 12 depth steps), real weights, fp32 ===\n");
    std::printf("weights as stored in the checkpoint ([in][out]), split by output columns:\n");
    std::printf("%zu matrix products, %.0f MMAC, %.0f MB of weights read per frame\n", jobs.size(), macs / 1e6,
                bytes / 1e6);
    frame(p1); // warm-up: pages the weights in from disk
    double best_all = 1e9;
    for (Pool *pool : {&p1, &p4, &pall}) {
        double best = 1e9;
        for (int r = 0; r < 5; r++) {
            auto t0 = Clock::now();
            frame(*pool);
            best = std::min(best, ms_since(t0));
        }
        std::printf("%2d threads: %5.1f ms per frame (%5.1f GB/s of weights)\n", pool->size(), best,
                    bytes / best / 1e6);
        best_all = std::min(best_all, best);
    }

    // Same frame with the weights transposed once at load time to
    // [out][in]: each thread then streams whole contiguous rows instead of
    // short slices of every row, which is what memory prefetchers want.
    std::vector<std::vector<float>> wt(jobs.size());
    for (size_t k = 0; k < jobs.size(); k++) {
        const Job &j = jobs[k];
        if (j.out_in) continue; // already [out][in]
        wt[k].resize(j.in * j.cols);
        for (size_t i = 0; i < j.in; i++)
            for (size_t c = 0; c < j.cols; c++) wt[k][c * j.in + i] = j.W[i * j.ld + c];
    }
    auto frame_t = [&](Pool &pool) {
        const int n = pool.size() == 1 ? 1 : pool.size() * 2;
        for (size_t k = 0; k < jobs.size(); k++) {
            const Job &j = jobs[k];
            const float *W = j.out_in ? j.W : wt[k].data();
            const std::function<void(int)> part = [&](int i) {
                const size_t a = j.cols * i / n, e = j.cols * (i + 1) / n;
                matvec(W + a * j.in, e - a, j.in, x.data(), y.data() + a);
                if (j.b) for (size_t c = a; c < e; c++) y[c] += j.b[c];
            };
            pool.run(n, part);
        }
    };
    std::printf("weights transposed to [out][in], single-accumulator matvec (ops.cpp today):\n");
    frame_t(p1);
    for (Pool *pool : {&p1, &p4, &pall}) {
        double best = 1e9;
        for (int r = 0; r < 5; r++) {
            auto t0 = Clock::now();
            frame_t(*pool);
            best = std::min(best, ms_since(t0));
        }
        std::printf("%2d threads: %5.1f ms per frame (%5.1f GB/s of weights)\n", pool->size(), best,
                    bytes / best / 1e6);
        best_all = std::min(best_all, best);
    }
    // Fix 1: [out][in] with 16 independent accumulators per row. Float
    // additions may not be reordered, so ONE accumulator forms a serial
    // chain the compiler can't vectorize; 16 separate ones are 16 chains
    // it can run as vectors, summed once at the end.
    auto dot16 = [](const float *r, const float *xv, size_t n) {
        float acc[16] = {};
        size_t k = 0;
        for (; k + 16 <= n; k += 16)
            for (int j = 0; j < 16; j++) acc[j] += r[k + j] * xv[k + j];
        float s = 0;
        for (int j = 0; j < 16; j++) s += acc[j];
        for (; k < n; k++) s += r[k] * xv[k];
        return s;
    };
    auto frame_t16 = [&](Pool &pool) {
        const int n = pool.size() == 1 ? 1 : pool.size() * 2;
        for (size_t k = 0; k < jobs.size(); k++) {
            const Job &j = jobs[k];
            const float *W = j.out_in ? j.W : wt[k].data();
            const std::function<void(int)> part = [&](int i) {
                const size_t a = j.cols * i / n, e = j.cols * (i + 1) / n;
                for (size_t c = a; c < e; c++)
                    y[c] = dot16(W + c * j.in, x.data(), j.in) + (j.b ? j.b[c] : 0.0f);
            };
            pool.run(n, part);
        }
    };
    // Fix 2: keep the checkpoint's [in][out] layout, but give each thread a
    // block of INPUT rows: it streams one contiguous region and writes a
    // partial result; the partials are added at the end.
    std::vector<float> partial(64 * 4096);
    auto frame_rows = [&](Pool &pool) {
        const int n = pool.size() == 1 ? 1 : pool.size() * 2;
        for (const Job &j : jobs) {
            if (j.out_in) { // the two [out][in] products: row split is already contiguous
                const std::function<void(int)> part = [&](int i) {
                    const size_t a = j.cols * i / n, e = j.cols * (i + 1) / n;
                    for (size_t c = a; c < e; c++) y[c] = dot16(j.W + c * j.in, x.data(), j.in);
                };
                pool.run(n, part);
                continue;
            }
            const std::function<void(int)> part = [&](int i) {
                const size_t a = j.in * i / n, e = j.in * (i + 1) / n;
                vecmat(x.data() + a, e - a, j.W + a * j.ld, j.ld, j.cols, nullptr, partial.data() + i * 4096);
            };
            pool.run(n, part);
            for (size_t c = 0; c < j.cols; c++) {
                float s = j.b ? j.b[c] : 0.0f;
                for (int i = 0; i < n; i++) s += partial[i * 4096 + c];
                y[c] = s;
            }
        }
    };
    for (auto variant : {std::make_pair("transposed + 16-accumulator dot (fix 1):", 1),
                         std::make_pair("checkpoint layout, split by input rows (fix 2):", 2)}) {
        std::printf("%s\n", variant.first);
        for (Pool *pool : {&p1, &p4, &pall}) {
            double best = 1e9;
            for (int r = 0; r < 5; r++) {
                auto t0 = Clock::now();
                if (variant.second == 1) frame_t16(*pool); else frame_rows(*pool);
                best = std::min(best, ms_since(t0));
            }
            std::printf("%2d threads: %5.1f ms per frame (%5.1f GB/s of weights)\n", pool->size(), best,
                        bytes / best / 1e6);
            best_all = std::min(best_all, best);
        }
    }
    std::printf("(the transformer is limited by reading weights: int8 weights read 4x fewer bytes,\n"
                " so int8 would take about %.1f ms at the best bandwidth above)\n", best_all / 4);
    return best_all;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s mrt2_small.safetensors\n", argv[0]);
        return 1;
    }
    Weights w;
    std::string err;
    if (!w.open(argv[1], err)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }
    const int hw = (int)std::thread::hardware_concurrency();
#if defined(__ARM_NEON)
    const char *kernel = "ARM NEON intrinsics";
#else
    const char *kernel = "plain C++ (no NEON on this CPU)";
#endif
    std::printf("CPU threads: %d | micro-kernel: %s\n", hw, kernel);
    Pool p1(1), p4(4), pall(hw);
    const double codec = bench_codec(p1, p4, pall);
    const double tf = bench_transformer(w, p1, p4, pall);
    std::printf("\n=== verdict: fp32, best thread count for each part ===\n");
    std::printf("codec %.1f ms + transformer %.1f ms = %.1f ms per 40 ms frame -> %.2fx real time\n", codec, tf,
                codec + tf, 40.0 / (codec + tf));
    std::printf("(run one after the other; a pipelined engine overlaps codec(t) with transformer(t+1))\n");
    return 0;
}
