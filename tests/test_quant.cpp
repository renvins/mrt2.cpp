// test_quant.cpp - int8 (Q8) weights: the format, the kernel, and how much
// precision the real model's products lose.
//
// Usage: test_quant [MODEL.safetensors REF.safetensors]
#include "../depth.h"
#include "../encoder.h"
#include "../ops.h"
#include "../quant.h"
#include "../temporal.h"
#include "../threads.h"
#include "test.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace mrt2;

static const Weights *W = nullptr; // the model
static const Weights *R = nullptr; // the reference dump (50 frames)

// Gaussian-ish random numbers (sum of 4 uniforms), reproducible.
static std::vector<float> random_vec(size_t n, uint64_t seed, float scale = 1.0f) {
    Rng rng(seed);
    std::vector<float> v(n);
    for (float &x : v) x = scale * (float)(rng.uniform() + rng.uniform() + rng.uniform() + rng.uniform() - 2.0);
    return v;
}

TEST(quantize_rounds_to_nearest_step) {
    std::vector<float> x = random_vec(256, 1);
    for (int k = 64; k < 96; k++) x[k] = 0.0f;  // block 2: all zero
    x[100] = -5.0f;                             // block 3: one big value
    int8_t q[256];
    float d[8];
    quantize_q8(x.data(), 256, q, d);
    bool within_half_step = true, uses_full_range = true;
    for (int b = 0; b < 8; b++) {
        int qmax = 0;
        for (int k = 0; k < 32; k++) {
            const int i = b * 32 + k;
            within_half_step &= std::fabs(x[i] - d[b] * q[i]) <= 0.5f * d[b] * 1.0001f;
            qmax = std::max(qmax, std::abs((int)q[i]));
        }
        if (b != 2) uses_full_range &= qmax == 127; // the largest |x| maps to +-127
    }
    CHECK(within_half_step);
    CHECK(uses_full_range);
    CHECK(d[2] == 0.0f);
    bool zeros = true;
    for (int k = 64; k < 96; k++) zeros &= q[k] == 0;
    CHECK(zeros);
    CHECK(q[100] == -127);
}

TEST(kernel_matches_reference_bit_for_bit) {
    // The SIMD kernel against the plain version, on several shapes, on 1
    // thread and on all threads.
    const int shapes[][2] = {{1, 128}, {7, 256}, {300, 1024}, {1024, 1024}, {768, 3072}, {1024, 4096}};
    const int hw = (int)std::thread::hardware_concurrency();
    bool same = true, same_threads = true;
    for (auto &s : shapes) {
        const int rows = s[0], in = s[1];
        const std::vector<float> w = random_vec((size_t)rows * in, rows + in, 0.05f);
        const std::vector<float> x = random_vec(in, 7 * in), b = random_vec(rows, 3);
        QMatrix m;
        q8_append_rows(m, w.data(), rows, in);
        std::vector<int8_t> qx(in);
        std::vector<float> dx(in / 32), want(rows), y1(rows), yn(rows);
        quantize_q8(x.data(), in, qx.data(), dx.data());
        for (int r = 0; r < rows; r++)
            want[r] = q8_dot_ref(&m.q[(size_t)r * in], &m.scale[(size_t)r * (in / 32)], qx.data(), dx.data(), in) + b[r];
        set_threads(1);
        q8_matvec(m, x.data(), b.data(), y1.data());
        set_threads(hw);
        q8_matvec(m, x.data(), b.data(), yn.data());
        same &= std::memcmp(want.data(), y1.data(), rows * 4) == 0;
        same_threads &= std::memcmp(y1.data(), yn.data(), rows * 4) == 0;
    }
    std::printf("      6 shapes: SIMD = reference %s, 1 = %d threads %s\n", same ? "bit for bit" : "NO",
                hw, same_threads ? "bit for bit" : "NO");
    CHECK(same);
    CHECK(same_threads);
}

TEST(kernel_layout_and_fusion) {
    // A slice of a Flax [in][ld] kernel gives the same matrix as its
    // transpose given row by row; fusing two kernels = stacking their rows.
    const int in = 256, ld = 300, c0 = 17, cols = 100;
    const std::vector<float> w = random_vec((size_t)in * ld, 11);
    std::vector<float> t((size_t)cols * in); // the slice, transposed
    for (int c = 0; c < cols; c++)
        for (int i = 0; i < in; i++) t[(size_t)c * in + i] = w[(size_t)i * ld + c0 + c];
    QMatrix a, b;
    q8_append_kernel(a, w.data(), in, ld, c0, cols);
    q8_append_rows(b, t.data(), cols, in);
    CHECK(a.rows == cols && a.in == in);
    CHECK(a.q == b.q);
    CHECK(a.scale == b.scale);

    QMatrix fused;
    q8_append_kernel(fused, w.data(), in, ld, c0, cols);
    q8_append_kernel(fused, w.data(), in, ld, 0, 50);
    QMatrix second;
    q8_append_kernel(second, w.data(), in, ld, 0, 50);
    const std::vector<float> x = random_vec(in, 5);
    std::vector<float> yf(150), y1(cols), y2(50);
    q8_matvec(fused, x.data(), nullptr, yf.data());
    q8_matvec(a, x.data(), nullptr, y1.data());
    q8_matvec(second, x.data(), nullptr, y2.data());
    CHECK(std::memcmp(yf.data(), y1.data(), cols * 4) == 0);
    CHECK(std::memcmp(yf.data() + cols, y2.data(), 50 * 4) == 0);

    bool threw = false;
    try {
        q8_append_kernel(fused, w.data(), 128, ld, 0, 10); // different input size
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    CHECK(threw);
}

// ---- the real model ------------------------------------------------------------

// Relative error of the int8 product against float32, ||yq - y|| / ||y||,
// for one weight matrix (Flax layout [in][out]) over 50 real inputs.
static double rel_error(const char *name, const float *Wf, int in, int out, const float *b,
                        const std::vector<std::vector<float>> &inputs) {
    QMatrix m;
    q8_append_kernel(m, Wf, in, out, 0, out);
    std::vector<float> y(out), yq(out);
    double num = 0, den = 0;
    for (const auto &x : inputs) {
        vecmat(x.data(), in, Wf, out, out, b, y.data());
        q8_matvec(m, x.data(), b, yq.data());
        for (int j = 0; j < out; j++) {
            num += (double)(yq[j] - y[j]) * (yq[j] - y[j]);
            den += (double)y[j] * y[j];
        }
    }
    const double e = std::sqrt(num / den);
    std::printf("      %-28s [%4d x %4d]  relative error %.2e\n", name, in, out, e);
    return e;
}

TEST(real_weights_lose_little_precision) {
    // Real inputs: the reference's activations at the point where each
    // matrix is used (normalized as the model does), all 50 frames.
    const TemporalWeights tw = load_temporal(*W);
    const DepthWeights dw = load_depth(*W);
    const int F = 50;
    const float *layers = R->f32("temporal/layers", {F, kTemporalLayers, kModel});
    const float *l0_cross = R->f32("temporal/l0_cross", {F, kModel});
    const float *final_ln = R->f32("depth/final_ln", {F, kLevels, kDepthModel});

    std::vector<std::vector<float>> self_in(F), ffn_in(F), ffn_hidden(F), logit_in(F);
    const auto &l1 = tw.layer[1], &l0 = tw.layer[0];
    for (int f = 0; f < F; f++) {
        self_in[f].resize(kModel);
        rmsnorm(layers + (size_t)f * kTemporalLayers * kModel, l1.self.pre_norm, kModel, 1e-6f, self_in[f].data());
        ffn_in[f].resize(kModel);
        rmsnorm(l0_cross + (size_t)f * kModel, l0.ffn.pre_norm, kModel, 1e-6f, ffn_in[f].data());
        ffn_hidden[f].resize(kFFN);
        vecmat(ffn_in[f].data(), kModel, l0.ffn.w1, kFFN, kFFN, l0.ffn.b1, ffn_hidden[f].data());
        gelu_tanh(ffn_hidden[f].data(), kFFN);
        logit_in[f].assign(final_ln + (size_t)f * kLevels * kDepthModel,
                           final_ln + (size_t)f * kLevels * kDepthModel + kDepthModel);
    }
    double worst = 0;
    worst = std::max(worst, rel_error("temporal L1 attn query", l1.self.wq, kModel, kModel, nullptr, self_in));
    worst = std::max(worst, rel_error("temporal L0 ffn up", l0.ffn.w1, kModel, kFFN, l0.ffn.b1, ffn_in));
    worst = std::max(worst, rel_error("temporal L0 ffn down", l0.ffn.w2, kFFN, kModel, l0.ffn.b2, ffn_hidden));
    worst = std::max(worst, rel_error("depth logits, level 0", dw.logits_packed.data(), kDepthModel, kCodebook,
                                      dw.logits_b + kReserved, logit_in));
    // Rounding to 1/127 of each block's largest value: about 0.5% for the
    // weights, the same again for the inputs.
    CHECK(worst < 0.02);
}

// The whole model on the reference's 50 frames, teacher-forced: every step
// gets Google's previous tokens as input, so one changed choice can't
// snowball. Returns the logits before soft-cap, [50][12][1024].
static std::vector<float> forced_logits(bool int8) {
    const int F = 50;
    const EncoderWeights ew = load_encoder(*W);
    TemporalWeights tw = load_temporal(*W);
    DepthWeights dw = load_depth(*W);
    if (int8) {
        make_int8(tw);
        make_int8(dw);
    }
    const int32_t *cond = R->i32("cond", {F, kCondTokens});
    const int32_t *tokens = R->i32("tokens", {F, kLevels});
    std::vector<float> logits((size_t)F * kLevels * kCodebook);
    TemporalState ts;
    DepthState ds;
    Rng rng(0);
    int32_t prev[kLevels] = {}, tok[kLevels];
    float e[256], x[kModel], t[kModel];
    for (int f = 0; f < F; f++) {
        encode(ew, cond + f * kCondTokens, e);
        temporal_input(tw, prev, x);
        temporal_step(tw, ts, x, e, t);
        DepthDebug dbg;
        dbg.logits = &logits[(size_t)f * kLevels * kCodebook];
        dbg.force = tokens + f * kLevels;
        depth_frame(dw, ds, t, Sampling{0.0f, 40}, rng, tok, &dbg);
        std::copy(tokens + f * kLevels, tokens + (f + 1) * kLevels, prev);
    }
    return logits;
}

// The distribution the sampler draws from at temperature T (before top-k):
// softmax(30 * tanh(z / 30) / T), in double.
static void sampling_probs(const float *z, double T, double *p) {
    double mx = -1e300;
    for (int i = 0; i < kCodebook; i++) mx = std::max(mx, 30.0 * std::tanh(z[i] / 30.0) / T);
    double sum = 0;
    for (int i = 0; i < kCodebook; i++) sum += p[i] = std::exp(30.0 * std::tanh(z[i] / 30.0) / T - mx);
    for (int i = 0; i < kCodebook; i++) p[i] /= sum;
}

TEST(int8_model_matches_float32) {
    // What matters when sampling is the distribution the sampler draws
    // from: KL(float32 || int8) measures how far int8 moves it. A changed
    // greedy choice is fine only where float32 itself nearly hesitated,
    // so for those we check the gap between float32's two best logits.
    const std::vector<float> lf = forced_logits(false), lq = forced_logits(true);
    const int n = 50 * kLevels;
    int same_top = 0;
    float widest_gap = 0; // largest float32 top-2 gap where int8 chose differently
    double kl_sum = 0, kl_max = 0, entropy_sum = 0;
    std::vector<double> pf(kCodebook), pq(kCodebook);
    for (int i = 0; i < n; i++) {
        const float *zf = &lf[(size_t)i * kCodebook], *zq = &lq[(size_t)i * kCodebook];
        const size_t af = argmax(zf, kCodebook), aq = argmax(zq, kCodebook);
        if (af == aq) same_top++;
        else widest_gap = std::max(widest_gap, zf[af] - zf[aq]);
        sampling_probs(zf, 1.3, pf.data());
        sampling_probs(zq, 1.3, pq.data());
        double kl = 0, h = 0;
        for (int j = 0; j < kCodebook; j++) {
            if (pf[j] > 0) {
                kl += pf[j] * std::log(pf[j] / pq[j]);
                h -= pf[j] * std::log(pf[j]);
            }
        }
        kl_sum += kl;
        kl_max = std::max(kl_max, kl);
        entropy_sum += h;
    }
    std::printf("      sampling distribution (T=1.3): KL(float32 || int8) mean %.4f, max %.4f nats;"
                " its entropy %.2f nats\n", kl_sum / n, kl_max, entropy_sum / n);
    std::printf("      greedy: same choice at %d of %d positions; the others were near-ties"
                " (float32 gap at most %.2f logits)\n", same_top, n, widest_gap);
    // Measured: KL mean 0.0037 (M4) / 0.0038 (x86), max 0.0115; 561/600;
    // widest gap 0.51 (M4) / 1.21 (x86: float32 rounds a little
    // differently there, so other near-ties flip).
    CHECK(kl_sum / n < 0.01);
    CHECK(kl_max < 0.05);
    // Every changed choice must be a token float32 itself samples at least
    // 1/5 as often as its favourite at T = 1.3: gap <= 1.3 * ln 5 = 2.09.
    CHECK(widest_gap <= 1.3f * std::log(5.0f));
}

int main(int argc, char **argv) {
    RUN(quantize_rounds_to_nearest_step);
    RUN(kernel_matches_reference_bit_for_bit);
    RUN(kernel_layout_and_fusion);
    if (argc < 3) {
        std::printf("(model tests skipped: pass MODEL and REF paths)\n");
        return test::summary("test_quant");
    }
    Weights model, ref;
    std::string err;
    if (!model.open(argv[1], err) || !ref.open(argv[2], err)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }
    W = &model;
    R = &ref;
    set_threads((int)std::thread::hardware_concurrency());
    try {
        RUN(real_weights_lose_little_precision);
        RUN(int8_model_matches_float32);
    } catch (const std::runtime_error &e) {
        std::printf("    FAIL %s\n", e.what());
        test::failures++;
    }
    return test::summary("test_quant");
}
