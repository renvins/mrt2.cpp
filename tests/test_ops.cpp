// test_ops.cpp - check every op in ops.h.
//
// Two levels:
//   1. Small exact cases, against values computed by JAX (the library
//      Google's model runs on).
//   2. Real model data: run the ops on Google's own intermediate values
//      (the reference dump) and compare with what Google's model produced.
//
// Usage: test_ops [MODEL.safetensors REF.safetensors]
// Without the two files only level 1 runs.
#include "../ops.h"
#include "../weights.h"
#include "test.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

using namespace mrt2;

// ---- level 1: small exact cases ---------------------------------------------

// Inputs and the values jax.nn produced for them (float32).
static const float X[7] = {-3.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 3.0f};

TEST(gelu_tanh_matches_jax) {
    const float want[7] = {-0.003637344f, -0.158807963f, -0.154285997f, 0.0f,
                           0.345714003f, 0.841192007f, 2.996362686f};
    float x[7];
    std::copy(X, X + 7, x);
    gelu_tanh(x, 7);
    for (int i = 0; i < 7; i++) CHECK_NEAR(x[i], want[i], 1e-6);
}

TEST(elu_matches_jax) {
    const float want[7] = {-0.950212955f, -0.632120550f, -0.393469334f, 0.0f,
                           0.5f, 1.0f, 3.0f};
    float x[7];
    std::copy(X, X + 7, x);
    elu(x, 7);
    for (int i = 0; i < 7; i++) CHECK_NEAR(x[i], want[i], 1e-6);
}

TEST(softplus_matches_jax) {
    const float want[7] = {0.048587352f, 0.313261688f, 0.474076986f, 0.693147182f,
                           0.974076986f, 1.313261747f, 3.048587322f};
    for (int i = 0; i < 7; i++) CHECK_NEAR(softplus(X[i]), want[i], 1e-6);
    CHECK_NEAR(softplus(100.0f), 100.0f, 1e-6); // large x: no overflow
}

TEST(softmax_matches_jax) {
    const float want[7] = {0.001880408f, 0.013894441f, 0.022908060f, 0.037769005f,
                           0.062270559f, 0.102666795f, 0.758610725f};
    float x[7];
    std::copy(X, X + 7, x);
    softmax(x, 7);
    double sum = 0;
    for (int i = 0; i < 7; i++) {
        CHECK_NEAR(x[i], want[i], 1e-7);
        sum += x[i];
    }
    CHECK_NEAR(sum, 1.0, 1e-6);
    float big[3] = {1000.0f, 999.0f, 0.0f}; // exp(1000) would overflow without the max shift
    softmax(big, 3);
    CHECK(std::isfinite(big[0]) && big[0] > big[1] && big[2] == 0.0f);
}

TEST(softcap_matches_jax) {
    float x[5] = {-100.0f, -30.0f, 0.0f, 15.0f, 60.0f};
    const float want[5] = {-29.923738480f, -22.847826004f, 0.0f, 13.863514900f, 28.920827866f};
    softcap(x, 5, 30.0f);
    for (int i = 0; i < 5; i++) CHECK_NEAR(x[i], want[i], 1e-5);
}

TEST(norms_match_jax) {
    const float v[4] = {3.0f, 4.0f, -1.0f, 2.0f};
    const float g[4] = {1.0f, 0.5f, 2.0f, -1.0f};
    const float b[4] = {0.1f, 0.2f, 0.3f, 0.4f};
    const float want_rms[4] = {1.095445156f, 0.730296731f, -0.730296731f, -0.730296731f};
    const float want_ln[4] = {0.634522438f, 0.734522402f, -2.907134533f, 0.400000006f};
    float out[4];
    rmsnorm(v, g, 4, 1e-6f, out);
    for (int i = 0; i < 4; i++) CHECK_NEAR(out[i], want_rms[i], 1e-6);
    layernorm(v, g, b, 4, 1e-6f, out);
    for (int i = 0; i < 4; i++) CHECK_NEAR(out[i], want_ln[i], 1e-6);
    // In place (out == x) must give the same result.
    float w[4] = {3.0f, 4.0f, -1.0f, 2.0f};
    rmsnorm(w, g, 4, 1e-6f, w);
    for (int i = 0; i < 4; i++) CHECK_NEAR(w[i], want_rms[i], 1e-6);
}

TEST(vecmat_small) {
    // x = [1, 2], W = [[1, 2, 3], [4, 5, 6]] ([in=2][out=3]), b = [0.5, 0, -1]
    const float x[2] = {1, 2};
    const float W[6] = {1, 2, 3, 4, 5, 6};
    const float b[3] = {0.5f, 0, -1};
    float y[3];
    vecmat(x, 2, W, 3, 3, b, y);
    CHECK(y[0] == 9.5f && y[1] == 12.0f && y[2] == 14.0f);
    // Column slice: start at column 1, row stride still 3, 2 columns, no bias.
    vecmat(x, 2, W + 1, 3, 2, nullptr, y);
    CHECK(y[0] == 12.0f && y[1] == 15.0f);
}

TEST(matvec_agrees_with_vecmat) {
    // matvec on W stored [out][in] must equal vecmat on the same matrix
    // stored [in][out] (its transpose). Odd sizes catch indexing slips.
    const size_t in = 53, out = 37;
    std::vector<float> Wio(in * out), Woi(out * in), x(in), y1(out), y2(out);
    Rng rng(7);
    for (size_t i = 0; i < in; i++) x[i] = (float)rng.uniform() - 0.5f;
    for (size_t i = 0; i < in; i++)
        for (size_t o = 0; o < out; o++)
            Woi[o * in + i] = Wio[i * out + o] = (float)rng.uniform() - 0.5f;
    vecmat(x.data(), in, Wio.data(), out, out, nullptr, y1.data());
    matvec(Woi.data(), out, in, x.data(), y2.data());
    for (size_t o = 0; o < out; o++) CHECK_NEAR(y1[o], y2[o], 1e-5);
}

TEST(argmax_first_wins) {
    const float x[5] = {1, 3, 3, 2, -1};
    CHECK(argmax(x, 5) == 1);
}

TEST(rng_is_deterministic_and_uniform) {
    Rng a(42), b(42), c(43);
    bool same = true, differ = false;
    for (int i = 0; i < 100; i++) {
        uint64_t x = a.next(), y = b.next(), z = c.next();
        same &= x == y;
        differ |= x != z;
    }
    CHECK(same);   // same seed -> same sequence, on every platform
    CHECK(differ); // another seed -> another sequence
    double sum = 0, lo = 1, hi = 0;
    const int n = 1000000;
    for (int i = 0; i < n; i++) {
        double u = a.uniform();
        sum += u;
        lo = std::min(lo, u);
        hi = std::max(hi, u);
    }
    CHECK(lo > 0.0 && hi < 1.0); // open interval: log(u) is always finite
    CHECK_NEAR(sum / n, 0.5, 0.002);
}

// Draw `n` samples and return how often each index came up.
static std::vector<double> frequencies(const float *logits, size_t len, size_t k, float t, int n) {
    std::vector<double> f(len, 0.0);
    Rng rng(1234);
    for (int i = 0; i < n; i++) f[sample_topk(logits, len, k, t, rng)] += 1.0 / n;
    return f;
}

TEST(sample_topk_greedy) {
    const float l[5] = {0.5f, 2.0f, -1.0f, 2.0f, 1.0f};
    Rng rng(9), before(9);
    CHECK(sample_topk(l, 5, 3, 0.0f, rng) == 1); // first maximum
    CHECK(rng.next() == before.next());          // greedy used no randomness
}

TEST(sample_topk_distribution) {
    // k = 3, T = 1: only the 3 largest can be drawn, with probabilities
    // softmax([2, 1, 0.5]); the others never.
    const float l[5] = {2.0f, 1.0f, 0.5f, 0.0f, -1.0f};
    std::vector<double> f = frequencies(l, 5, 3, 1.0f, 200000);
    float p[3] = {2.0f, 1.0f, 0.5f};
    softmax(p, 3);
    for (int i = 0; i < 3; i++) CHECK_NEAR(f[i], p[i], 0.006);
    CHECK(f[3] == 0.0 && f[4] == 0.0);

    // T = 2 flattens the distribution: softmax(logits / 2).
    f = frequencies(l, 5, 3, 2.0f, 200000);
    float q[3] = {1.0f, 0.5f, 0.25f};
    softmax(q, 3);
    for (int i = 0; i < 3; i++) CHECK_NEAR(f[i], q[i], 0.006);
}

TEST(sample_topk_keeps_ties) {
    // k = 2, but three values tie for the top: Google's rule keeps every
    // value >= the k-th largest, so all three stay, each ~1/3.
    const float l[4] = {1.0f, 1.0f, 1.0f, 0.0f};
    std::vector<double> f = frequencies(l, 4, 2, 1.0f, 150000);
    for (int i = 0; i < 3; i++) CHECK_NEAR(f[i], 1.0 / 3, 0.006);
    CHECK(f[3] == 0.0);
}

// ---- level 2: Google's real numbers ---------------------------------------------

static const Weights *W = nullptr; // the model
static const Weights *R = nullptr; // the reference dump
static const int FRAMES = 50;
static const std::string P = "params/depthformer/";

TEST(layernorm_on_encoder_output) {
    // enc/out = encoder_ln(enc/combined), for every frame.
    const float *in = R->f32("enc/combined", {FRAMES, 256});
    const float *want = R->f32("enc/out", {FRAMES, 256});
    const float *g = W->f32(P + "encoder/body/encoder_ln/scale", {256});
    const float *b = W->f32(P + "encoder/body/encoder_ln/bias", {256});
    float out[256];
    double maxerr = 0;
    for (int f = 0; f < FRAMES; f++) {
        layernorm(in + f * 256, g, b, 256, 1e-6f, out);
        for (int i = 0; i < 256; i++) maxerr = std::max(maxerr, (double)std::fabs(out[i] - want[f * 256 + i]));
    }
    std::printf("      layernorm vs Google, 50 frames: max error %.2e\n", maxerr);
    CHECK(maxerr < 1e-5);
}

TEST(vecmat_on_musiccoca_branch) {
    // MusicCoCa branch = (sum of 12 embedding rows) x adapter [768 -> 256].
    // Google only saved the combined output, so recover the branch as
    // 2 * combined - regular (combined is the average of the two branches).
    const int32_t *cond = R->i32("cond", {FRAMES, 144});
    const float *comb = R->f32("enc/combined", {FRAMES, 256});
    const float *reg = R->f32("enc/regular", {FRAMES, 256});
    const std::string br = P + "encoder/body/layers_1/branched_mulan_embedder/mulan_embedder/";
    const float *E = W->f32(br + "mulan_dequantizer/embedding", {12372, 768});
    const float *A = W->f32(br + "depth_input_adapter/kernel", {768, 256});
    float sum[768], out[256];
    double maxerr = 0;
    for (int f = 0; f < FRAMES; f++) {
        std::fill(sum, sum + 768, 0.0f);
        for (int k = 0; k < 12; k++) {
            const float *row = E + (size_t)(k * 1031 + cond[f * 144 + k]) * 768;
            for (int i = 0; i < 768; i++) sum[i] += row[i];
        }
        vecmat(sum, 768, A, 256, 256, nullptr, out);
        for (int i = 0; i < 256; i++) {
            const float want = 2 * comb[f * 256 + i] - reg[f * 256 + i];
            maxerr = std::max(maxerr, (double)std::fabs(out[i] - want));
        }
    }
    std::printf("      vecmat vs Google, 50 frames: max error %.2e\n", maxerr);
    CHECK(maxerr < 1e-5);
}

TEST(vecmat_slice_on_depth_logits) {
    // Each level's 1024 logits = final_ln output x to_logits columns
    // [6 + 1024k, 6 + 1024(k+1)) + bias: the strided-slice use of vecmat.
    const float *h = R->f32("depth/final_ln", {FRAMES, 12, 768});
    const float *want = R->f32("depth/logits", {FRAMES, 12, 1024});
    const float *Wl = W->f32(P + "decoder/depth_body/to_logits/kernel", {768, 12294});
    const float *bl = W->f32(P + "decoder/depth_body/to_logits/bias", {12294});
    float out[1024];
    double maxerr = 0, maxabs = 0;
    for (int f = 0; f < FRAMES; f++)
        for (int k = 0; k < 12; k++) {
            const size_t c0 = 6 + 1024 * k;
            vecmat(h + (f * 12 + k) * 768, 768, Wl + c0, 12294, 1024, bl + c0, out);
            const float *w = want + (f * 12 + k) * 1024;
            for (int i = 0; i < 1024; i++) {
                maxerr = std::max(maxerr, (double)std::fabs(out[i] - w[i]));
                maxabs = std::max(maxabs, (double)std::fabs(w[i]));
            }
        }
    std::printf("      logits vs Google, 50x12 levels: max error %.2e (values up to %.1f)\n", maxerr, maxabs);
    CHECK(maxerr < 1e-4);
}

TEST(greedy_tokens_match_google) {
    // soft-cap, then greedy sampling over each level's slice, must pick
    // exactly Google's token at all 600 positions.
    const float *logits = R->f32("depth/logits", {FRAMES, 12, 1024});
    const int32_t *tokens = R->i32("tokens", {FRAMES, 12});
    Rng rng(0);
    int match = 0;
    float l[1024];
    for (int f = 0; f < FRAMES; f++)
        for (int k = 0; k < 12; k++) {
            std::copy(logits + (f * 12 + k) * 1024, logits + (f * 12 + k + 1) * 1024, l);
            softcap(l, 1024, 30.0f);
            const int tok = 6 + 1024 * k + (int)sample_topk(l, 1024, 40, 0.0f, rng);
            match += tok == tokens[f * 12 + k];
        }
    std::printf("      greedy tokens identical to Google: %d / 600\n", match);
    CHECK(match == 600);
}

int main(int argc, char **argv) {
    RUN(gelu_tanh_matches_jax);
    RUN(elu_matches_jax);
    RUN(softplus_matches_jax);
    RUN(softmax_matches_jax);
    RUN(softcap_matches_jax);
    RUN(norms_match_jax);
    RUN(vecmat_small);
    RUN(matvec_agrees_with_vecmat);
    RUN(argmax_first_wins);
    RUN(rng_is_deterministic_and_uniform);
    RUN(sample_topk_greedy);
    RUN(sample_topk_distribution);
    RUN(sample_topk_keeps_ties);

    if (argc < 3) {
        std::printf("(real-data tests skipped: pass MODEL and REF paths)\n");
        return test::summary("test_ops");
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
        RUN(layernorm_on_encoder_output);
        RUN(vecmat_on_musiccoca_branch);
        RUN(vecmat_slice_on_depth_logits);
        RUN(greedy_tokens_match_google);
    } catch (const std::runtime_error &e) { // a missing/mis-shaped tensor
        std::printf("    FAIL %s\n", e.what());
        test::failures++;
    }
    return test::summary("test_ops");
}
