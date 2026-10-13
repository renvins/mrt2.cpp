#include "ops.h"
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>

namespace mrt2 {

// ---- matrix x vector ------------------------------------------------------

void vecmat(const float *x, size_t in, const float *W, size_t ld, size_t cols,
            const float *b, float *y) {
    for (size_t j = 0; j < cols; j++) y[j] = b ? b[j] : 0.0f;
    // Walk W one row at a time. Row i is contiguous in memory, so the
    // weights are read front to back exactly once: the cache-friendly
    // order for the [in][out] layout. Each row adds x[i] * row to y.
    for (size_t i = 0; i < in; i++) {
        const float xi = x[i];
        const float *row = W + i * ld;
        for (size_t j = 0; j < cols; j++) y[j] += xi * row[j];
    }
}

float dot(const float *a, const float *b, size_t n) {
    float acc[16] = {};
    size_t i = 0;
    for (; i + 16 <= n; i += 16)
        for (int j = 0; j < 16; j++) acc[j] += a[i + j] * b[i + j];
    float s = 0.0f;
    for (int j = 0; j < 16; j++) s += acc[j];
    for (; i < n; i++) s += a[i] * b[i]; // the last n % 16 elements
    return s;
}

void matvec(const float *W, size_t rows, size_t in, const float *x, float *y) {
    // [rows][in] layout: each output is a dot product of one contiguous
    // row of W with x, so again W is read front to back once.
    for (size_t r = 0; r < rows; r++) y[r] = dot(W + r * in, x, in);
}

// ---- normalization --------------------------------------------------------
//
// The sums below use double. A float sum of 1024 values loses a few low
// bits, and in a different way than JAX does (it adds in another order).
// Summing in double gives (almost) the exact value, which sits within
// rounding distance of any float32 order, including Google's.

void rmsnorm(const float *x, const float *gamma, size_t n, float eps, float *out) {
    double ss = 0.0;
    for (size_t i = 0; i < n; i++) ss += (double)x[i] * x[i];
    const float r = (float)(1.0 / std::sqrt(ss / n + eps));
    for (size_t i = 0; i < n; i++) out[i] = (x[i] * r) * gamma[i];
}

void layernorm(const float *x, const float *gamma, const float *beta, size_t n,
               float eps, float *out) {
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) sum += x[i];
    const double mean = sum / n;
    // Two passes (mean first, then squared distances) instead of
    // E[x^2] - mean^2, which can cancel catastrophically.
    double var = 0.0;
    for (size_t i = 0; i < n; i++) {
        const double d = x[i] - mean;
        var += d * d;
    }
    var /= n;
    const float r = (float)(1.0 / std::sqrt(var + eps));
    const float m = (float)mean;
    for (size_t i = 0; i < n; i++) out[i] = (x[i] - m) * r * gamma[i] + beta[i];
}

// ---- element-wise -----------------------------------------------------------

// exp(x) built from plain arithmetic, so loops calling it can be
// vectorized (a call to std::exp can't be):
//   x = n * ln2 + r, n whole, |r| <= ln2 / 2      (Cody-Waite: ln2 in two
//                                                    parts keeps r exact)
//   exp(x) = 2^n * exp(r), exp(r) from a degree-6 polynomial (error ~1e-7),
//   2^n made directly in the float's exponent bits.
// Inputs are clamped to [-87, 88], the range of normal float results.
static inline float fast_exp(float x) {
    x = x < -87.0f ? -87.0f : x > 88.0f ? 88.0f : x;
    const float n = std::floor(x * 1.44269504f + 0.5f); // round(x / ln2)
    float r = x - n * 0.693145752f;                     // ln2, high part
    r = r - n * 1.42860677e-6f;                         // ln2, low part
    const float p = 1.0f + r * (1.0f + r * (0.5f + r * (1.0f / 6 + r * (1.0f / 24 + r * (1.0f / 120 + r * (1.0f / 720))))));
    return p * std::bit_cast<float>(((int32_t)n + 127) << 23);
}

void gelu_tanh(float *x, size_t n) {
    // 0.5 * (1 + tanh(y)) == 1 / (1 + exp(-2y)), so
    // gelu(x) = x / (1 + exp(-2y)) with y = sqrt(2/pi) (x + 0.044715 x^3):
    // the same function, without tanh and without cancellation near 0.
    const float c2 = 2.0f * 0.7978845608028654f;
    for (size_t i = 0; i < n; i++) {
        const float v = x[i];
        x[i] = v / (1.0f + fast_exp(-c2 * (v + 0.044715f * v * v * v)));
    }
}

void elu(float *x, size_t n) {
    // x > 0: x. Otherwise exp(x) - 1, except near 0 where that subtraction
    // would lose digits (exp(x) ~ 1): there the series of expm1 is used,
    // x + x^2/2 + ... + x^8/8!, exact to float precision for |x| < 0.35.
    // Every branch is a select, so the loop vectorizes.
    for (size_t i = 0; i < n; i++) {
        const float v = x[i];
        const float series = v * (1.0f + v * (0.5f + v * (1.0f / 6 + v * (1.0f / 24 + v * (1.0f / 120 + v * (1.0f / 720 + v * (1.0f / 5040 + v * (1.0f / 40320))))))));
        const float neg = v > -0.35f ? series : fast_exp(v) - 1.0f;
        x[i] = v > 0.0f ? v : neg;
    }
}

void softcap(float *x, size_t n, float cap) {
    for (size_t i = 0; i < n; i++) x[i] = cap * std::tanh(x[i] / cap);
}

void softmax(float *x, size_t n) {
    float mx = x[0];
    for (size_t i = 1; i < n; i++) mx = x[i] > mx ? x[i] : mx;
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        x[i] = std::exp(x[i] - mx);
        sum += x[i];
    }
    const float inv = (float)(1.0 / sum);
    for (size_t i = 0; i < n; i++) x[i] *= inv;
}

float softplus(float x) {
    // For x > 20, log(1 + e^x) equals x to within float precision, and
    // e^x would overflow for x > ~88.
    return x > 20.0f ? x : std::log1p(std::exp(x));
}

// ---- sampling -----------------------------------------------------------------

// splitmix64: spreads one 64-bit seed into the 4 words of xoshiro's state
// (the seeding method its authors recommend).
static uint64_t splitmix64(uint64_t &z) {
    z += 0x9e3779b97f4a7c15ULL;
    uint64_t r = z;
    r = (r ^ (r >> 30)) * 0xbf58476d1ce4e5b9ULL;
    r = (r ^ (r >> 27)) * 0x94d049bb133111ebULL;
    return r ^ (r >> 31);
}

static inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

Rng::Rng(uint64_t seed) {
    for (uint64_t &w : s) w = splitmix64(seed);
}

uint64_t Rng::next() {
    const uint64_t result = rotl(s[1] * 5, 7) * 9;
    const uint64_t t = s[1] << 17;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = rotl(s[3], 45);
    return result;
}

double Rng::uniform() {
    // Top 53 bits -> an integer in [0, 2^53), +0.5 moves it off 0, and
    // scaling by 2^-53 lands strictly inside (0, 1).
    return ((double)(next() >> 11) + 0.5) * 0x1.0p-53;
}

size_t argmax(const float *x, size_t n) {
    size_t best = 0;
    for (size_t i = 1; i < n; i++)
        if (x[i] > x[best]) best = i; // strict '>': the first maximum wins
    return best;
}

size_t sample_topk(const float *logits, size_t n, size_t k, float temperature, Rng &rng) {
    // Greedy: the largest value is always in the top k, so top-k changes
    // nothing and no random numbers are needed.
    if (temperature <= 0.0f) return argmax(logits, n);

    // Same clamping as Google's code (k is clipped to [1, n]).
    if (k < 1) k = 1;
    if (k > n) k = n;
    if (k > kMaxTopK) k = kMaxTopK;

    // Find the k-th largest value: keep the k largest seen so far in
    // `top`, sorted from largest to smallest (insertion sort: k is small,
    // and it needs no memory allocation).
    float top[kMaxTopK];
    size_t m = 0;
    for (size_t i = 0; i < n; i++) {
        const float v = logits[i];
        if (m == k && v <= top[k - 1]) continue; // not among the k largest
        size_t p = m < k ? m++ : k - 1;          // slot to fill (drop the smallest if full)
        while (p > 0 && top[p - 1] < v) {        // shift smaller ones down
            top[p] = top[p - 1];
            p--;
        }
        top[p] = v;
    }
    const float kth = top[k - 1];

    // Gumbel-max: argmax(logit + T * g) with g = -log(-log(u)) draws index
    // i with probability softmax(logits / T)[i], restricted to the kept set.
    size_t best = 0;
    double best_score = -std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < n; i++) {
        if (logits[i] < kth) continue; // outside the top k
        const double g = -std::log(-std::log(rng.uniform()));
        const double score = logits[i] + (double)temperature * g;
        if (score > best_score) {
            best_score = score;
            best = i;
        }
    }
    return best;
}

} // namespace mrt2
