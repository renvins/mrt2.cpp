#include "ops.h"
#include <cmath>
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

void matvec(const float *W, size_t rows, size_t in, const float *x, float *y) {
    // [rows][in] layout: each output is a dot product of one contiguous
    // row of W with x, so again W is read front to back once.
    for (size_t r = 0; r < rows; r++) {
        const float *row = W + r * in;
        float acc = 0.0f;
        for (size_t k = 0; k < in; k++) acc += row[k] * x[k];
        y[r] = acc;
    }
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

void gelu_tanh(float *x, size_t n) {
    const float c = 0.7978845608028654f; // sqrt(2 / pi)
    for (size_t i = 0; i < n; i++) {
        const float v = x[i];
        x[i] = 0.5f * v * (1.0f + std::tanh(c * (v + 0.044715f * v * v * v)));
    }
}

void elu(float *x, size_t n) {
    // expm1(v) = exp(v) - 1, computed accurately even for v near 0.
    for (size_t i = 0; i < n; i++)
        if (x[i] <= 0.0f) x[i] = std::expm1(x[i]);
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
