#include "ops.h"
#include "cpu.h"
#include "threads.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>
#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#endif

namespace mrt2 {

// ---- matrix x vector ------------------------------------------------------

// y = sum over rows [r0, r1) of x[i] * W[i] (y is overwritten). Row i is
// contiguous in memory, so the weights are read front to back exactly
// once: the cache-friendly order for the [in][out] layout.
static void vecmat_rows(const float *x, size_t r0, size_t r1, const float *W, size_t ld, size_t cols, float *y) {
    for (size_t j = 0; j < cols; j++) y[j] = 0.0f;
    for (size_t i = r0; i < r1; i++) {
        const float xi = x[i];
        const float *row = W + i * ld;
        for (size_t j = 0; j < cols; j++) y[j] += xi * row[j];
    }
}

// Big products are split into this many chunks of input rows, ALWAYS this
// many, whatever the number of threads: each chunk sums its rows into a
// partial result, and the partials are added in a fixed order. So the
// rounding, and every output bit, is the same on 1 thread or 10.
constexpr size_t kRowChunks = 16;

void vecmat(const float *x, size_t in, const float *W, size_t ld, size_t cols,
            const float *b, float *y) {
    // Small products: one pass on this thread (waking workers costs more).
    if (in < kRowChunks * 32 || in * cols < 128 * 1024) {
        vecmat_rows(x, 0, in, W, ld, cols, y);
        if (b) for (size_t j = 0; j < cols; j++) y[j] += b[j];
        return;
    }
    // Partial results, one row of `cols` per chunk. Only the generation
    // thread calls vecmat, so one buffer that only ever grows is enough.
    static std::vector<float> partial;
    if (partial.size() < kRowChunks * cols) partial.resize(kRowChunks * cols);
    float *part = partial.data();
    pool().run((int)kRowChunks, [&](int c) {
        vecmat_rows(x, in * c / kRowChunks, in * (c + 1) / kRowChunks, W, ld, cols, part + (size_t)c * cols);
    });
    // y = b + partial[0] + partial[1] + ... in that order, split by columns.
    const int parts = 8;
    pool().run(parts, [&](int k) {
        for (size_t j = cols * k / parts; j < cols * (k + 1) / parts; j++) {
            float s = b ? b[j] : 0.0f;
            for (size_t c = 0; c < kRowChunks; c++) s += part[c * cols + j];
            y[j] = s;
        }
    });
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
    // row of W with x, so again W is read front to back once. Each output
    // is computed whole by one thread: the same bits for any thread count.
    const int chunks = rows * in < 128 * 1024 ? 1 : 16;
    pool().run(chunks, [&](int c) {
        for (size_t r = rows * c / chunks; r < rows * (c + 1) / chunks; r++) y[r] = dot(W + r * in, x, in);
    });
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

// ---- the same exp, 4 (NEON) or 8 (AVX2) numbers at a time -------------------------
//
// Plain C++ loops using fast_exp vectorize on ARM, but not on baseline
// x86: SSE2 has no vector floor, so x86 would compute one number at a
// time (CI: ELU 8 ms per frame on Windows). And GCC doesn't vectorize them
// on ARM either. So ELU and GELU are also written with intrinsics, sharing
// these two helpers. Same algorithm as fast_exp: x = n ln2 + r,
// exp(x) = 2^n * poly(r), with 2^n built in a float's exponent bits.

#if defined(__aarch64__) || defined(_M_ARM64)
static inline float32x4_t exp_neon(float32x4_t x) {
    const float32x4_t one = vdupq_n_f32(1.0f);
    const float32x4_t c = vminq_f32(vmaxq_f32(x, vdupq_n_f32(-87.0f)), vdupq_n_f32(88.0f));
    // vfmaq_f32(a, b, c) = a + b * c; vfmsq_f32(a, b, c) = a - b * c.
    const float32x4_t n = vrndmq_f32(vfmaq_f32(vdupq_n_f32(0.5f), c, vdupq_n_f32(1.44269504f))); // floor
    float32x4_t r = vfmsq_f32(c, n, vdupq_n_f32(0.693145752f)); // ln2, high part
    r = vfmsq_f32(r, n, vdupq_n_f32(1.42860677e-6f));          // ln2, low part
    float32x4_t p = vdupq_n_f32(1.0f / 720);
    p = vfmaq_f32(vdupq_n_f32(1.0f / 120), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.0f / 24), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.0f / 6), p, r);
    p = vfmaq_f32(vdupq_n_f32(0.5f), p, r);
    p = vfmaq_f32(one, p, r);
    p = vfmaq_f32(one, p, r);
    const int32x4_t e = vshlq_n_s32(vaddq_s32(vcvtq_s32_f32(n), vdupq_n_s32(127)), 23); // 2^n
    return vmulq_f32(p, vreinterpretq_f32_s32(e));
}
#endif

#if defined(MRT2_X86)
MRT2_AVX2 static inline __m256 exp_avx2(__m256 x) {
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 c = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(-87.0f)), _mm256_set1_ps(88.0f));
    const __m256 n = _mm256_floor_ps(_mm256_fmadd_ps(c, _mm256_set1_ps(1.44269504f), _mm256_set1_ps(0.5f)));
    __m256 r = _mm256_fnmadd_ps(n, _mm256_set1_ps(0.693145752f), c); // ln2, high part
    r = _mm256_fnmadd_ps(n, _mm256_set1_ps(1.42860677e-6f), r);      // ln2, low part
    __m256 p = _mm256_set1_ps(1.0f / 720);
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f / 120));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f / 24));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f / 6));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.5f));
    p = _mm256_fmadd_ps(p, r, one);
    p = _mm256_fmadd_ps(p, r, one);
    const __m256i e = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(p, _mm256_castsi256_ps(e));
}
#endif

// ---- GELU -----------------------------------------------------------------------------
//
// 0.5 * (1 + tanh(y)) == 1 / (1 + exp(-2y)), so
// gelu(x) = x / (1 + exp(-2y)) with y = sqrt(2/pi) (x + 0.044715 x^3):
// the same function, without tanh and without cancellation near 0.
static const float kGeluC2 = 2.0f * 0.7978845608028654f;

static void gelu_range(float *x, size_t i0, size_t i1) {
    for (size_t i = i0; i < i1; i++) {
        const float v = x[i];
        x[i] = v / (1.0f + fast_exp(-kGeluC2 * (v + 0.044715f * v * v * v)));
    }
}

#if defined(__aarch64__) || defined(_M_ARM64)
static void gelu_range_neon(float *x, size_t i0, size_t i1) {
    size_t i = i0;
    for (; i + 4 <= i1; i += 4) {
        const float32x4_t v = vld1q_f32(x + i);
        const float32x4_t inner = vfmaq_f32(v, vmulq_f32(vdupq_n_f32(0.044715f), v), vmulq_f32(v, v)); // v + 0.0447 v^3
        const float32x4_t e = exp_neon(vmulq_f32(vdupq_n_f32(-kGeluC2), inner));
        vst1q_f32(x + i, vdivq_f32(v, vaddq_f32(vdupq_n_f32(1.0f), e)));
    }
    gelu_range(x, i, i1); // the last < 4 numbers
}
#endif

#if defined(MRT2_X86)
MRT2_AVX2 static void gelu_range_avx2(float *x, size_t i0, size_t i1) {
    size_t i = i0;
    for (; i + 8 <= i1; i += 8) {
        const __m256 v = _mm256_loadu_ps(x + i);
        const __m256 inner = _mm256_fmadd_ps(_mm256_mul_ps(_mm256_set1_ps(0.044715f), v), _mm256_mul_ps(v, v), v);
        const __m256 e = exp_avx2(_mm256_mul_ps(_mm256_set1_ps(-kGeluC2), inner));
        _mm256_storeu_ps(x + i, _mm256_div_ps(v, _mm256_add_ps(_mm256_set1_ps(1.0f), e)));
    }
    gelu_range(x, i, i1); // the last < 8 numbers
}
#endif

void gelu_tanh(float *x, size_t n) {
    // At most 4096 numbers (one FFN hidden layer): too few to share out.
#if defined(__aarch64__) || defined(_M_ARM64)
    gelu_range_neon(x, 0, n);
#else
#if defined(MRT2_X86)
    if (cpu_avx2()) return gelu_range_avx2(x, 0, n);
#endif
    gelu_range(x, 0, n);
#endif
}

// ---- ELU ------------------------------------------------------------------------------

static void elu_range(float *x, size_t i0, size_t i1) {
    // x > 0: x. Otherwise exp(x) - 1, except near 0 where that subtraction
    // would lose digits (exp(x) ~ 1): there the series of expm1 is used,
    // x + x^2/2 + ... + x^8/8!, exact to float precision for |x| < 0.35.
    // Every branch is a select, so the loop can vectorize.
    for (size_t i = i0; i < i1; i++) {
        const float v = x[i];
        const float series = v * (1.0f + v * (0.5f + v * (1.0f / 6 + v * (1.0f / 24 + v * (1.0f / 120 + v * (1.0f / 720 + v * (1.0f / 5040 + v * (1.0f / 40320))))))));
        const float neg = v > -0.35f ? series : fast_exp(v) - 1.0f;
        x[i] = v > 0.0f ? v : neg;
    }
}

#if defined(__aarch64__) || defined(_M_ARM64)
// Both branches for all 4 lanes, then a mask picks one per lane (vbslq).
static void elu_range_neon(float *x, size_t i0, size_t i1) {
    const float32x4_t one = vdupq_n_f32(1.0f);
    size_t i = i0;
    for (; i + 4 <= i1; i += 4) {
        const float32x4_t v = vld1q_f32(x + i);
        // Near 0: the expm1 series.
        float32x4_t s = vdupq_n_f32(1.0f / 40320);
        s = vfmaq_f32(vdupq_n_f32(1.0f / 5040), s, v);
        s = vfmaq_f32(vdupq_n_f32(1.0f / 720), s, v);
        s = vfmaq_f32(vdupq_n_f32(1.0f / 120), s, v);
        s = vfmaq_f32(vdupq_n_f32(1.0f / 24), s, v);
        s = vfmaq_f32(vdupq_n_f32(1.0f / 6), s, v);
        s = vfmaq_f32(vdupq_n_f32(0.5f), s, v);
        s = vfmaq_f32(one, s, v);
        const float32x4_t series = vmulq_f32(s, v);
        const float32x4_t expm1 = vsubq_f32(exp_neon(v), one);
        const float32x4_t neg = vbslq_f32(vcgtq_f32(v, vdupq_n_f32(-0.35f)), series, expm1);
        vst1q_f32(x + i, vbslq_f32(vcgtq_f32(v, vdupq_n_f32(0.0f)), v, neg));
    }
    elu_range(x, i, i1); // the last < 4 numbers
}
#endif

#if defined(MRT2_X86)
// Both branches for all 8 lanes, then a mask picks one per lane (blendv).
MRT2_AVX2 static void elu_range_avx2(float *x, size_t i0, size_t i1) {
    const __m256 zero = _mm256_setzero_ps(), one = _mm256_set1_ps(1.0f);
    size_t i = i0;
    for (; i + 8 <= i1; i += 8) {
        const __m256 v = _mm256_loadu_ps(x + i);
        // Near 0: the expm1 series, x + x^2/2 + ... + x^8/8!
        __m256 s = _mm256_set1_ps(1.0f / 40320);
        s = _mm256_fmadd_ps(s, v, _mm256_set1_ps(1.0f / 5040));
        s = _mm256_fmadd_ps(s, v, _mm256_set1_ps(1.0f / 720));
        s = _mm256_fmadd_ps(s, v, _mm256_set1_ps(1.0f / 120));
        s = _mm256_fmadd_ps(s, v, _mm256_set1_ps(1.0f / 24));
        s = _mm256_fmadd_ps(s, v, _mm256_set1_ps(1.0f / 6));
        s = _mm256_fmadd_ps(s, v, _mm256_set1_ps(0.5f));
        s = _mm256_fmadd_ps(s, v, one);
        const __m256 series = _mm256_mul_ps(s, v);
        const __m256 expm1 = _mm256_sub_ps(exp_avx2(v), one);
        const __m256 neg = _mm256_blendv_ps(expm1, series, _mm256_cmp_ps(v, _mm256_set1_ps(-0.35f), _CMP_GT_OQ));
        _mm256_storeu_ps(x + i, _mm256_blendv_ps(neg, v, _mm256_cmp_ps(v, zero, _CMP_GT_OQ)));
    }
    elu_range(x, i, i1); // the last < 8 numbers
}
#endif

void elu(float *x, size_t n) {
    // Element-wise, so any split gives the same bits.
    const int chunks = n < 64 * 1024 ? 1 : 32;
#if defined(__aarch64__) || defined(_M_ARM64)
    pool().run(chunks, [&](int c) { elu_range_neon(x, n * c / chunks, n * (c + 1) / chunks); });
#else
#if defined(MRT2_X86)
    if (cpu_avx2()) {
        pool().run(chunks, [&](int c) { elu_range_avx2(x, n * c / chunks, n * (c + 1) / chunks); });
        return;
    }
#endif
    pool().run(chunks, [&](int c) { elu_range(x, n * c / chunks, n * (c + 1) / chunks); });
#endif
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
