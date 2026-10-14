#include "quant.h"
#include "cpu.h"
#include "threads.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#if defined(__ARM_FEATURE_DOTPROD)
#include <arm_neon.h>
#endif

// The SIMD kernels must give exactly the bits of q8_dot_ref(), which
// multiplies and adds in separate steps. GCC would otherwise fuse a
// multiply and a following add into one instruction (rounding once
// instead of twice) in some places and not others.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC optimize("fp-contract=off")
#endif

namespace mrt2 {

void quantize_q8(const float *x, size_t n, int8_t *q, float *d) {
    for (size_t b = 0; b < n / kQBlock; b++) {
        const float *xb = x + b * kQBlock;
        float amax = 0.0f;
        for (int k = 0; k < kQBlock; k++) amax = std::max(amax, std::fabs(xb[k]));
        const float scale = amax / 127.0f;
        const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
        d[b] = scale;
        // Round to nearest (ties to even); |x * inv| <= 127 by construction.
        for (int k = 0; k < kQBlock; k++) q[b * kQBlock + k] = (int8_t)std::nearbyint(xb[k] * inv);
    }
}

// New rows go at the end; returns the index of the first one.
static int grow(QMatrix &m, int in, int rows) {
    if (in % (4 * kQBlock) != 0 || in > kQMaxIn)
        throw std::invalid_argument("q8: input size must be a multiple of 128, at most 4096");
    if (m.rows > 0 && m.in != in) throw std::invalid_argument("q8: fused matrices need the same input size");
    const int r0 = m.rows;
    m.in = in;
    m.rows += rows;
    m.q.resize((size_t)m.rows * in);
    m.scale.resize((size_t)m.rows * (in / kQBlock));
    return r0;
}

void q8_append_kernel(QMatrix &m, const float *W, int in, int ld, int c0, int cols) {
    const int r0 = grow(m, in, cols);
    const int nb = in / kQBlock;
    // One chunk per block of 32 inputs: it reads 32 consecutive rows of W
    // (cache-friendly), and writes 32 bytes + 1 scale per output row.
    pool().run(nb, [&](int b) {
        float col[kQBlock];
        for (int c = 0; c < cols; c++) {
            for (int k = 0; k < kQBlock; k++) col[k] = W[(size_t)(b * kQBlock + k) * ld + c0 + c];
            const size_t row = (size_t)(r0 + c);
            quantize_q8(col, kQBlock, &m.q[row * in + b * kQBlock], &m.scale[row * nb + b]);
        }
    });
}

void q8_append_rows(QMatrix &m, const float *W, int rows, int in) {
    const int r0 = grow(m, in, rows);
    const int nb = in / kQBlock;
    pool().run(16, [&](int c) {
        for (int r = rows * c / 16; r < rows * (c + 1) / 16; r++)
            quantize_q8(W + (size_t)r * in, in, &m.q[(size_t)(r0 + r) * in], &m.scale[(size_t)(r0 + r) * nb]);
    });
}

float q8_dot_ref(const int8_t *qw, const float *dw, const int8_t *qx, const float *dx, int in) {
    float lane[4] = {};
    for (int b = 0; b < in / kQBlock; b++) {
        int32_t s = 0; // at most 32 * 127 * 127: no overflow
        for (int k = 0; k < kQBlock; k++) s += (int32_t)qw[b * kQBlock + k] * qx[b * kQBlock + k];
        // Two statements, so the compiler can't fuse them into one
        // multiply-add (which rounds once instead of twice).
        const float p = (float)s * (dw[b] * dx[b]);
        lane[b % 4] += p;
    }
    return (lane[0] + lane[1]) + (lane[2] + lane[3]);
}

#if defined(__ARM_FEATURE_DOTPROD)
// 4 blocks (128 numbers) per iteration. vdotq_s32 multiplies 16 int8 pairs
// and adds them, in groups of 4, into the 4 int32 lanes of its accumulator;
// three pairwise adds then give the 4 block sums [S0 S1 S2 S3] in one
// vector, which is converted to float and scaled with one multiply.
static float q8_dot(const int8_t *qw, const float *dw, const int8_t *qx, const float *dx, int in) {
    const int32x4_t zero = vdupq_n_s32(0);
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (int b = 0; b < in / kQBlock; b += 4) {
        const int8_t *w = qw + b * kQBlock, *x = qx + b * kQBlock;
        int32x4_t s[4];
        for (int j = 0; j < 4; j++)
            s[j] = vdotq_s32(vdotq_s32(zero, vld1q_s8(w + 32 * j), vld1q_s8(x + 32 * j)),
                             vld1q_s8(w + 32 * j + 16), vld1q_s8(x + 32 * j + 16));
        const int32x4_t sums = vpaddq_s32(vpaddq_s32(s[0], s[1]), vpaddq_s32(s[2], s[3]));
        const float32x4_t d = vmulq_f32(vld1q_f32(dw + b), vld1q_f32(dx + b));
        acc = vaddq_f32(acc, vmulq_f32(vcvtq_f32_s32(sums), d));
    }
    return (vgetq_lane_f32(acc, 0) + vgetq_lane_f32(acc, 1)) + (vgetq_lane_f32(acc, 2) + vgetq_lane_f32(acc, 3));
}
#else
// No int8 dot instruction known at compile time: the plain version.
static float q8_dot(const int8_t *qw, const float *dw, const int8_t *qx, const float *dx, int in) {
    return q8_dot_ref(qw, dw, qx, dx, in);
}
#endif

#if defined(MRT2_X86)
// AVX2 has no signed int8 x int8 dot product, but it has maddubs:
// unsigned bytes x signed bytes, neighbouring pairs added into int16.
// Make the weights unsigned (|w|) and move their sign onto the inputs:
// |w| * (x * sign(w)) = w * x. A pair is at most 2 * 127 * 127 = 32258,
// which fits in int16 because Q8 values are -127..127 (never -128).
MRT2_AVX2 static inline __m256i q8_block_avx2(const int8_t *w, const int8_t *x) {
    const __m256i vw = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(w));
    const __m256i vx = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(x));
    const __m256i pairs = _mm256_maddubs_epi16(_mm256_sign_epi8(vw, vw), _mm256_sign_epi8(vx, vw));
    return _mm256_madd_epi16(pairs, _mm256_set1_epi16(1)); // 8 int32 partial sums of the block
}

// The 8 lanes of each of 4 vectors, added up: [S0 S1 S2 S3]. hadd works
// within each 128-bit half, so after two rounds the low half holds lanes
// 0-3 of each sum and the high half lanes 4-7; adding the halves finishes.
MRT2_AVX2 static inline __m128i q8_sum4_avx2(__m256i a, __m256i b, __m256i c, __m256i d) {
    const __m256i abcd = _mm256_hadd_epi32(_mm256_hadd_epi32(a, b), _mm256_hadd_epi32(c, d));
    return _mm_add_epi32(_mm256_castsi256_si128(abcd), _mm256_extracti128_si256(abcd, 1));
}

// Same structure as the NEON kernel: 4 blocks per iteration, block i
// into float lane i % 4, lanes added as (l0 + l1) + (l2 + l3).
MRT2_AVX2 static float q8_dot_avx2(const int8_t *qw, const float *dw, const int8_t *qx, const float *dx, int in) {
    __m128 acc = _mm_setzero_ps();
    for (int b = 0; b < in / kQBlock; b += 4) {
        const int8_t *w = qw + b * kQBlock, *x = qx + b * kQBlock;
        const __m128i sums = q8_sum4_avx2(q8_block_avx2(w, x), q8_block_avx2(w + 32, x + 32),
                                          q8_block_avx2(w + 64, x + 64), q8_block_avx2(w + 96, x + 96));
        const __m128 d = _mm_mul_ps(_mm_loadu_ps(dw + b), _mm_loadu_ps(dx + b));
        acc = _mm_add_ps(acc, _mm_mul_ps(_mm_cvtepi32_ps(sums), d));
    }
    float l[4];
    _mm_storeu_ps(l, acc);
    return (l[0] + l[1]) + (l[2] + l[3]);
}
#endif

// The kernel for this CPU, chosen once.
using QDot = float (*)(const int8_t *, const float *, const int8_t *, const float *, int);
static QDot pick_q8_dot() {
#if defined(MRT2_X86)
    if (cpu_avx2()) return q8_dot_avx2;
#endif
    return q8_dot;
}

void q8_matvec(const QMatrix &m, const int8_t *qx, const float *dx, const float *b, float *y) {
    static const QDot dot_fn = pick_q8_dot();
    const int nb = m.in / kQBlock;
    // Small matrices on this thread: waking the workers costs more.
    const int chunks = (size_t)m.rows * m.in < 64 * 1024 ? 1 : std::min(32, std::max(1, m.rows / 16));
    pool().run(chunks, [&](int c) {
        for (int r = m.rows * c / chunks; r < m.rows * (c + 1) / chunks; r++) {
            const float v = dot_fn(&m.q[(size_t)r * m.in], &m.scale[(size_t)r * nb], qx, dx, m.in);
            y[r] = b ? v + b[r] : v;
        }
    });
}

void q8_matvec(const QMatrix &m, const float *x, const float *b, float *y) {
    int8_t qx[kQMaxIn];
    float dx[kQMaxIn / kQBlock];
    quantize_q8(x, m.in, qx, dx);
    q8_matvec(m, qx, dx, b, y);
}

} // namespace mrt2
