#include "quant.h"
#include "threads.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#if defined(__ARM_FEATURE_DOTPROD)
#include <arm_neon.h>
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
// No int8 dot instruction (x86 for now; an AVX2 kernel comes later).
static float q8_dot(const int8_t *qw, const float *dw, const int8_t *qx, const float *dx, int in) {
    return q8_dot_ref(qw, dw, qx, dx, in);
}
#endif

void q8_matvec(const QMatrix &m, const int8_t *qx, const float *dx, const float *b, float *y) {
    const int nb = m.in / kQBlock;
    // Small matrices on this thread: waking the workers costs more.
    const int chunks = (size_t)m.rows * m.in < 64 * 1024 ? 1 : std::min(32, std::max(1, m.rows / 16));
    pool().run(chunks, [&](int c) {
        for (int r = m.rows * c / chunks; r < m.rows * (c + 1) / chunks; r++) {
            const float v = q8_dot(&m.q[(size_t)r * m.in], &m.scale[(size_t)r * nb], qx, dx, m.in);
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
