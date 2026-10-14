#include "conv.h"
#include "cpu.h"
#include "ops.h"
#include "profile.h"
#include "threads.h"
#include <algorithm>
#include <cassert>
#include <cstring>
#if defined(__ARM_NEON) || defined(_M_ARM64) // _M_ARM64: Visual Studio's name for 64-bit ARM
#include <arm_neon.h>
#endif

namespace mrt2 {

// ---- geometry -------------------------------------------------------------------
//
// Output (t, f) and tap (kt, kf) read input position
//   ((t + kt - pt0) / st, (f + kf - pf0) / sf)
// when both divisions are exact and the result is inside the input;
// otherwise the tap lands on padding or an inserted zero and reads nothing.
// For a regular conv: st = sf = 1, pt0 = kt - 1 (causal), pf0 = pl.
// The output tensor may start at time t0 (skipped steps): its row r is
// output time t0 + r. t0 is a multiple of st, so a row's phase (r % st)
// is its time's phase.

struct Geometry {
    int st, sf;   // strides
    int pt0, pf0; // left padding, time and frequency
    int to, fo;   // output size (rows computed, frequency bins)
    int t0;       // time of output row 0
};

static inline const float *tap_source(const Tensor &x, const Geometry &g, int t, int f, int kt, int kf) {
    const int pt = g.t0 + t + kt - g.pt0, pf = f + kf - g.pf0;
    if (g.st == 1 && g.sf == 1) { // regular conv: no divisions needed (they cost ~10 cycles each)
        if (pt < 0 || pf < 0 || pt >= x.t || pf >= x.f) return nullptr;
        return x.at(pt, pf);
    }
    if (pt < 0 || pf < 0 || pt % g.st || pf % g.sf) return nullptr;
    const int ti = pt / g.st, fi = pf / g.sf;
    if (ti >= x.t || fi >= x.f) return nullptr;
    return x.at(ti, fi);
}

static Geometry conv_geometry(const Tensor &x, const Conv &c, int pl, int pr, int skip) {
    assert(skip >= 0 && skip <= x.t);
    return {1, 1, c.kt - 1, pl, x.t - skip, x.f + pl + pr - c.kf + 1, skip};
}

static Geometry transpose_geometry(const Tensor &x, const Conv &c, int st, int sf, int skip) {
    assert(skip >= 0 && skip % st == 0 && skip <= x.t * st);
    // Google's paddings: causal in time, "same" in frequency (see plan 2.6).
    const int pa = c.kf + sf - 2;
    const int pfl = sf > c.kf - 1 ? c.kf - 1 : (pa + 1) / 2;
    return {st, sf, c.kt - 1, pfl, x.t * st - skip, x.f * sf, skip};
}

// ---- reference versions ---------------------------------------------------------

static Tensor conv_ref(const Tensor &x, const Conv &c, const Geometry &g) {
    Tensor y(g.to, g.fo, c.cout);
    for (int t = 0; t < y.t; t++)
        for (int f = 0; f < y.f; f++) {
            float *out = y.at(t, f);
            for (int o = 0; o < c.cout; o++) out[o] = c.b[o];
            for (int kt = 0; kt < c.kt; kt++)
                for (int kf = 0; kf < c.kf; kf++) {
                    const float *in = tap_source(x, g, t, f, kt, kf);
                    if (!in) continue;
                    const float *w = c.k + (size_t)(kt * c.kf + kf) * c.cin * c.cout;
                    for (int i = 0; i < c.cin; i++) {
                        const float xv = in[i];
                        const float *wr = w + (size_t)i * c.cout;
                        for (int o = 0; o < c.cout; o++) out[o] += xv * wr[o];
                    }
                }
        }
    return y;
}

Tensor conv2d_ref(const Tensor &x, const Conv &c, int pl, int pr) {
    return conv_ref(x, c, conv_geometry(x, c, pl, pr, 0));
}

Tensor conv_transpose2d_ref(const Tensor &x, const Conv &c, int st, int sf) {
    return conv_ref(x, c, transpose_geometry(x, c, st, sf, 0));
}

// ---- packing ----------------------------------------------------------------------

constexpr int P = 4;   // output positions per micro-kernel call
constexpr int CT = 16; // output channels per micro-kernel call

void Conv::pack() {
    const int taps = kt * kf;
    packed.resize((size_t)taps * cin * cout);
    size_t i = 0;
    if (cout % CT == 0) {
        // [cout/16][tap][cin][16]: the 16 weights the kernel needs at each
        // step are 64 contiguous bytes, one cache line.
        for (int c0 = 0; c0 < cout; c0 += CT)
            for (int k = 0; k < taps; k++)
                for (int ci = 0; ci < cin; ci++)
                    for (int j = 0; j < CT; j++) packed[i++] = this->k[((size_t)k * cin + ci) * cout + c0 + j];
    } else {
        // [cout][tap][cin]: each output channel's weights for a tap are a
        // contiguous run of cin, for a dot product with the input.
        for (int o = 0; o < cout; o++)
            for (int k = 0; k < taps; k++)
                for (int ci = 0; ci < cin; ci++) packed[i++] = this->k[((size_t)k * cin + ci) * cout + o];
    }
}

// ---- half precision ---------------------------------------------------------------

uint16_t float_to_half(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    const uint32_t sign = (u >> 16) & 0x8000, mant = u & 0x7FFFFF;
    const int exp = (int)((u >> 23) & 0xFF);
    if (exp == 255) return (uint16_t)(sign | 0x7C00 | (mant ? 0x200 : 0)); // inf, NaN
    const int e = exp - 127 + 15;                                       // fp16 exponent
    if (e >= 31) return (uint16_t)(sign | 0x7C00);                      // too big: inf
    if (e <= 0) {                                                       // subnormal or 0
        if (e < -10) return (uint16_t)sign;
        const uint32_t m = mant | 0x800000; // with the implicit leading 1
        const int shift = 14 - e;
        uint32_t h = m >> shift;
        const uint32_t rest = m & ((1u << shift) - 1), half = 1u << (shift - 1);
        if (rest > half || (rest == half && (h & 1))) h++; // round to nearest even
        return (uint16_t)(sign | h);
    }
    uint32_t h = ((uint32_t)e << 10) | (mant >> 13);
    const uint32_t rest = mant & 0x1FFF;
    if (rest > 0x1000 || (rest == 0x1000 && (h & 1))) h++; // a carry may bump the exponent: still right
    return (uint16_t)(sign | h);
}

float half_to_float(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t e = (h >> 10) & 0x1F, m = h & 0x3FF, u;
    if (e == 0) {
        if (m == 0) u = sign;
        else { // subnormal: shift the fraction up until its leading 1 is implicit
            e = 113;
            while (!(m & 0x400)) m <<= 1, e--;
            u = sign | (e << 23) | ((m & 0x3FF) << 13);
        }
    } else if (e == 31) u = sign | 0x7F800000 | (m << 13); // inf, NaN
    else u = sign | ((e + 112) << 23) | (m << 13);
    float x;
    std::memcpy(&x, &u, 4);
    return x;
}

void Conv::pack_half() {
    assert(cout % CT == 0 && !packed.empty());
    packed16.resize(packed.size());
    for (size_t i = 0; i < packed.size(); i++) packed16[i] = float_to_half(packed[i]);
    std::vector<float>().swap(packed); // free the float32 copy
}

// ---- the micro-kernel -------------------------------------------------------------

// The micro-kernels keep their 4 x 16 running sums in arrays indexed by
// small loops (p < 4, j < 4). They only stay in registers if the compiler
// unrolls those loops completely; GCC at -O2 doesn't (it kept the sums on
// the stack: CI measured 38% of peak on Linux vs 70% with MSVC), so ask.
// Clang accepts the same pragma; MSVC unrolls by itself.
#if defined(__GNUC__)
#define MRT2_UNROLL _Pragma("GCC unroll 16")
#else
#define MRT2_UNROLL
#endif

#if defined(MRT2_X86)
// The same tile with AVX2: 8 floats per register, so the 4 x 16 sums are
// 8 registers. Per input channel: load the 16 weights once (2 registers),
// then for each position broadcast its input value to all 8 lanes and do
// 2 fused multiply-adds: 8 FMAs per 2 weight loads.
MRT2_AVX2 static void micro_4x16_avx2(const float *const x[P], const float *w, int cin, float acc[P][CT]) {
    __m256 a[P][2];
    MRT2_UNROLL
    for (int p = 0; p < P; p++) {
        a[p][0] = _mm256_loadu_ps(acc[p]);
        a[p][1] = _mm256_loadu_ps(acc[p] + 8);
    }
    for (int ci = 0; ci < cin; ci++, w += CT) {
        const __m256 w0 = _mm256_loadu_ps(w), w1 = _mm256_loadu_ps(w + 8);
        MRT2_UNROLL
        for (int p = 0; p < P; p++) {
            const __m256 xv = _mm256_broadcast_ss(x[p] + ci);
            a[p][0] = _mm256_fmadd_ps(xv, w0, a[p][0]);
            a[p][1] = _mm256_fmadd_ps(xv, w1, a[p][1]);
        }
    }
    MRT2_UNROLL
    for (int p = 0; p < P; p++) {
        _mm256_storeu_ps(acc[p], a[p][0]);
        _mm256_storeu_ps(acc[p] + 8, a[p][1]);
    }
}
#endif

// acc[p][0..15] += sum_ci x[p][ci] * w[ci][0..15], for P = 4 positions.
// The 4 x 16 = 64 running sums live in 16 vector registers for the whole
// loop; each 16-weight vector loaded feeds 4 fused multiply-adds (one per
// position) and each input value feeds 16 (one per channel).
static inline void micro_4x16(const float *const x[P], const float *w, int cin, float acc[P][CT]) {
#if defined(__ARM_NEON) || defined(_M_ARM64)
    float32x4_t a[P][4];
    MRT2_UNROLL
    for (int p = 0; p < P; p++)
        MRT2_UNROLL
        for (int j = 0; j < 4; j++) a[p][j] = vld1q_f32(acc[p] + 4 * j);
    for (int ci = 0; ci < cin; ci++, w += CT) {
        const float32x4_t w0 = vld1q_f32(w), w1 = vld1q_f32(w + 4);
        const float32x4_t w2 = vld1q_f32(w + 8), w3 = vld1q_f32(w + 12);
        MRT2_UNROLL
        for (int p = 0; p < P; p++) {
            const float xv = x[p][ci];
            a[p][0] = vfmaq_n_f32(a[p][0], w0, xv);
            a[p][1] = vfmaq_n_f32(a[p][1], w1, xv);
            a[p][2] = vfmaq_n_f32(a[p][2], w2, xv);
            a[p][3] = vfmaq_n_f32(a[p][3], w3, xv);
        }
    }
    MRT2_UNROLL
    for (int p = 0; p < P; p++)
        MRT2_UNROLL
        for (int j = 0; j < 4; j++) vst1q_f32(acc[p] + 4 * j, a[p][j]);
#else
#if defined(MRT2_X86)
    static const bool avx2 = cpu_avx2(); // asked once; the branch then always goes the same way
    if (avx2) return micro_4x16_avx2(x, w, cin, acc);
#endif
    // Portable version.
    for (int ci = 0; ci < cin; ci++, w += CT)
        MRT2_UNROLL
        for (int p = 0; p < P; p++) {
            const float xv = x[p][ci];
            for (int j = 0; j < CT; j++) acc[p][j] += xv * w[j];
        }
#endif
}

// The same tile with half-precision weights: 16 halves are loaded and
// converted to 16 floats (one instruction per 4 or 8), then the very same
// multiply-adds. Used by the layers limited by reading their weights.
#if defined(MRT2_X86)
MRT2_AVX2 static void micro_4x16_avx2_h(const float *const x[P], const uint16_t *w, int cin, float acc[P][CT]) {
    __m256 a[P][2];
    MRT2_UNROLL
    for (int p = 0; p < P; p++) {
        a[p][0] = _mm256_loadu_ps(acc[p]);
        a[p][1] = _mm256_loadu_ps(acc[p] + 8);
    }
    for (int ci = 0; ci < cin; ci++, w += CT) {
        const __m256 w0 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(w)));     // F16C
        const __m256 w1 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(w + 8)));
        MRT2_UNROLL
        for (int p = 0; p < P; p++) {
            const __m256 xv = _mm256_broadcast_ss(x[p] + ci);
            a[p][0] = _mm256_fmadd_ps(xv, w0, a[p][0]);
            a[p][1] = _mm256_fmadd_ps(xv, w1, a[p][1]);
        }
    }
    MRT2_UNROLL
    for (int p = 0; p < P; p++) {
        _mm256_storeu_ps(acc[p], a[p][0]);
        _mm256_storeu_ps(acc[p] + 8, a[p][1]);
    }
}
#endif

static inline void micro_4x16(const float *const x[P], const uint16_t *w, int cin, float acc[P][CT]) {
#if defined(__ARM_NEON) || defined(_M_ARM64)
    float32x4_t a[P][4];
    MRT2_UNROLL
    for (int p = 0; p < P; p++)
        MRT2_UNROLL
        for (int j = 0; j < 4; j++) a[p][j] = vld1q_f32(acc[p] + 4 * j);
    for (int ci = 0; ci < cin; ci++, w += CT) {
        const uint16x8_t h0 = vld1q_u16(w), h1 = vld1q_u16(w + 8);
        const float32x4_t w0 = vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(h0)));
        const float32x4_t w1 = vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(h0)));
        const float32x4_t w2 = vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(h1)));
        const float32x4_t w3 = vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(h1)));
        MRT2_UNROLL
        for (int p = 0; p < P; p++) {
            const float xv = x[p][ci];
            a[p][0] = vfmaq_n_f32(a[p][0], w0, xv);
            a[p][1] = vfmaq_n_f32(a[p][1], w1, xv);
            a[p][2] = vfmaq_n_f32(a[p][2], w2, xv);
            a[p][3] = vfmaq_n_f32(a[p][3], w3, xv);
        }
    }
    MRT2_UNROLL
    for (int p = 0; p < P; p++)
        MRT2_UNROLL
        for (int j = 0; j < 4; j++) vst1q_f32(acc[p] + 4 * j, a[p][j]);
#else
#if defined(MRT2_X86)
    static const bool avx2 = cpu_avx2();
    if (avx2) return micro_4x16_avx2_h(x, w, cin, acc);
#endif
    for (int ci = 0; ci < cin; ci++, w += CT) {
        float wf[CT];
        for (int j = 0; j < CT; j++) wf[j] = half_to_float(w[j]);
        MRT2_UNROLL
        for (int p = 0; p < P; p++) {
            const float xv = x[p][ci];
            for (int j = 0; j < CT; j++) acc[p][j] += xv * wf[j];
        }
    }
#endif
}

// ---- few output channels: one long dot product per output ----------------------

// sum over j < n of x[j] . w[tap[j]] (each a run of cin floats), with one set
// of 16 running sums for the whole thing and a single horizontal add at the
// end. (Calling dot() per tap would pay that final add 49 times per output
// of the 7x7 layer, as much work as the multiplications themselves.)
// Element i of a run goes to sum i % 16 (cin is a multiple of 16).
#if defined(__aarch64__) || defined(_M_ARM64)
static float tap_dot(const float *const x[], const int tap[], int n, const float *w, int cin) {
    float32x4_t a0 = vdupq_n_f32(0), a1 = a0, a2 = a0, a3 = a0; // sums 0-3, 4-7, 8-11, 12-15
    for (int j = 0; j < n; j++) {
        const float *a = x[j], *b = w + (size_t)tap[j] * cin;
        for (int i = 0; i < cin; i += 16) {
            a0 = vfmaq_f32(a0, vld1q_f32(a + i), vld1q_f32(b + i));
            a1 = vfmaq_f32(a1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
            a2 = vfmaq_f32(a2, vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
            a3 = vfmaq_f32(a3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
        }
    }
    return vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
}
#else
static float tap_dot_plain(const float *const x[], const int tap[], int n, const float *w, int cin) {
    float acc[16] = {};
    for (int j = 0; j < n; j++) {
        const float *a = x[j], *b = w + (size_t)tap[j] * cin;
        for (int i = 0; i < cin; i += 16)
            for (int k = 0; k < 16; k++) acc[k] += a[i + k] * b[i + k];
    }
    float s = 0.0f;
    for (int k = 0; k < 16; k++) s += acc[k];
    return s;
}

#if defined(MRT2_X86)
MRT2_AVX2 static float tap_dot_avx2(const float *const x[], const int tap[], int n, const float *w, int cin) {
    __m256 a0 = _mm256_setzero_ps(), a1 = a0; // sums 0-7, 8-15
    for (int j = 0; j < n; j++) {
        const float *a = x[j], *b = w + (size_t)tap[j] * cin;
        for (int i = 0; i < cin; i += 16) {
            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), a0);
            a1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), a1);
        }
    }
    const __m256 t = _mm256_add_ps(a0, a1);
    __m128 h = _mm_add_ps(_mm256_castps256_ps128(t), _mm256_extractf128_ps(t, 1));
    h = _mm_add_ps(h, _mm_movehl_ps(h, h));
    return _mm_cvtss_f32(_mm_add_ss(h, _mm_shuffle_ps(h, h, 1)));
}
#endif
static float tap_dot(const float *const x[], const int tap[], int n, const float *w, int cin) {
#if defined(MRT2_X86)
    static const bool avx2 = cpu_avx2();
    if (avx2) return tap_dot_avx2(x, tap, n, w, cin);
#endif
    return tap_dot_plain(x, tap, n, w, cin);
}
#endif

// Inputs that fall on padding point here, so every tap of a block can be
// computed the same way (zeros add nothing). Large enough for any cin.
static const float kZeros[4096] = {};

// ---- the fast kernel --------------------------------------------------------------

// One "item" of work: a block of 4 same-phase output positions x one tile
// of 16 output channels. Computes it and stores it.
static void conv_item(const Tensor &x, const Conv &c, const Geometry &g, Tensor &y, int pt, int pf, int nf,
                      int count, int block, int tile) {
    const int taps = c.kt * c.kf;
    const int i0 = block * P, np = std::min(P, count - i0);
    // Positions past the end of the class repeat the last one (computed, not stored).
    int t[P], f[P];
    for (int p = 0; p < P; p++) {
        const int i = i0 + std::min(p, np - 1);
        t[p] = pt + g.st * (i / nf);
        f[p] = pf + g.sf * (i % nf);
    }
    const int c0 = tile * CT;
    float acc[P][CT] = {};
    const bool half = !c.packed16.empty(); // fp16 weights (same layout) instead of float32
    const size_t wt = (size_t)tile * taps * c.cin * CT;
    for (int kt = 0; kt < c.kt; kt++)
        for (int kf = 0; kf < c.kf; kf++) {
            const float *in[P];
            bool any = false;
            for (int p = 0; p < P; p++) {
                in[p] = tap_source(x, g, t[p], f[p], kt, kf);
                any |= in[p] != nullptr;
                if (!in[p]) in[p] = kZeros;
            }
            if (!any) continue; // this tap is padding for all 4
            const size_t at = wt + (size_t)(kt * c.kf + kf) * c.cin * CT;
            if (half) micro_4x16(in, &c.packed16[at], c.cin, acc);
            else micro_4x16(in, &c.packed[at], c.cin, acc);
        }
    for (int p = 0; p < np; p++) {
        float *out = y.at(t[p], f[p]) + c0;
        for (int j = 0; j < CT; j++) out[j] = acc[p][j] + c.b[c0 + j];
    }
}

static Tensor conv_core(const Tensor &x, const Conv &c, const Geometry &g) {
    assert(c.cin <= 4096 && (!c.packed.empty() || !c.packed16.empty())); // kZeros size; pack() called
    Tensor y(g.to, g.fo, c.cout);
    const int taps = c.kt * c.kf;
    const int threads = pool().size();

    if (c.cout % CT) {
        // Few output channels: each output is one long dot product. The
        // taps' input positions are found once per position, then every
        // output channel runs over them. Split by positions.
        const int positions = y.t * y.f;
        const int chunks = threads == 1 ? 1 : std::min(positions, threads * 8);
        pool().run(chunks, [&](int ch) {
            const float *src[64];
            int tap[64];
            assert(taps <= 64);
            for (int pos = positions * ch / chunks; pos < positions * (ch + 1) / chunks; pos++) {
                const int t = pos / y.f, f = pos % y.f;
                int n = 0;
                for (int k = 0; k < taps; k++)
                    if (const float *in = tap_source(x, g, t, f, k / c.kf, k % c.kf)) {
                        src[n] = in;
                        tap[n++] = k;
                    }
                for (int o = 0; o < c.cout; o++) {
                    const float *wo = &c.packed[(size_t)o * taps * c.cin];
                    float s;
                    if (c.cin % 16 == 0) s = tap_dot(src, tap, n, wo, c.cin);
                    else { // any other size (not in this model): dot() per tap
                        s = 0.0f;
                        for (int j = 0; j < n; j++) s += dot(src[j], wo + (size_t)tap[j] * c.cin, c.cin);
                    }
                    y.at(t, f)[o] = c.b[o] + s;
                }
            }
        });
        return y;
    }

    // Outputs with the same phase (t % st, f % sf) use the same taps, so
    // they are grouped 4 at a time within one phase class. The work is a
    // flat list of items (phase class, block of 4, tile of 16 channels);
    // each thread takes a contiguous range of it. Every output belongs to
    // exactly one item: the same bits for any number of threads.
    struct Phase { int pt, pf, nf, count, blocks; long first; };
    Phase ph[16];
    int nph = 0;
    long items = 0;
    const int tiles = c.cout / CT;
    for (int pt = 0; pt < g.st; pt++)
        for (int pf = 0; pf < g.sf; pf++) {
            const int nt = (y.t - pt + g.st - 1) / g.st, nf = (y.f - pf + g.sf - 1) / g.sf;
            const int count = nt * nf, blocks = (count + P - 1) / P;
            ph[nph++] = {pt, pf, nf, count, blocks, items};
            items += (long)blocks * tiles;
        }
    assert(nph <= 16);
    const int chunks = threads == 1 ? 1 : (int)std::min<long>(items, threads * 8);
    pool().run(chunks, [&](int ch) {
        int k = 0;
        for (long it = items * ch / chunks; it < items * (ch + 1) / chunks; it++) {
            while (k + 1 < nph && it >= ph[k + 1].first) k++; // items are in phase order
            const long local = it - ph[k].first;
            conv_item(x, c, g, y, ph[k].pt, ph[k].pf, ph[k].nf, ph[k].count, (int)(local / tiles),
                      (int)(local % tiles));
        }
    });
    return y;
}

// ---- Winograd F(2x2, 3x3) ------------------------------------------------------------
//
// A 2x2 block of outputs needs a 4x4 patch d of inputs. With
//   U = G g G^T   (the 3x3 weights g -> 4x4, once at load time)
//   V = B^T d B   (the input patch -> 4x4, additions only)
//   Y = A^T (U . V) A   (element-wise product, then 4x4 -> 2x2, additions only)
// Y is exactly the 2x2 conv output (Lavin & Gray, arXiv 1509.09308), with
//   G = [1 0 0; 1/2 1/2 1/2; 1/2 -1/2 1/2; 0 0 1]
//   B^T = [1 0 -1 0; 0 1 1 0; 0 -1 1 0; 0 1 0 -1]
//   A^T = [1 1 1 0; 0 1 -1 -1].
// Summed over input channels, "U . V" is 16 independent matrix products
// (one per position ij of the 4x4 grid): M[ij] = V[ij] (tiles x cin) times
// U[ij] (cin x cout), done by the same micro-kernel as the direct conv.
// 16 multiplies per 4 outputs instead of 36: 2.25x fewer.

void Conv::pack_winograd() {
    assert(kt == 3 && kf == 3 && cin % CT == 0 && cout % CT == 0);
    wino.assign((size_t)16 * cin * cout, 0.0f);
    const double G[4][3] = {{1, 0, 0}, {0.5, 0.5, 0.5}, {0.5, -0.5, 0.5}, {0, 0, 1}};
    for (int ci = 0; ci < cin; ci++)
        for (int co = 0; co < cout; co++) {
            double g[3][3], t[4][3];
            for (int a = 0; a < 3; a++)
                for (int b = 0; b < 3; b++) g[a][b] = k[((size_t)(a * 3 + b) * cin + ci) * cout + co];
            for (int i = 0; i < 4; i++) // t = G g
                for (int b = 0; b < 3; b++) t[i][b] = G[i][0] * g[0][b] + G[i][1] * g[1][b] + G[i][2] * g[2][b];
            for (int i = 0; i < 4; i++) // U = t G^T
                for (int j = 0; j < 4; j++) {
                    const double u = t[i][0] * G[j][0] + t[i][1] * G[j][1] + t[i][2] * G[j][2];
                    wino[(((size_t)(co / CT) * 16 + i * 4 + j) * cin + ci) * CT + co % CT] = (float)u;
                }
        }
}

static Tensor conv_winograd(const Tensor &x, const Conv &c, int skip, const char *label) {
    const int rows = x.t - skip, F = x.f, ci = c.cin, co = c.cout;
    assert(rows % 2 == 0 && skip % 2 == 0);
    const int tf = (F + 1) / 2, tiles = rows / 2 * tf; // 2x2 output tiles
    // 16 multiplies per tile per (input, output) channel pair, + transforms
    MRT2_PROF(label, 2.0 * 16 * tiles * ci * co, 4.0 * (16.0 * ci * co + x.v.size() + (double)rows * F * co));
    Tensor y(rows, F, co);
    // Scratch: V [16][tiles][cin], M [16][tiles][cout]. Only the generation
    // thread runs convs, so buffers that only grow are enough.
    static std::vector<float> Vb, Mb;
    if (Vb.size() < (size_t)16 * tiles * ci) Vb.resize((size_t)16 * tiles * ci);
    if (Mb.size() < (size_t)16 * tiles * co) Mb.resize((size_t)16 * tiles * co);
    float *V = Vb.data(), *M = Mb.data();
    const int chunks = std::min(tiles, pool().size() * 4);

    // 1. Input transform. Output rows (2r, 2r+1) [+ skip] read input rows
    //    2r + skip - 2 .. 2r + skip + 1 (causal); columns 2c-1 .. 2c+2.
    pool().run(chunks, [&](int ch) {
        for (int tl = tiles * ch / chunks; tl < tiles * (ch + 1) / chunks; tl++) {
            const int r = tl / tf, cc = tl % tf;
            const float *d[4][4];
            for (int a = 0; a < 4; a++)
                for (int b = 0; b < 4; b++) {
                    const int t = 2 * r + skip - 2 + a, f = 2 * cc - 1 + b;
                    d[a][b] = (t >= 0 && t < x.t && f >= 0 && f < F) ? x.at(t, f) : kZeros;
                }
            float *v[16];
            for (int ij = 0; ij < 16; ij++) v[ij] = V + ((size_t)ij * tiles + tl) * ci;
            for (int k = 0; k < ci; k++) {
                float t[4][4], u[4][4];
                for (int b = 0; b < 4; b++) { // t = B^T d (combine rows)
                    t[0][b] = d[0][b][k] - d[2][b][k];
                    t[1][b] = d[1][b][k] + d[2][b][k];
                    t[2][b] = d[2][b][k] - d[1][b][k];
                    t[3][b] = d[1][b][k] - d[3][b][k];
                }
                for (int a = 0; a < 4; a++) { // u = t B (combine columns)
                    u[a][0] = t[a][0] - t[a][2];
                    u[a][1] = t[a][1] + t[a][2];
                    u[a][2] = t[a][2] - t[a][1];
                    u[a][3] = t[a][1] - t[a][3];
                }
                for (int ij = 0; ij < 16; ij++) v[ij][k] = u[ij / 4][ij % 4];
            }
        }
    });

    // 2. The 16 matrix products, 4 tiles x 16 output channels per item.
    const int blocks = (tiles + P - 1) / P, ctiles = co / CT;
    const long items = 16L * blocks * ctiles;
    const int gchunks = (int)std::min<long>(items, pool().size() * 8);
    pool().run(gchunks, [&](int ch) {
        for (long it = items * ch / gchunks; it < items * (ch + 1) / gchunks; it++) {
            const int ij = (int)(it / ((long)blocks * ctiles)), rest = (int)(it % ((long)blocks * ctiles));
            const int blk = rest / ctiles, ct = rest % ctiles;
            const int t0 = blk * P, np = std::min(P, tiles - t0);
            const float *in[P];
            for (int p = 0; p < P; p++) in[p] = V + ((size_t)ij * tiles + t0 + std::min(p, np - 1)) * ci;
            float acc[P][CT] = {};
            micro_4x16(in, &c.wino[((size_t)ct * 16 + ij) * ci * CT], ci, acc);
            for (int p = 0; p < np; p++)
                std::copy(acc[p], acc[p] + CT, M + ((size_t)ij * tiles + t0 + p) * co + ct * CT);
        }
    });

    // 3. Output transform Y = A^T M A, plus the bias.
    pool().run(chunks, [&](int ch) {
        for (int tl = tiles * ch / chunks; tl < tiles * (ch + 1) / chunks; tl++) {
            const int r = tl / tf, cc = tl % tf;
            const float *m[16];
            for (int ij = 0; ij < 16; ij++) m[ij] = M + ((size_t)ij * tiles + tl) * co;
            float *o00 = y.at(2 * r, 2 * cc), *o10 = y.at(2 * r + 1, 2 * cc);
            const bool right = 2 * cc + 1 < F; // the last tile of an odd F has one column
            for (int k = 0; k < co; k++) {
                float s0[4], s1[4];
                for (int b = 0; b < 4; b++) { // A^T m (combine rows)
                    s0[b] = m[b][k] + m[4 + b][k] + m[8 + b][k];
                    s1[b] = m[4 + b][k] - m[8 + b][k] - m[12 + b][k];
                }
                o00[k] = c.b[k] + (s0[0] + s0[1] + s0[2]);
                o10[k] = c.b[k] + (s1[0] + s1[1] + s1[2]);
                if (right) {
                    o00[co + k] = c.b[k] + (s0[1] - s0[2] - s0[3]);
                    o10[co + k] = c.b[k] + (s1[1] - s1[2] - s1[3]);
                }
            }
        }
    });
    return y;
}

Tensor conv2d(const Tensor &x, const Conv &c, int pl, int pr, const char *label, int skip) {
    if (!c.wino.empty()) {
        assert(pl == 1 && pr == 1);
        return conv_winograd(x, c, skip, label);
    }
    const Geometry g = conv_geometry(x, c, pl, pr, skip);
    MRT2_PROF(label, 2.0 * g.to * g.fo * c.kt * c.kf * c.cin * c.cout,
              (c.packed16.empty() ? 4.0 : 2.0) * c.kt * c.kf * c.cin * c.cout +
                  4.0 * (x.v.size() + (double)g.to * g.fo * c.cout));
    return conv_core(x, c, g);
}

Tensor conv_transpose2d(const Tensor &x, const Conv &c, int st, int sf, const char *label, int skip) {
    const Geometry g = transpose_geometry(x, c, st, sf, skip);
    // each output gets (kt/st) x (kf/sf) real taps: no work on inserted zeros
    MRT2_PROF(label, 2.0 * g.to * g.fo * (c.kt / st) * (c.kf / sf) * c.cin * c.cout,
              (c.packed16.empty() ? 4.0 : 2.0) * c.kt * c.kf * c.cin * c.cout +
                  4.0 * (x.v.size() + (double)g.to * g.fo * c.cout));
    return conv_core(x, c, g);
}

} // namespace mrt2
