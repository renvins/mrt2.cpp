#include "conv.h"
#include "cpu.h"
#include "ops.h"
#include "profile.h"
#include "threads.h"
#include <algorithm>
#include <cassert>
#if defined(__ARM_NEON)
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

// ---- the micro-kernel -------------------------------------------------------------

#if defined(MRT2_X86)
// The same tile with AVX2: 8 floats per register, so the 4 x 16 sums are
// 8 registers. Per input channel: load the 16 weights once (2 registers),
// then for each position broadcast its input value to all 8 lanes and do
// 2 fused multiply-adds: 8 FMAs per 2 weight loads.
MRT2_AVX2 static void micro_4x16_avx2(const float *const x[P], const float *w, int cin, float acc[P][CT]) {
    __m256 a[P][2];
    for (int p = 0; p < P; p++) {
        a[p][0] = _mm256_loadu_ps(acc[p]);
        a[p][1] = _mm256_loadu_ps(acc[p] + 8);
    }
    for (int ci = 0; ci < cin; ci++, w += CT) {
        const __m256 w0 = _mm256_loadu_ps(w), w1 = _mm256_loadu_ps(w + 8);
        for (int p = 0; p < P; p++) {
            const __m256 xv = _mm256_broadcast_ss(x[p] + ci);
            a[p][0] = _mm256_fmadd_ps(xv, w0, a[p][0]);
            a[p][1] = _mm256_fmadd_ps(xv, w1, a[p][1]);
        }
    }
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
#if defined(MRT2_X86)
    static const bool avx2 = cpu_avx2(); // asked once; the branch then always goes the same way
    if (avx2) return micro_4x16_avx2(x, w, cin, acc);
#endif
    // Portable version.
    for (int ci = 0; ci < cin; ci++, w += CT)
        for (int p = 0; p < P; p++) {
            const float xv = x[p][ci];
            for (int j = 0; j < CT; j++) acc[p][j] += xv * w[j];
        }
#endif
}

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
    const float *wt = &c.packed[(size_t)tile * taps * c.cin * CT];
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
            micro_4x16(in, wt + (size_t)(kt * c.kf + kf) * c.cin * CT, c.cin, acc);
        }
    for (int p = 0; p < np; p++) {
        float *out = y.at(t[p], f[p]) + c0;
        for (int j = 0; j < CT; j++) out[j] = acc[p][j] + c.b[c0 + j];
    }
}

static Tensor conv_core(const Tensor &x, const Conv &c, const Geometry &g) {
    assert(c.cin <= 4096 && !c.packed.empty()); // kZeros size; pack() called
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
                    float s = c.b[o];
                    const float *wo = &c.packed[(size_t)o * taps * c.cin];
                    for (int j = 0; j < n; j++) s += dot(src[j], wo + (size_t)tap[j] * c.cin, c.cin);
                    y.at(t, f)[o] = s;
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

Tensor conv2d(const Tensor &x, const Conv &c, int pl, int pr, const char *label, int skip) {
    const Geometry g = conv_geometry(x, c, pl, pr, skip);
    MRT2_PROF(label, 2.0 * g.to * g.fo * c.kt * c.kf * c.cin * c.cout,
              4.0 * ((double)c.kt * c.kf * c.cin * c.cout + x.v.size() + (double)g.to * g.fo * c.cout));
    return conv_core(x, c, g);
}

Tensor conv_transpose2d(const Tensor &x, const Conv &c, int st, int sf, const char *label, int skip) {
    const Geometry g = transpose_geometry(x, c, st, sf, skip);
    // each output gets (kt/st) x (kf/sf) real taps: no work on inserted zeros
    MRT2_PROF(label, 2.0 * g.to * g.fo * (c.kt / st) * (c.kf / sf) * c.cin * c.cout,
              4.0 * ((double)c.kt * c.kf * c.cin * c.cout + x.v.size() + (double)g.to * g.fo * c.cout));
    return conv_core(x, c, g);
}

} // namespace mrt2
