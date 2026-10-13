#include "conv.h"
#include "ops.h"
#include "profile.h"
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

struct Geometry {
    int st, sf;   // strides
    int pt0, pf0; // left padding, time and frequency
    int to, fo;   // output size
};

static inline const float *tap_source(const Tensor &x, const Geometry &g, int t, int f, int kt, int kf) {
    const int pt = t + kt - g.pt0, pf = f + kf - g.pf0;
    if (g.st == 1 && g.sf == 1) { // regular conv: no divisions needed (they cost ~10 cycles each)
        if (pt < 0 || pf < 0 || pt >= x.t || pf >= x.f) return nullptr;
        return x.at(pt, pf);
    }
    if (pt < 0 || pf < 0 || pt % g.st || pf % g.sf) return nullptr;
    const int ti = pt / g.st, fi = pf / g.sf;
    if (ti >= x.t || fi >= x.f) return nullptr;
    return x.at(ti, fi);
}

static Geometry conv_geometry(const Tensor &x, const Conv &c, int pl, int pr) {
    return {1, 1, c.kt - 1, pl, x.t, x.f + pl + pr - c.kf + 1};
}

static Geometry transpose_geometry(const Tensor &x, const Conv &c, int st, int sf) {
    // Google's paddings: causal in time, "same" in frequency (see plan 2.6).
    const int pa = c.kf + sf - 2;
    const int pfl = sf > c.kf - 1 ? c.kf - 1 : (pa + 1) / 2;
    return {st, sf, c.kt - 1, pfl, x.t * st, x.f * sf};
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
    return conv_ref(x, c, conv_geometry(x, c, pl, pr));
}

Tensor conv_transpose2d_ref(const Tensor &x, const Conv &c, int st, int sf) {
    return conv_ref(x, c, transpose_geometry(x, c, st, sf));
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
    // Portable version (an AVX2 one comes with x86 support).
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

static Tensor conv_core(const Tensor &x, const Conv &c, const Geometry &g) {
    assert(c.cin <= 4096 && !c.packed.empty()); // kZeros size; pack() called
    Tensor y(g.to, g.fo, c.cout);
    const int taps = c.kt * c.kf;

    if (c.cout % CT) {
        // Few output channels: each output is one long dot product. The
        // taps' input positions are found once per position, then every
        // output channel runs over them.
        std::vector<const float *> src(taps);
        std::vector<int> tap(taps);
        for (int t = 0; t < y.t; t++)
            for (int f = 0; f < y.f; f++) {
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
        return y;
    }

    // Outputs with the same phase (t % st, f % sf) use the same taps, so
    // they are grouped 4 at a time within one phase class. Positions past
    // the end of a class repeat the last one (computed, not stored).
    for (int pt = 0; pt < g.st; pt++)
        for (int pf = 0; pf < g.sf; pf++) {
            const int nt = (y.t - pt + g.st - 1) / g.st, nf = (y.f - pf + g.sf - 1) / g.sf;
            const int count = nt * nf;
            for (int i0 = 0; i0 < count; i0 += P) {
                const int np = std::min(P, count - i0);
                int t[P], f[P];
                for (int p = 0; p < P; p++) {
                    const int i = i0 + std::min(p, np - 1);
                    t[p] = pt + g.st * (i / nf);
                    f[p] = pf + g.sf * (i % nf);
                }
                for (int c0 = 0; c0 < c.cout; c0 += CT) {
                    float acc[P][CT] = {};
                    const float *wt = &c.packed[(size_t)(c0 / CT) * taps * c.cin * CT];
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
            }
        }
    return y;
}

Tensor conv2d(const Tensor &x, const Conv &c, int pl, int pr, const char *label) {
    const Geometry g = conv_geometry(x, c, pl, pr);
    MRT2_PROF(label, 2.0 * g.to * g.fo * c.kt * c.kf * c.cin * c.cout,
              4.0 * ((double)c.kt * c.kf * c.cin * c.cout + x.v.size() + (double)g.to * g.fo * c.cout));
    return conv_core(x, c, g);
}

Tensor conv_transpose2d(const Tensor &x, const Conv &c, int st, int sf, const char *label) {
    const Geometry g = transpose_geometry(x, c, st, sf);
    // each output gets (kt/st) x (kf/sf) real taps: no work on inserted zeros
    MRT2_PROF(label, 2.0 * g.to * g.fo * (c.kt / st) * (c.kf / sf) * c.cin * c.cout,
              4.0 * ((double)c.kt * c.kf * c.cin * c.cout + x.v.size() + (double)g.to * g.fo * c.cout));
    return conv_core(x, c, g);
}

} // namespace mrt2
