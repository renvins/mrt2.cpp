// conv.h - the codec's 2D convolutions over (time, frequency, channel).
//
// Two versions of each:
//   *_ref : plain loops, obviously correct; the reference for tests.
//   fast  : the same results, 2-3x faster. Weights are "packed" once at
//           load time so a register-blocked micro-kernel can compute a
//           tile of 4 output positions x 16 output channels at a time:
//           each weight loaded from memory then feeds 4 multiply-adds
//           instead of 1 (the core idea of every fast conv/GEMM library).
//
// Both convs are written in "gather" form: every output position sums,
// over the kernel taps, (the input position that tap reads) x (the tap's
// weights). A regular conv is the special case stride 1 of the transposed
// one, so one kernel serves both:
//   conv2d            stride 1; time causal (kt-1 zeros of past); freq pads pl, pr
//   conv_transpose2d  stride (st, sf), as Google computes it: zeros between
//                     input samples, causal time / "same" freq padding,
//                     kernel not flipped; only taps on real samples count.
#pragma once
#include <vector>

namespace mrt2 {

// Activations: [time][freq][channel], channels contiguous (like JAX).
struct Tensor {
    int t = 0, f = 0, c = 0;
    std::vector<float> v;
    Tensor() = default;
    Tensor(int t_, int f_, int c_) : t(t_), f(f_), c(c_), v((size_t)t_ * f_ * c_, 0.0f) {}
    float *at(int ti, int fi) { return v.data() + ((size_t)ti * f + fi) * c; }
    const float *at(int ti, int fi) const { return v.data() + ((size_t)ti * f + fi) * c; }
};

// One convolution: kernel [kt][kf][cin][cout] (the checkpoint's layout,
// read straight from the mapped file), bias [cout], and its packed copy.
struct Conv {
    const float *k = nullptr, *b = nullptr;
    int kt = 0, kf = 0, cin = 0, cout = 0;
    std::vector<float> packed;
    // Fill `packed` (call once after setting the fields above):
    //   cout a multiple of 16: [cout/16][kt*kf][cin][16] for the micro-kernel;
    //   otherwise (the 2-channel output layer): [cout][kt*kf][cin], so each
    //   output is one long dot product.
    void pack();

    // Winograd form of a 3x3 conv's weights, [cout/16][16][cin][16]; empty
    // unless pack_winograd() was called. When present, conv2d() computes
    // this layer with Winograd F(2x2, 3x3) (see conv.cpp): 16 multiplies per
    // 2x2 outputs instead of 36. A per-layer choice made at load time, so a
    // layer is always computed the same way (offline and streaming give the
    // same bits). Needs: 3x3, padding 1/1, cin and cout multiples of 16, and
    // an even number of output rows (with an even `skip`).
    std::vector<float> wino;
    void pack_winograd();
};

// Reference versions: plain loops, for tests.
Tensor conv2d_ref(const Tensor &x, const Conv &c, int pl, int pr);
Tensor conv_transpose2d_ref(const Tensor &x, const Conv &c, int st, int sf);

// Fast versions (need c.pack()). `label` names the layer in the profiler.
// `skip`: don't compute the first `skip` output time steps; the result
// starts at step `skip` (for streaming, where the first input rows are
// only the remembered past). For the transposed conv, a multiple of st.
Tensor conv2d(const Tensor &x, const Conv &c, int pl, int pr, const char *label = "conv", int skip = 0);
Tensor conv_transpose2d(const Tensor &x, const Conv &c, int st, int sf, const char *label = "convT",
                        int skip = 0);

} // namespace mrt2
