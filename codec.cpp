#include "codec.h"
#include "ops.h"
#include "profile.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>

namespace mrt2 {

// ---- loading ----------------------------------------------------------------

static Conv load_conv(const Weights &w, const std::string &name, int kt, int kf, int cin, int cout) {
    const std::string p = "params/soundstream/decoder/" + name + "/conv/";
    Conv c;
    c.k = w.f32(p + "kernel", {kt, kf, cin, cout});
    c.b = w.f32(p + "bias", {cout});
    c.kt = kt;
    c.kf = kf;
    c.cin = cin;
    c.cout = cout;
    c.pack(); // reorder once for the fast conv kernel (conv.h)
    return c;
}

CodecWeights load_codec(const Weights &w) {
    CodecWeights cw;
    cw.codebooks = w.f32("params/soundstream/quantizer/embedding", {64, 1024, 256});
    cw.in_first = load_conv(w, "input_layer/conv1x1_first", 1, 1, 256, 2560);
    cw.in_b1 = load_conv(w, "input_layer/shortcut_layer/conv1x1_b1", 1, 1, 256, 2560);
    cw.in_b2 = load_conv(w, "input_layer/shortcut_layer/conv1x1_b2", 1, 1, 2560, 2560);
    cw.res_a = load_conv(w, "input_layers_residual_unit/conv2d_3x3_a", 3, 3, 512, 512);
    cw.res_b = load_conv(w, "input_layers_residual_unit/conv2d_3x3", 3, 3, 512, 512);
    // The 7 upsampling blocks: (transposed-conv kernel, stride, channels).
    struct Spec { int kt, kf, st, sf, cin, cout; };
    const Spec spec[7] = {{4, 3, 2, 1, 512, 1024}, {4, 4, 2, 2, 512, 256}, {3, 4, 1, 2, 256, 256},
                          {3, 4, 1, 2, 256, 256},  {3, 6, 1, 3, 256, 128}, {3, 4, 1, 2, 128, 128},
                          {3, 4, 1, 2, 128, 64}};
    for (int i = 0; i < 7; i++) {
        const Spec &s = spec[i];
        const std::string b = "decoder_" + std::to_string(i) + "/";
        CodecWeights::Block &B = cw.block[i];
        B.up = load_conv(w, b + "conv2dtranspose_" + std::to_string(s.kt) + "x" + std::to_string(s.kf),
                         s.kt, s.kf, s.cin, s.cout);
        B.conv = load_conv(w, b + "conv2d_3x3", 3, 3, s.cout, s.cout);
        if (s.cin != s.cout) B.shortcut = load_conv(w, b + "shortcut_layer/conv1x1", 1, 1, s.cin, s.cout);
        B.st = s.st;
        B.sf = s.sf;
    }
    cw.out = load_conv(w, "output_layer/base_conv_last", 7, 7, 64, 2);
    return cw;
}

void dequantize(const CodecWeights &cw, const int32_t tokens[12], float z[256]) {
    // Token id 6 + level * 1024 + code -> code; then one vector per level.
    for (int i = 0; i < 256; i++) z[i] = 0.0f;
    for (int k = 0; k < 12; k++) {
        const int code = (tokens[k] - 6) % 1024;
        const float *row = cw.codebooks + ((size_t)k * 1024 + code) * 256;
        for (int i = 0; i < 256; i++) z[i] += row[i];
    }
}

// ---- the decoder ------------------------------------------------------------------
//
// The convolutions themselves live in conv.h/.cpp.

static Tensor elu_of(Tensor x) {
    MRT2_PROF("elu", (double)x.v.size(), 8.0 * x.v.size());
    elu(x.v.data(), x.v.size());
    return x;
}

// Labels for the profiler, per block.
static const char *const kUpLabel[7] = {"d0.convT", "d1.convT", "d2.convT", "d3.convT", "d4.convT", "d5.convT", "d6.convT"};
static const char *const kConvLabel[7] = {"d0.3x3", "d1.3x3", "d2.3x3", "d3.3x3", "d4.3x3", "d5.3x3", "d6.3x3"};
static const char *const kShortLabel[7] = {"d0.1x1", "d1.1x1", "d2.1x1", "d3.1x1", "d4.1x1", "d5.1x1", "d6.1x1"};

// One upsampling block: ELU -> transposed conv -> ELU -> 3x3 conv, plus
// the shortcut (1x1 conv on the raw input if channels change), repeated
// to the new size (nearest-neighbour upsampling).
static Tensor block(const Tensor &x, const CodecWeights::Block &B, int i) {
    Tensor y = conv2d(elu_of(conv_transpose2d(elu_of(x), B.up, B.st, B.sf, kUpLabel[i])), B.conv, 1, 1,
                      kConvLabel[i]);
    Tensor sc;
    const Tensor *s = &x;
    if (B.shortcut.k) {
        sc = conv2d(x, B.shortcut, 0, 0, kShortLabel[i]);
        s = &sc;
    }
    MRT2_PROF("add", (double)y.v.size(), 12.0 * y.v.size());
    for (int t = 0; t < y.t; t++)
        for (int f = 0; f < y.f; f++) {
            const float *a = s->at(t / B.st, f / B.sf);
            float *out = y.at(t, f);
            for (int o = 0; o < y.c; o++) out[o] += a[o];
        }
    return y;
}

Tensor decode(const CodecWeights &cw, const float *z, int frames, CodecDebug *dbg) {
    MRT2_STAGE("codec");
    Tensor x(frames, 1, 256);
    std::copy(z, z + (size_t)frames * 256, x.v.begin());

    // Input layer: first(x) + b2(elu(b1(x))), all 1x1 convs.
    Tensor a = conv2d(x, cw.in_first, 0, 0, "in.1x1");
    const Tensor b = conv2d(elu_of(conv2d(x, cw.in_b1, 0, 0, "in.1x1")), cw.in_b2, 0, 0, "in.1x1");
    for (size_t i = 0; i < a.v.size(); i++) a.v[i] += b.v[i];
    if (dbg) dbg->input_layer = a;

    // [T][1][2560] -> [T][5][512]: the same memory, read as 5 bins x 512.
    a.f = 5;
    a.c = 512;

    // Residual unit: x + conv(elu(conv(elu(x)))).
    const Tensor r = conv2d(elu_of(conv2d(elu_of(a), cw.res_a, 1, 1, "res.3x3")), cw.res_b, 1, 1, "res.3x3");
    for (size_t i = 0; i < a.v.size(); i++) a.v[i] += r.v[i];
    if (dbg) dbg->input_res = a;

    const Tensor d0 = block(a, cw.block[0], 0); // [2T][5][1024]
    if (dbg) dbg->decoder0 = d0;

    // Stereo: channels [0, 512) are the left group, [512, 1024) the right;
    // both go through the same blocks and output layer.
    Tensor spec(4 * frames, 480, 4);
    for (int g = 0; g < 2; g++) {
        Tensor h(d0.t, d0.f, 512);
        for (int t = 0; t < d0.t; t++)
            for (int f = 0; f < d0.f; f++) std::copy(d0.at(t, f) + 512 * g, d0.at(t, f) + 512 * (g + 1), h.at(t, f));
        for (int i = 1; i < 7; i++) {
            h = block(h, cw.block[i], i);
            if (dbg) dbg->block[i - 1][g] = h;
        }
        const Tensor o = conv2d(elu_of(h), cw.out, 3, 3, "out.7x7"); // [4T][480][2]: real, imag
        for (int t = 0; t < o.t; t++)
            for (int f = 0; f < o.f; f++) {
                spec.at(t, f)[2 * g] = o.at(t, f)[0];
                spec.at(t, f)[2 * g + 1] = o.at(t, f)[1];
            }
    }
    if (dbg) dbg->spectrum = spec;
    return spec;
}

// ---- inverse STFT ---------------------------------------------------------------

std::vector<float> inverse_stft(const Tensor &spec) {
    MRT2_STAGE("codec");
    const int N = 960, H = 480, bins = 480; // frame length, hop, stored bins
    // direct inverse DFT: per step and channel, 960 outputs x 480 bins x 4 flops
    MRT2_PROF("istft", 4.0 * spec.t * 2 * N * bins, 4.0 * spec.v.size());
    const double pi = 3.14159265358979323846;
    // Periodic Hann window, and Google's correction that makes the
    // half-overlapped windows add up to exactly 1: w / (w[n]^2 + w[n+H]^2).
    float win[N], cs[N], sn[N];
    double hann[N];
    for (int n = 0; n < N; n++) hann[n] = 0.5 - 0.5 * std::cos(2 * pi * n / N);
    for (int n = 0; n < N; n++) {
        const double den = hann[n % H] * hann[n % H] + hann[n % H + H] * hann[n % H + H];
        win[n] = den == 0.0 ? 0.0f : (float)(hann[n] / den);
        cs[n] = (float)std::cos(2 * pi * n / N);
        sn[n] = (float)std::sin(2 * pi * n / N);
    }

    const int steps = spec.t;
    std::vector<float> out((size_t)steps * H * 2, 0.0f);
    std::vector<float> prev(2 * H, 0.0f), cur(2 * N); // second halves of the last frame
    for (int t = 0; t < steps; t++) {
        // Frame 0 (its 4 steps) is the decoder's lookahead: invalid, so it
        // counts as silence, exactly as in Google's streaming output.
        const bool valid = t >= 4;
        for (int ch = 0; ch < 2; ch++) {
            // Inverse real FFT of bins 0..479 (bin 480, Nyquist, is 0):
            // x[n] = (Re X0 + 2 sum_k (Re Xk cos - Im Xk sin)(2 pi k n / N)) / N.
            // The imaginary part of bin 0 is ignored, like numpy's irfft.
            for (int n = 0; n < N; n++) {
                double s = 0.0;
                if (valid) {
                    s = spec.at(t, 0)[2 * ch];
                    for (int k = 1; k < bins; k++) {
                        const float *X = spec.at(t, k);
                        const int m = (int)(((long)k * n) % N);
                        s += 2.0 * ((double)X[2 * ch] * cs[m] - (double)X[2 * ch + 1] * sn[m]);
                    }
                }
                cur[ch * N + n] = (float)(s / N) * win[n];
            }
            // Overlap-add: this step's 480 samples = first half of this
            // frame + second half of the previous one.
            for (int j = 0; j < H; j++) {
                out[((size_t)t * H + j) * 2 + ch] = cur[ch * N + j] + prev[ch * H + j];
                prev[ch * H + j] = cur[ch * N + H + j];
            }
        }
    }
    return out;
}

int16_t to_int16(float x) {
    float c = 0.5f * x;
    c = c < -1.0f ? -1.0f : c > 1.0f ? 1.0f : c;
    const double v = 32767.5 * (double)c - 0.5;
    // jnp.round rounds half to even (-0.5 -> 0, 2.5 -> 2); std::round would
    // turn every silent sample into -1.
    double r = std::floor(v);
    const double frac = v - r;
    if (frac > 0.5 || (frac == 0.5 && std::fmod(r, 2.0) != 0.0)) r += 1.0;
    return (int16_t)r;
}

bool write_wav(const char *path, const std::vector<int16_t> &stereo, int sample_rate) {
    FILE *f = std::fopen(path, "wb");
    if (!f) return false;
    // A 44-byte RIFF header, every field little-endian, then the samples.
    auto u32 = [&](uint32_t v) { unsigned char b[4] = {(unsigned char)v, (unsigned char)(v >> 8), (unsigned char)(v >> 16), (unsigned char)(v >> 24)}; std::fwrite(b, 1, 4, f); };
    auto u16 = [&](uint16_t v) { unsigned char b[2] = {(unsigned char)v, (unsigned char)(v >> 8)}; std::fwrite(b, 1, 2, f); };
    const uint32_t data = (uint32_t)(stereo.size() * 2);
    std::fwrite("RIFF", 1, 4, f); u32(36 + data); std::fwrite("WAVE", 1, 4, f);
    std::fwrite("fmt ", 1, 4, f); u32(16); u16(1); u16(2);       // PCM, 2 channels
    u32(sample_rate); u32(sample_rate * 4); u16(4); u16(16);       // byte rate, block align, 16-bit
    std::fwrite("data", 1, 4, f); u32(data);
    for (int16_t s : stereo) u16((uint16_t)s);
    return std::fclose(f) == 0;
}

} // namespace mrt2
