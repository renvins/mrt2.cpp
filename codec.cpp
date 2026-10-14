#include "codec.h"
#include "ops.h"
#include "profile.h"
#include "threads.h"
#include <algorithm>
#include <cmath>
#include <complex>
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
static void add_shortcut(Tensor &y, const Tensor &x, const CodecWeights::Block &B, int i) {
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
}

static Tensor block(const Tensor &x, const CodecWeights::Block &B, int i) {
    Tensor y = conv2d(elu_of(conv_transpose2d(elu_of(x), B.up, B.st, B.sf, kUpLabel[i])), B.conv, 1, 1,
                      kConvLabel[i]);
    add_shortcut(y, x, B, i);
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

namespace {
constexpr int kN = 960, kHop = 480, kBins = 480; // frame length, hop, stored bins
using cd = std::complex<double>;

// Computed once: the synthesis window and the FFT's "twiddle factors".
struct IstftTables {
    float win[kN];
    cd tw[kN]; // e^(+2 pi i j / N): the rotations the inverse FFT combines with
    IstftTables() {
        const double pi = 3.14159265358979323846;
        // Periodic Hann window, and Google's correction that makes the
        // half-overlapped windows add up to exactly 1: w / (w[n]^2 + w[n+H]^2).
        double hann[kN];
        for (int n = 0; n < kN; n++) hann[n] = 0.5 - 0.5 * std::cos(2 * pi * n / kN);
        for (int n = 0; n < kN; n++) {
            const double den = hann[n % kHop] * hann[n % kHop] + hann[n % kHop + kHop] * hann[n % kHop + kHop];
            win[n] = den == 0.0 ? 0.0f : (float)(hann[n] / den);
            tw[n] = std::polar(1.0, 2 * pi * n / kN);
        }
    }
};

const IstftTables &istft_tables() {
    static const IstftTables t; // built on first use (thread-safe in C++)
    return t;
}

// Inverse FFT (unnormalized): out[k] = sum_j in[j * stride] * e^(+2 pi i j k / n),
// for n = 960 / (product of the factors already used), mixed radix.
//
// Divide and conquer: with n = p * m, split the input into p interleaved
// sequences (every p-th sample), transform each (size m, recursively),
// then combine: out[k + r m] = sum_q sub_q[k] * e^(2 pi i q (k + r m) / n).
// 960 = 4 * 4 * 4 * 3 * 5: ~20k complex multiplies instead of the direct
// sum's 460k.
constexpr int kFactors[] = {4, 4, 4, 3, 5, 1};

void ifft(const cd *in, int stride, cd *out, int n, const int *factor, const cd *tw) {
    if (n == 1) {
        out[0] = in[0];
        return;
    }
    const int p = factor[0], m = n / p;
    for (int q = 0; q < p; q++) ifft(in + q * stride, stride * p, out + q * m, m, factor + 1, tw);
    // e^(2 pi i x / n) = tw[x * (N / n) mod N]
    const int step = kN / n;
    for (int k = 0; k < m; k++) {
        cd t[5]; // the p sub-results for this k (out[q m + k], overwritten below)
        for (int q = 0; q < p; q++) t[q] = out[q * m + k];
        for (int r = 0; r < p; r++) {
            cd sum = t[0];
            for (int q = 1; q < p; q++) sum += t[q] * tw[(q * (k + r * m) * step) % kN];
            out[k + r * m] = sum;
        }
    }
}

// One spectrum step of one channel -> its windowed 960 samples. An invalid
// step (the first frame: the decoder's lookahead) counts as silence,
// exactly as in Google's streaming output.
void istft_step(const Tensor &spec, int t, int ch, bool valid, float cur[kN]) {
    if (!valid) {
        std::fill(cur, cur + kN, 0.0f);
        return;
    }
    const IstftTables &T = istft_tables();
    // The full spectrum of a real signal: bin N - k is the conjugate of bin
    // k. Bin 0 keeps only its real part and bin 480 (Nyquist) is 0, like
    // numpy's irfft.
    cd z[kN], x[kN];
    z[0] = cd(spec.at(t, 0)[2 * ch], 0.0);
    z[kBins] = cd(0.0, 0.0);
    for (int k = 1; k < kBins; k++) {
        const float *X = spec.at(t, k);
        z[k] = cd(X[2 * ch], X[2 * ch + 1]);
        z[kN - k] = std::conj(z[k]);
    }
    ifft(z, 1, x, kN, kFactors, T.tw);
    for (int n = 0; n < kN; n++) cur[n] = (float)(x[n].real() / kN) * T.win[n];
}
} // namespace

std::vector<float> inverse_stft(const Tensor &spec) {
    MRT2_STAGE("codec");
    // inverse FFT: per step and channel ~20k complex multiply-adds (8 flops)
    MRT2_PROF("istft", 8.0 * 20000 * spec.t * 2, 4.0 * spec.v.size());
    const int steps = spec.t;
    // 1. Every step's windowed inverse DFT, for both channels, independently:
    //    spread over the threads. frames[t][ch][n]. Frame 0's 4 steps are invalid.
    std::vector<float> frames((size_t)steps * 2 * kN);
    pool().run(steps * 2, [&](int job) { istft_step(spec, job / 2, job % 2, job / 2 >= 4, &frames[(size_t)job * kN]); });
    // 2. Overlap-add, in order: step t's 480 samples = first half of frame
    //    t + second half of frame t - 1.
    std::vector<float> out((size_t)steps * kHop * 2, 0.0f);
    for (int t = 0; t < steps; t++)
        for (int ch = 0; ch < 2; ch++) {
            const float *cur = &frames[((size_t)t * 2 + ch) * kN];
            const float *prev = t > 0 ? &frames[((size_t)(t - 1) * 2 + ch) * kN] : nullptr;
            for (int j = 0; j < kHop; j++)
                out[((size_t)t * kHop + j) * 2 + ch] = cur[j] + (prev ? prev[kHop + j] : 0.0f);
        }
    return out;
}

// ---- streaming ----------------------------------------------------------------------

// A causal conv on this frame's new input rows `x`: put the remembered
// rows in front, compute only the outputs of the new rows, and remember
// the last rows for the next frame. The first call creates the memory as
// zeros: the same causal padding the offline conv sees.
static Tensor stream_conv(Tensor &past, const Tensor &x, const Conv &c, bool transposed, int st, int sf,
                          int pl, int pr, const char *label) {
    // How far back the kernel reaches, in input rows: kt - 1 for a regular
    // conv; floor((kt - 1) / st) for a transposed one (its taps step over
    // the zeros inserted between input rows).
    const int keep = transposed ? (c.kt - 1) / st : c.kt - 1;
    if (past.t != keep || past.f != x.f || past.c != x.c) past = Tensor(keep, x.f, x.c);
    Tensor all(keep + x.t, x.f, x.c);
    std::copy(past.v.begin(), past.v.end(), all.v.begin());
    std::copy(x.v.begin(), x.v.end(), all.v.begin() + past.v.size());
    std::copy(all.v.end() - past.v.size(), all.v.end(), past.v.begin());
    return transposed ? conv_transpose2d(all, c, st, sf, label, keep * st) : conv2d(all, c, pl, pr, label, keep);
}

static Tensor stream_block(const Tensor &x, const CodecWeights::Block &B, int i, Tensor &up, Tensor &conv) {
    Tensor y = stream_conv(conv, elu_of(stream_conv(up, elu_of(x), B.up, true, B.st, B.sf, 0, 0, kUpLabel[i])),
                           B.conv, false, 1, 1, 1, 1, kConvLabel[i]);
    add_shortcut(y, x, B, i); // 1x1 and per time step: nothing to remember
    return y;
}

void decode_frame(const CodecWeights &cw, CodecStream &st, const float z[256], float out[2 * kSamplesPerFrame]) {
    MRT2_STAGE("codec");
    Tensor x(1, 1, 256);
    std::copy(z, z + 256, x.v.begin());

    // Input layer: 1x1 convs, nothing to remember. Same as decode().
    Tensor a = conv2d(x, cw.in_first, 0, 0, "in.1x1");
    const Tensor b = conv2d(elu_of(conv2d(x, cw.in_b1, 0, 0, "in.1x1")), cw.in_b2, 0, 0, "in.1x1");
    for (size_t i = 0; i < a.v.size(); i++) a.v[i] += b.v[i];
    a.f = 5;
    a.c = 512;

    const Tensor r = stream_conv(st.res_b, elu_of(stream_conv(st.res_a, elu_of(a), cw.res_a, false, 1, 1, 1, 1, "res.3x3")),
                                 cw.res_b, false, 1, 1, 1, 1, "res.3x3");
    for (size_t i = 0; i < a.v.size(); i++) a.v[i] += r.v[i];

    const Tensor d0 = stream_block(a, cw.block[0], 0, st.up[0][0], st.conv[0][0]); // [2][5][1024]

    Tensor spec(4, 480, 4); // this frame's 4 spectrum steps
    for (int g = 0; g < 2; g++) {
        Tensor h(d0.t, d0.f, 512);
        for (int t = 0; t < d0.t; t++)
            for (int f = 0; f < d0.f; f++) std::copy(d0.at(t, f) + 512 * g, d0.at(t, f) + 512 * (g + 1), h.at(t, f));
        for (int i = 1; i < 7; i++) h = stream_block(h, cw.block[i], i, st.up[i][g], st.conv[i][g]);
        const Tensor o = stream_conv(st.out[g], elu_of(h), cw.out, false, 1, 1, 3, 3, "out.7x7"); // [4][480][2]
        for (int t = 0; t < o.t; t++)
            for (int f = 0; f < o.f; f++) {
                spec.at(t, f)[2 * g] = o.at(t, f)[0];
                spec.at(t, f)[2 * g + 1] = o.at(t, f)[1];
            }
    }

    // Inverse STFT of the 4 steps, then overlap-add with the remembered
    // half window. Frame 0 is the lookahead: silent.
    MRT2_PROF("istft", 8.0 * 20000 * 4 * 2, 4.0 * spec.v.size());
    float frames[4][2][kN];
    const bool valid = st.frame > 0;
    pool().run(8, [&](int job) { istft_step(spec, job / 2, job % 2, valid, frames[job / 2][job % 2]); });
    for (int t = 0; t < 4; t++)
        for (int ch = 0; ch < 2; ch++) {
            for (int j = 0; j < kHop; j++) out[((size_t)t * kHop + j) * 2 + ch] = frames[t][ch][j] + st.overlap[ch][j];
            std::copy(frames[t][ch] + kHop, frames[t][ch] + kN, st.overlap[ch]);
        }
    st.frame++;
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
