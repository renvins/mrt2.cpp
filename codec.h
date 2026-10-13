// codec.h - the SpectroStream decoder: tokens -> stereo 48 kHz audio.
//
// Per 40 ms frame, 12 tokens (one per RVQ level) become 1920 stereo
// samples:
//
//   1. dequantize: z = sum over the 12 levels of codebook[level][code]  [256]
//   2. decoder: 2D convolutions over (time, frequency) that grow the time
//      axis 4x (25 Hz -> 100 Hz) and the frequency axis 5 -> 480 bins,
//      ending in a spectrum [time][480 bins][4 = L re, L im, R re, R im]:
//        input layer (1x1 convs) -> reshape to 5 bins x 512 channels
//        -> residual unit -> decoder_0 (time x2)
//        -> split channels into left / right halves, same weights for both:
//           decoder_1 (time x2, freq x2) -> decoder_2..6 (freq x2, x2, x3,
//           x2, x2) -> ELU -> 7x7 conv to 2 channels (real, imag)
//      Each block: ELU -> transposed conv (upsample) -> ELU -> 3x3 conv,
//      plus a shortcut (1x1 conv if channels change, then upsample).
//      All convs are causal in time: an output never sees the future.
//   3. inverse STFT: each 100 Hz spectrum step -> 960 samples (inverse
//      real FFT, Hann-based window), overlapped by half: 480 new samples
//      per step, 1920 per frame.
//
// The decoder looks one frame ahead, so audio comes out one frame late:
// like Google's streaming output, frame 0 is silent and frame t is the
// audio that frame t's tokens complete.
//
// This is the offline version (a whole clip at once, simple loops); the
// real-time streaming version comes in the speed phase.
#pragma once
#include "weights.h"
#include <cstdint>
#include <vector>

namespace mrt2 {

constexpr int kSamplesPerFrame = 1920; // 48000 Hz / 25 frames per second
constexpr int kSampleRate = 48000;

// Activations: [time][freq][channel], channels contiguous (like JAX).
struct Tensor {
    int t = 0, f = 0, c = 0;
    std::vector<float> v;
    Tensor() = default;
    Tensor(int t_, int f_, int c_) : t(t_), f(f_), c(c_), v((size_t)t_ * f_ * c_, 0.0f) {}
    float *at(int ti, int fi) { return v.data() + ((size_t)ti * f + fi) * c; }
    const float *at(int ti, int fi) const { return v.data() + ((size_t)ti * f + fi) * c; }
};

// One convolution's weights: kernel [kt][kf][cin][cout] (the checkpoint's
// layout) and bias [cout].
struct Conv {
    const float *k = nullptr, *b = nullptr;
    int kt = 0, kf = 0, cin = 0, cout = 0;
};

struct CodecWeights {
    const float *codebooks;               // [64][1024][256]; levels 0..11 used
    Conv in_first, in_b1, in_b2;          // input layer: 1x1, 256 -> 2560
    Conv res_a, res_b;                    // input residual unit: 3x3, 512
    struct Block {
        Conv up;                          // transposed conv
        Conv conv;                        // 3x3 conv
        Conv shortcut;                    // 1x1, only if channels change (else k == nullptr)
        int st, sf;                       // upsampling factors (time, freq)
    } block[7];                           // decoder_0 .. decoder_6
    Conv out;                             // output 7x7, 64 -> 2
};

CodecWeights load_codec(const Weights &w);

// 12 global token ids -> z[256].
void dequantize(const CodecWeights &cw, const int32_t tokens[12], float z[256]);

// Optional intermediate outputs, for tests (empty pointer = skipped).
struct CodecDebug {
    Tensor input_layer;       // [T][1][2560]
    Tensor input_res;         // [T][5][512]
    Tensor decoder0;          // [2T][5][1024]
    Tensor block[6][2];       // decoder_1..6, per stereo group (left, right)
    Tensor spectrum;          // [4T][480][4]
};

// z [frames][256] -> spectrum [4 * frames][480][4].
Tensor decode(const CodecWeights &cw, const float *z, int frames, CodecDebug *dbg = nullptr);

// spectrum [4 * frames][480][4] -> interleaved stereo float samples
// [frames * 1920][2]. Frame 0 is silent (see the lookahead note above).
std::vector<float> inverse_stft(const Tensor &spectrum);

// Float sample -> 16-bit, exactly as Google: round(32767.5 * clip(0.5 x) - 0.5).
int16_t to_int16(float x);

// Write interleaved stereo 16-bit samples as a .wav file.
bool write_wav(const char *path, const std::vector<int16_t> &stereo, int sample_rate);

} // namespace mrt2
