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
// Two ways to run it, with the same results:
//   offline   decode() + inverse_stft(): a whole clip at once (the
//             reference for tests);
//   streaming decode_frame(): one frame in, its 1920 samples out, with no
//             limit on length. Every conv is causal in time, so each one
//             only needs to remember its last few input rows (CodecStream).
#pragma once
#include "conv.h"
#include "weights.h"
#include <cstdint>
#include <vector>

namespace mrt2 {

constexpr int kSamplesPerFrame = 1920; // 48000 Hz / 25 frames per second
constexpr int kSampleRate = 48000;

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

// Store the early layers' weights (input layer, input residual unit,
// decoder_0, decoder_1: ~120 MB read every frame) as fp16, halving the
// bytes they read. Measured on Google's 50 reference frames: 76 dB SNR vs
// float32, at most a few int16 steps (inaudible). Without it: exact float32,
// the mode the tests compare against Google's numbers.
void make_fp16(CodecWeights &cw);

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

// ---- streaming --------------------------------------------------------------------

// Everything the decoder remembers between frames: for each causal conv
// its last input rows (as many as its kernel reaches back: 2 for a 3x3,
// 6 for the 7x7, 1 for a stride-2 transposed conv), and the second half of
// the last inverse-STFT window. Starts as zeros, exactly the causal
// padding the offline decoder sees. A few hundred KB; the first frame
// sizes it, later frames reuse it.
struct CodecStream {
    Tensor res_a, res_b;          // input residual unit
    Tensor up[7][2], conv[7][2];  // block i, stereo group g (block 0: g = 0 only)
    Tensor out[2];                // output 7x7, per group
    float overlap[2][480] = {};   // per channel
    long frame = 0;               // frames decoded so far
};

// One frame: z[256] -> 1920 stereo samples, interleaved (L, R, L, R...).
// Frame t's samples equal the offline decoder's samples for frame t.
void decode_frame(const CodecWeights &cw, CodecStream &st, const float z[256], float out[2 * kSamplesPerFrame]);

// Float sample -> 16-bit, exactly as Google: round(32767.5 * clip(0.5 x) - 0.5).
int16_t to_int16(float x);

// Write interleaved stereo 16-bit samples as a .wav file.
bool write_wav(const char *path, const std::vector<int16_t> &stereo, int sample_rate);

} // namespace mrt2
