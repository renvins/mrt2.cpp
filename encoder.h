// encoder.h - the conditioning encoder (runs once per frame).
//
// Every frame the model is told what to play through 144 integer tokens:
//
//   [0, 12)     MusicCoCa style tokens        (codebook 1024 each)
//   [12, 140)   piano roll, one per MIDI pitch (4 states each)
//   140         drums                         (2 states)
//   141, 142    CFG strength: musiccoca, notes (41 bins)
//   143         CFG strength: drums           (9 bins)
//
// The encoder turns them into one vector e_t[256] that the temporal
// decoder attends to. There is no attention here (Google's encoder body
// is Identity), only two embedding branches, averaged, then a LayerNorm:
//
//   mulan   = (sum of 12 style-token embeddings [768]) x adapter [768 -> 256]
//   regular = mean of the 132 other channels' embeddings [256]
//   e_t     = LayerNorm((mulan + regular) / 2)
#pragma once
#include "weights.h"
#include <cstdint>

namespace mrt2 {

constexpr int kCondTokens = 144;  // conditioning tokens per frame
constexpr int kEncDim = 256;      // encoder output size (mrt2_small)
constexpr int kStyleTokens = 12;
constexpr int kPitches = 128;

// User-level conditioning for one frame. -1 means "no condition" for a
// token (the model was trained with such dropped-out inputs).
struct Conditioning {
    int style[kStyleTokens];   // MusicCoCa tokens 0..1023, or -1
    int pitch[kPitches];       // piano-roll state 0..3, or -1
    int drums = -1;            // 0..1, or -1
    // Guidance strengths (Google's defaults). double, not float, because
    // Google computes the bins with Python floats (doubles): as a float,
    // 0.3 is 0.30000001..., which lands just above a rounding boundary
    // and gives a different bin.
    double cfg_musiccoca = 3.0;
    double cfg_notes = 1.0;
    double cfg_drums = 1.0;

    Conditioning(); // everything -1, default CFG strengths
};

// A CFG strength -> its bin, exactly like Google's discretize_cfg():
// clamp to [-1, 7], divide by `step`, round half to even, clamp to max_bin.
int discretize_cfg(double value, double step, int max_bin);

// Conditioning -> the 144 tokens the model sees: each value + 7 (6
// reserved ids + 1 "dropout" id), so -1 becomes 6. Returns false (and
// fills `err`) if a value is outside its channel's range.
bool encoder_tokens(const Conditioning &c, int32_t out[kCondTokens], const char **err);

// Pointers to the encoder's weights inside the mapped model file.
struct EncoderWeights {
    const float *style_emb;    // [12 * 1031, 768] MusicCoCa token embeddings
    const float *adapter;      // [768, 256]
    const float *regular_emb;  // [1536, 256] all other channels, one table
    const float *ln_scale;     // [256]
    const float *ln_bias;      // [256]
};

// Look the weights up by name; throws std::runtime_error if one is
// missing or has the wrong shape.
EncoderWeights load_encoder(const Weights &w);

// Optional intermediate outputs, for tests (any pointer may be nullptr).
struct EncoderDebug {
    float *mulan = nullptr;    // [256]
    float *regular = nullptr;  // [256]
    float *combined = nullptr; // [256], before LayerNorm
};

// tokens[144] -> out[256]. Tokens must come from encoder_tokens() (or be
// valid in the same way): they index the embedding tables directly.
void encode(const EncoderWeights &e, const int32_t tokens[kCondTokens], float out[kEncDim],
            EncoderDebug *dbg = nullptr);

} // namespace mrt2
