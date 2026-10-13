// temporal.h - the temporal decoder (runs once per frame).
//
// It is the model's memory of the music so far. Each frame it takes
//   x   = the previous frame's 12 tokens, embedded and averaged [1024],
//   e_t = this frame's encoder output (what the user asked for) [256],
// and runs 12 transformer layers, each:
//
//   x = x + post_norm( Wo * SelfAttention ( pre_norm(x) ) )    look back at own past
//   x = x + post_norm( Wo * CrossAttention( pre_norm(x), e ) ) look at the request
//   x = x + post_norm( W2 * gelu( W1 * pre_norm(x) + b1 ) + b2 )
//
// with RMSNorm everywhere. The output T_t[1024] is the last layer's x (no
// final norm) and feeds the depth decoder.
//
// The attention and FFN blocks themselves live in blocks.h (shared with the
// depth decoder). Their keys and values live in ring buffers of 42 slots
// (41 past frames + this one): each new frame overwrites the oldest slot,
// which works because the model has no positional encoding.
#pragma once
#include "blocks.h"
#include "weights.h"
#include <cstdint>
#include <vector>

namespace mrt2 {

constexpr int kModel = 1024;     // d_model of the temporal decoder
constexpr int kFFN = 4096;
constexpr int kTemporalLayers = 12;
constexpr int kWindow = 41;      // past frames each attention can see
constexpr int kSlots = kWindow + 1; // + the current frame
constexpr int kLevels = 12;      // RVQ tokens per frame
constexpr int kVocab = 12294;    // 6 reserved + 12 levels x 1024 codes

struct TemporalWeights {
    const float *embedding;     // [12294][1024], shared with the depth decoder
    struct Layer {
        AttentionWeights self, cross; // both with a sink
        FFNWeights ffn;
    } layer[kTemporalLayers];
};

// Look the weights up by name; throws std::runtime_error if one is
// missing or has the wrong shape.
TemporalWeights load_temporal(const Weights &w);

// Everything that carries over from one frame to the next: the keys and
// values of the last 42 frames for every layer. About 8 MB; allocated
// once, never during generation.
struct TemporalState {
    std::vector<float> self_k, self_v;   // [12][42][1024]
    std::vector<float> cross_k, cross_v; // [12][42][1024]
    int64_t frames = 0;                  // frames processed so far
    TemporalState();
    void reset() { frames = 0; } // slots past `frames` are never read
};

// The temporal input of a frame: mean over the 12 levels of
// 32 * embedding[token]. `prev` = the previous frame's 12 global token ids
// (all 0 = SOS for the very first frame).
void temporal_input(const TemporalWeights &tw, const int32_t prev[kLevels], float x[kModel]);

// Optional intermediate outputs, for tests (any pointer may be nullptr).
struct TemporalDebug {
    float *l0_self = nullptr;   // [1024] layer 0 after the self-attention block
    float *l0_cross = nullptr;  // [1024] layer 0 after the cross-attention block
    float *layers = nullptr;    // [12][1024] each layer's output
};

// One frame: x[1024] (from temporal_input) and e[256] (encoder output)
// -> out[1024]. Updates `st`.
void temporal_step(const TemporalWeights &tw, TemporalState &st, const float x[kModel],
                   const float e[256], float out[kModel], TemporalDebug *dbg = nullptr);

} // namespace mrt2
