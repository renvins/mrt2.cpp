// depth.h - the depth decoder: one frame's 12 RVQ tokens, one at a time.
//
// RVQ ("residual vector quantization") describes 40 ms of audio with 12
// tokens: level 0 is a coarse description, level 1 encodes what level 0
// got wrong, level 2 what is still wrong after both, and so on. So each
// level must see the ones already chosen, and the depth decoder runs 12
// small steps per frame:
//
//   step 0:  input = T_t (temporal decoder output)  -> pick token 0
//   step k:  input = 32 * embedding[token k-1]      -> pick token k
//
// Each step: adapter [1024 -> 768] -> 2 layers (self-attention over this
// frame's earlier steps, 6 heads, no sink; FFN 3072) -> LayerNorm ->
// this level's 1024 logits -> soft-cap 30 -> top-k / temperature sampling.
// The attention memory is reset at the start of every frame.
#pragma once
#include "blocks.h"
#include "ops.h"
#include "temporal.h"
#include "weights.h"
#include <vector>

namespace mrt2 {

constexpr int kDepthModel = 768;
constexpr int kDepthFFN = 3072;
constexpr int kDepthLayers = 2;
constexpr int kCodebook = 1024;  // codes per RVQ level
constexpr int kReserved = 6;     // token ids 0..5 are reserved

struct DepthWeights {
    const float *embedding;      // [12294][1024], the same table as the temporal input
    const float *adapter;        // [1024][768]
    struct Layer {
        AttentionWeights self;   // no sink
        FFNWeights ffn;
    } layer[kDepthLayers];
    const float *ln_scale, *ln_bias; // final LayerNorm [768]
    const float *logits_w;       // [768][12294]
    const float *logits_b;       // [12294]
    // Each level's 1024 logit columns copied at load time into one
    // contiguous block, [12][768][1024]: reading a 1024-column slice
    // straight out of the 12294-wide matrix jumps 45 KB between rows,
    // which the memory prefetcher handles badly (measured: 34% of bandwidth).
    std::vector<float> logits_packed;
    // int8 copies, empty unless make_int8() was called: the adapter, and
    // each level's 1024 logit rows (bias kept in float32).
    QMatrix q8_adapter, q8_logits[kLevels];
};

DepthWeights load_depth(const Weights &w);

// Switch the decoder to int8 matrix products, as make_int8(TemporalWeights&).
void make_int8(DepthWeights &dw);

struct Sampling {
    float temperature = 1.3f;    // Google's defaults; 0 = greedy
    int top_k = 40;
};

// Optional intermediate outputs, for tests (any pointer may be nullptr).
struct DepthDebug {
    float *adapter = nullptr;    // [12][768] adapter output at each step
    float *final_ln = nullptr;   // [12][768] after the final LayerNorm
    float *logits = nullptr;     // [12][1024] each level's logits, before soft-cap
    // Teacher forcing: use these 12 global ids instead of the sampled ones
    // (they still go to `tokens`), so two runs see the same inputs.
    const int32_t *force = nullptr;
};

// Choose one level's code from its 1024 logits the way Google does:
// soft-cap (30 * tanh(x / 30)), then top-k / temperature sampling.
// `logits` is modified in place (it ends up soft-capped). Returns 0..1023.
int sample_code(float logits[kCodebook], const Sampling &s, Rng &rng);

// The depth decoder's attention memory for one frame: one key and one
// value slot per step, for each layer (147 KB). Owned by the caller and
// allocated once; nothing needs resetting between frames, because step k
// only reads slots [0, k], all written earlier in the same frame.
struct DepthState {
    std::vector<float> keys, values; // [2][12][768]
    DepthState() : keys((size_t)kDepthLayers * kLevels * kDepthModel), values(keys.size()) {}
};

// One frame: T_t[1024] -> the 12 global token ids (6 + level*1024 + code).
void depth_frame(const DepthWeights &dw, DepthState &st, const float t[kModel], const Sampling &s,
                 Rng &rng, int32_t tokens[kLevels], DepthDebug *dbg = nullptr);

} // namespace mrt2
