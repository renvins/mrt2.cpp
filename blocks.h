// blocks.h - the two transformer building blocks, shared by the temporal
// decoder (d_model 1024, 8 heads, with attention sink) and the depth
// decoder (d_model 768, 6 heads, no sink). Same math, different sizes.
//
//   attention block:  x = x + post_norm( Wo * attention( pre_norm(x) ) )
//   ffn block:        x = x + post_norm( W2 * gelu( W1 * pre_norm(x) + b1 ) + b2 )
//
// Attention, per head (every head is 128 numbers wide):
//   score(sink)  = q . sink_key                 (q NOT scaled: Google's rule)
//   score(slot)  = (q * q_scale) . key(slot)    for the n valid slots
//   weights      = softmax(all scores)
//   result       = weights . values (+ weight(sink) * sink_value)
// with q_scale = 1.442695041 / sqrt(128) * softplus(per_dim_scale).
//
// Keys and values live in caller-owned slot buffers [slots][d_model]. The
// model has no positional encoding, so the order of the slots does not
// matter: only that slots [0, n) are the valid ones.
#pragma once
#include "weights.h"
#include <string>

namespace mrt2 {

constexpr int kHeadDim = 128;  // width of every attention head
constexpr int kMaxModel = 1024; // largest d_model (temporal decoder)
constexpr int kMaxHidden = 4096; // largest FFN hidden size
constexpr int kMaxKeys = 42;   // most slots an attention ever reads

struct AttentionWeights {
    int model = 0;              // d_model: 1024 or 768
    int heads = 0;              // model / 128
    const float *pre_norm;      // [model]
    const float *wq, *wk, *wv;  // [model or kv_in][model]
    const float *sink_k = nullptr, *sink_v = nullptr; // [heads * 128], or none
    const float *wo;            // [model][heads * 128]  ([out][in] layout)
    const float *post_norm;     // [model]
    float q_scale[kHeadDim];    // computed at load time from per_dim_scale
};

struct FFNWeights {
    int model = 0, hidden = 0;  // 1024/4096 or 768/3072
    const float *pre_norm;      // [model]
    const float *w1, *b1;       // [model][hidden], [hidden]
    const float *w2, *b2;       // [hidden][model], [model]
    const float *post_norm;     // [model]
};

// Load one block from the tensors under `prefix` (ending in '/').
// kv_in: size of what keys/values are projected from (= model for
// self-attention, 256 for cross-attention on the encoder output).
AttentionWeights load_attention(const Weights &w, const std::string &prefix, int model, int kv_in,
                                bool sink);
FFNWeights load_ffn(const Weights &w, const std::string &prefix, int model, int hidden);

// x += post_norm(Wo * attention(pre_norm(x))).
// The new key and value are projected from `src` (src_dim numbers), or
// from pre_norm(x) if src is nullptr (self-attention), and written into
// slot `slot` before attending, so the current position sees itself.
void attention_block(const AttentionWeights &a, float *x, const float *src, int src_dim,
                     float *keys, float *values, int slot, int n);

// x += post_norm(W2 * gelu(W1 * pre_norm(x) + b1) + b2).
void ffn_block(const FFNWeights &f, float *x);

} // namespace mrt2
