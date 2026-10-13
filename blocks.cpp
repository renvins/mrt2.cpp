#include "blocks.h"
#include "ops.h"
#include <cassert>
#include <cmath>

namespace mrt2 {

AttentionWeights load_attention(const Weights &w, const std::string &p, int model, int kv_in,
                                bool sink) {
    AttentionWeights a;
    a.model = model;
    a.heads = model / kHeadDim;
    const int64_t m = model, h = a.heads;
    a.pre_norm = w.f32(p + "pre_norm/scale", {m});
    a.wq = w.f32(p + "attention/query_projection/kernel", {m, h, kHeadDim});
    a.wk = w.f32(p + "attention/key_projection/kernel", {kv_in, h, kHeadDim});
    a.wv = w.f32(p + "attention/value_projection/kernel", {kv_in, h, kHeadDim});
    if (sink) {
        a.sink_k = w.f32(p + "attention/sink_key_embeddings", {1, h, kHeadDim});
        a.sink_v = w.f32(p + "attention/sink_value_embeddings", {1, h, kHeadDim});
    }
    a.wo = w.f32(p + "output_projection/kernel", {m, h, kHeadDim});
    a.post_norm = w.f32(p + "post_norm/scale", {m});
    // Query scale, exactly as Google's _scale_query: the constant is
    // 1 / softplus(0), so an untrained per_dim_scale (0) gives 1/sqrt(128).
    const float *pds = w.f32(p + "attention/per_dim_scale", {kHeadDim});
    const float base = (float)(1.442695041 / std::sqrt((double)kHeadDim));
    for (int d = 0; d < kHeadDim; d++) a.q_scale[d] = base * softplus(pds[d]);
    return a;
}

FFNWeights load_ffn(const Weights &w, const std::string &p, int model, int hidden) {
    FFNWeights f;
    f.model = model;
    f.hidden = hidden;
    f.pre_norm = w.f32(p + "pre_norm/scale", {model});
    f.w1 = w.f32(p + "ffn_layer1/kernel", {model, hidden});
    f.b1 = w.f32(p + "ffn_layer1/bias", {hidden});
    f.w2 = w.f32(p + "ffn_layer2/kernel", {hidden, model});
    f.b2 = w.f32(p + "ffn_layer2/bias", {model});
    f.post_norm = w.f32(p + "post_norm/scale", {model});
    return f;
}

static float dot(const float *a, const float *b, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

// One query against the (optional) sink and the n valid slots, all heads:
// ctx[head] = softmax(scores) . values.
static void attend(const AttentionWeights &a, const float *q, const float *keys, const float *values,
                   int n, float *ctx) {
    assert(n >= 1 && n <= kMaxKeys);
    const int s = a.sink_k ? 1 : 0; // index of the first real slot in p[]
    for (int h = 0; h < a.heads; h++) {
        const float *qh = q + h * kHeadDim;
        float qs[kHeadDim]; // the scaled query, used for the real keys
        for (int d = 0; d < kHeadDim; d++) qs[d] = qh[d] * a.q_scale[d];

        // Scores: the sink first (with the UNscaled query), then the slots.
        float p[1 + kMaxKeys];
        if (s) p[0] = dot(qh, a.sink_k + h * kHeadDim, kHeadDim);
        for (int j = 0; j < n; j++) p[s + j] = dot(qs, keys + (size_t)j * a.model + h * kHeadDim, kHeadDim);
        softmax(p, s + n);

        float *out = ctx + h * kHeadDim;
        for (int d = 0; d < kHeadDim; d++) out[d] = s ? p[0] * a.sink_v[h * kHeadDim + d] : 0.0f;
        for (int j = 0; j < n; j++) {
            const float *v = values + (size_t)j * a.model + h * kHeadDim;
            for (int d = 0; d < kHeadDim; d++) out[d] += p[s + j] * v[d];
        }
    }
}

void attention_block(const AttentionWeights &a, float *x, const float *src, int src_dim,
                     float *keys, float *values, int slot, int n) {
    const int m = a.model;
    float u[kMaxModel], q[kMaxModel], ctx[kMaxModel], y[kMaxModel];
    rmsnorm(x, a.pre_norm, m, 1e-6f, u);
    vecmat(u, m, a.wq, m, m, nullptr, q);
    // Self-attention projects the normalized x; cross-attention projects src.
    const float *kv_in = src ? src : u;
    vecmat(kv_in, src_dim, a.wk, m, m, nullptr, keys + (size_t)slot * m);
    vecmat(kv_in, src_dim, a.wv, m, m, nullptr, values + (size_t)slot * m);
    attend(a, q, keys, values, n, ctx);
    matvec(a.wo, m, m, ctx, y); // Wo is [out][in]
    rmsnorm(y, a.post_norm, m, 1e-6f, y);
    for (int i = 0; i < m; i++) x[i] += y[i];
}

void ffn_block(const FFNWeights &f, float *x) {
    float u[kMaxModel], y[kMaxModel], h[kMaxHidden]; // 24 KB of stack: fine
    rmsnorm(x, f.pre_norm, f.model, 1e-6f, u);
    vecmat(u, f.model, f.w1, f.hidden, f.hidden, f.b1, h);
    gelu_tanh(h, f.hidden);
    vecmat(h, f.hidden, f.w2, f.model, f.model, f.b2, y);
    rmsnorm(y, f.post_norm, f.model, 1e-6f, y);
    for (int i = 0; i < f.model; i++) x[i] += y[i];
}

} // namespace mrt2
