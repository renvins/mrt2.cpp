#include "blocks.h"
#include "ops.h"
#include "profile.h"
#include <algorithm>
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

void make_int8(AttentionWeights &a, int kv_in, bool self) {
    const int m = a.model;
    a.q8_in = a.q8_kv = a.q8_o = QMatrix();
    q8_append_kernel(a.q8_in, a.wq, m, m, 0, m);
    QMatrix &kv = self ? a.q8_in : a.q8_kv; // self: k and v follow q
    q8_append_kernel(kv, a.wk, kv_in, m, 0, m);
    q8_append_kernel(kv, a.wv, kv_in, m, 0, m);
    q8_append_rows(a.q8_o, a.wo, m, m);
}

void make_int8(FFNWeights &f) {
    f.q8_w1 = f.q8_w2 = QMatrix();
    q8_append_kernel(f.q8_w1, f.w1, f.model, f.hidden, 0, f.hidden);
    q8_append_kernel(f.q8_w2, f.w2, f.hidden, f.model, 0, f.model);
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
    const double mm = (double)m * m, km = (double)src_dim * m; // matrix sizes
    float u[kMaxModel], q[kMaxModel], ctx[kMaxModel], y[kMaxModel];
    {
        MRT2_PROF("norm", 4.0 * m, 8.0 * m);
        rmsnorm(x, a.pre_norm, m, 1e-6f, u);
    }
    float *key = keys + (size_t)slot * m, *value = values + (size_t)slot * m;
    if (a.q8_o.rows && !src) {
        // int8, self-attention: q, k and v in one product.
        MRT2_PROF("attn.qkv", 6 * mm, q8_bytes(3 * mm));
        float qkv[3 * kMaxModel];
        q8_matvec(a.q8_in, u, nullptr, qkv);
        std::copy(qkv, qkv + m, q);
        std::copy(qkv + m, qkv + 2 * m, key);
        std::copy(qkv + 2 * m, qkv + 3 * m, value);
    } else if (a.q8_o.rows) {
        // int8, cross-attention: k and v come from src.
        {
            MRT2_PROF("attn.q", 2 * mm, q8_bytes(mm));
            q8_matvec(a.q8_in, u, nullptr, q);
        }
        MRT2_PROF("attn.kv", 4 * km, q8_bytes(2 * km));
        float kv[2 * kMaxModel];
        q8_matvec(a.q8_kv, src, nullptr, kv);
        std::copy(kv, kv + m, key);
        std::copy(kv + m, kv + 2 * m, value);
    } else {
        {
            MRT2_PROF("attn.q", 2 * mm, 4 * mm);
            vecmat(u, m, a.wq, m, m, nullptr, q);
        }
        // Self-attention projects the normalized x; cross-attention projects src.
        MRT2_PROF("attn.kv", 4 * km, 8 * km);
        const float *kv_in = src ? src : u;
        vecmat(kv_in, src_dim, a.wk, m, m, nullptr, key);
        vecmat(kv_in, src_dim, a.wv, m, m, nullptr, value);
    }
    {
        // scores (n x 128 per head) + weighted sum of values: reads all n keys and values
        MRT2_PROF("attn.core", 4.0 * n * m, 8.0 * n * m);
        attend(a, q, keys, values, n, ctx);
    }
    {
        MRT2_PROF("attn.out", 2 * mm, a.q8_o.rows ? q8_bytes(mm) : 4 * mm);
        if (a.q8_o.rows) q8_matvec(a.q8_o, ctx, nullptr, y);
        else matvec(a.wo, m, m, ctx, y); // Wo is [out][in]
    }
    {
        MRT2_PROF("norm", 5.0 * m, 12.0 * m);
        rmsnorm(y, a.post_norm, m, 1e-6f, y);
        for (int i = 0; i < m; i++) x[i] += y[i];
    }
}

void ffn_block(const FFNWeights &f, float *x) {
    const double mh = (double)f.model * f.hidden;
    float u[kMaxModel], y[kMaxModel], h[kMaxHidden]; // 24 KB of stack: fine
    {
        MRT2_PROF("norm", 4.0 * f.model, 8.0 * f.model);
        rmsnorm(x, f.pre_norm, f.model, 1e-6f, u);
    }
    {
        MRT2_PROF("ffn.up", 2 * mh, f.q8_w1.rows ? q8_bytes(mh) : 4 * mh);
        if (f.q8_w1.rows) q8_matvec(f.q8_w1, u, f.b1, h);
        else vecmat(u, f.model, f.w1, f.hidden, f.hidden, f.b1, h);
    }
    {
        MRT2_PROF("gelu", f.hidden, 8.0 * f.hidden);
        gelu_tanh(h, f.hidden);
    }
    {
        MRT2_PROF("ffn.down", 2 * mh, f.q8_w2.rows ? q8_bytes(mh) : 4 * mh);
        if (f.q8_w2.rows) q8_matvec(f.q8_w2, h, f.b2, y);
        else vecmat(h, f.hidden, f.w2, f.model, f.model, f.b2, y);
    }
    {
        MRT2_PROF("norm", 5.0 * f.model, 12.0 * f.model);
        rmsnorm(y, f.post_norm, f.model, 1e-6f, y);
        for (int i = 0; i < f.model; i++) x[i] += y[i];
    }
}

} // namespace mrt2
