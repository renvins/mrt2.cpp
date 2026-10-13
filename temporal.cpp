#include "temporal.h"
#include "profile.h"
#include <algorithm>
#include <string>

namespace mrt2 {

// ---- loading ----------------------------------------------------------------

TemporalWeights load_temporal(const Weights &w) {
    TemporalWeights tw;
    tw.embedding = w.f32("params/depthformer/decoder/decoder_embedding/embedding/embedding",
                         {kVocab, kModel});
    for (int l = 0; l < kTemporalLayers; l++) {
        const std::string p = "params/depthformer/decoder/temporal_body/transformer/x_layers_" +
                              std::to_string(l) + "/";
        TemporalWeights::Layer &L = tw.layer[l];
        L.self = load_attention(w, p + "self_attention/", kModel, kModel, true);
        L.cross = load_attention(w, p + "cross_attention/", kModel, 256, true); // k, v from e[256]
        L.ffn = load_ffn(w, p + "ffn/", kModel, kFFN);
    }
    return tw;
}

void make_int8(TemporalWeights &tw) {
    for (auto &L : tw.layer) {
        make_int8(L.self, kModel, true);
        make_int8(L.cross, 256, false);
        make_int8(L.ffn);
    }
}

TemporalState::TemporalState()
    : self_k((size_t)kTemporalLayers * kSlots * kModel), self_v(self_k.size()),
      cross_k(self_k.size()), cross_v(self_k.size()) {}

// ---- the input --------------------------------------------------------------

void temporal_input(const TemporalWeights &tw, const int32_t prev[kLevels], float x[kModel]) {
    MRT2_STAGE("temporal");
    MRT2_PROF("embed", 2.0 * kLevels * kModel, 4.0 * kLevels * kModel);
    // Google: embed each token, scale by sqrt(1024) = 32, then average the
    // 12 levels. Same order of operations here.
    double sum[kModel] = {};
    for (int k = 0; k < kLevels; k++) {
        const float *row = tw.embedding + (size_t)prev[k] * kModel;
        for (int i = 0; i < kModel; i++) sum[i] += row[i] * 32.0f;
    }
    for (int i = 0; i < kModel; i++) x[i] = (float)(sum[i] / kLevels);
}

// ---- one frame ------------------------------------------------------------------

void temporal_step(const TemporalWeights &tw, TemporalState &st, const float x_in[kModel],
                   const float e[256], float out[kModel], TemporalDebug *dbg) {
    // Ring buffer: frame t writes slot t % 42. Until the buffer is full the
    // valid slots are 0..t; afterwards all 42 are valid.
    MRT2_STAGE("temporal");
    const int slot = (int)(st.frames % kSlots);
    const int n = st.frames + 1 < kSlots ? (int)st.frames + 1 : kSlots;

    float x[kModel];
    for (int i = 0; i < kModel; i++) x[i] = x_in[i];
    for (int l = 0; l < kTemporalLayers; l++) {
        const TemporalWeights::Layer &L = tw.layer[l];
        const size_t off = (size_t)l * kSlots * kModel;
        attention_block(L.self, x, nullptr, kModel, &st.self_k[off], &st.self_v[off], slot, n);
        if (dbg && dbg->l0_self && l == 0) std::copy(x, x + kModel, dbg->l0_self);
        attention_block(L.cross, x, e, 256, &st.cross_k[off], &st.cross_v[off], slot, n);
        if (dbg && dbg->l0_cross && l == 0) std::copy(x, x + kModel, dbg->l0_cross);
        ffn_block(L.ffn, x);
        if (dbg && dbg->layers) std::copy(x, x + kModel, dbg->layers + l * kModel);
    }
    for (int i = 0; i < kModel; i++) out[i] = x[i];
    st.frames++;
}

} // namespace mrt2
