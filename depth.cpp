#include "depth.h"
#include "profile.h"
#include <algorithm>
#include <string>

namespace mrt2 {

DepthWeights load_depth(const Weights &w) {
    const std::string d = "params/depthformer/decoder/depth_body/";
    DepthWeights dw;
    dw.embedding = w.f32("params/depthformer/decoder/decoder_embedding/embedding/embedding",
                         {kVocab, kModel});
    dw.adapter = w.f32(d + "depth_input_adapter/kernel", {kModel, kDepthModel});
    for (int l = 0; l < kDepthLayers; l++) {
        const std::string p = d + "transformer/x_layers_" + std::to_string(l) + "/";
        dw.layer[l].self = load_attention(w, p + "self_attention/", kDepthModel, kDepthModel, false);
        dw.layer[l].ffn = load_ffn(w, p + "ffn/", kDepthModel, kDepthFFN);
    }
    dw.ln_scale = w.f32(d + "final_ln/scale", {kDepthModel});
    dw.ln_bias = w.f32(d + "final_ln/bias", {kDepthModel});
    dw.logits_w = w.f32(d + "to_logits/kernel", {kDepthModel, kVocab});
    dw.logits_b = w.f32(d + "to_logits/bias", {kVocab});
    dw.logits_packed.resize((size_t)kLevels * kDepthModel * kCodebook);
    for (int k = 0; k < kLevels; k++)
        for (int i = 0; i < kDepthModel; i++) {
            const float *src = dw.logits_w + (size_t)i * kVocab + kReserved + (size_t)k * kCodebook;
            std::copy(src, src + kCodebook, &dw.logits_packed[((size_t)k * kDepthModel + i) * kCodebook]);
        }
    return dw;
}

void make_int8(DepthWeights &dw) {
    for (auto &L : dw.layer) {
        make_int8(L.self, kDepthModel, true);
        make_int8(L.ffn);
    }
    dw.q8_adapter = QMatrix();
    q8_append_kernel(dw.q8_adapter, dw.adapter, kModel, kDepthModel, 0, kDepthModel);
    for (int k = 0; k < kLevels; k++) {
        dw.q8_logits[k] = QMatrix();
        q8_append_kernel(dw.q8_logits[k], dw.logits_w, kDepthModel, kVocab, kReserved + k * kCodebook, kCodebook);
    }
}

int sample_code(float logits[kCodebook], const Sampling &s, Rng &rng) {
    MRT2_PROF("sample", 8.0 * kCodebook, 4.0 * kCodebook);
    // The soft-cap never changes which logit is largest (greedy picks the
    // same code), but it squeezes large logits together, which raises the
    // chance of the runners-up when sampling with temperature.
    softcap(logits, kCodebook, 30.0f);
    return (int)sample_topk(logits, kCodebook, s.top_k, s.temperature, rng);
}

void depth_frame(const DepthWeights &dw, DepthState &st, const float t[kModel], const Sampling &s,
                 Rng &rng, int32_t tokens[kLevels], DepthDebug *dbg) {
    MRT2_STAGE("depth");
    const size_t layer_slots = (size_t)kLevels * kDepthModel; // one layer's slots

    float input[kModel];
    std::copy(t, t + kModel, input); // step 0 starts from T_t
    for (int k = 0; k < kLevels; k++) {
        // adapter: 1024 -> 768
        float x[kDepthModel];
        {
            const double n = (double)kModel * kDepthModel;
            MRT2_PROF("adapter", 2 * n, dw.q8_adapter.rows ? q8_bytes(n) : 4 * n);
            if (dw.q8_adapter.rows) q8_matvec(dw.q8_adapter, input, nullptr, x);
            else vecmat(input, kModel, dw.adapter, kDepthModel, kDepthModel, nullptr, x);
        }
        if (dbg && dbg->adapter) std::copy(x, x + kDepthModel, dbg->adapter + k * kDepthModel);

        // 2 layers; step k attends over steps 0..k of this frame
        for (int l = 0; l < kDepthLayers; l++) {
            attention_block(dw.layer[l].self, x, nullptr, kDepthModel, &st.keys[l * layer_slots],
                            &st.values[l * layer_slots], k, k + 1);
            ffn_block(dw.layer[l].ffn, x);
        }
        float h[kDepthModel];
        {
            MRT2_PROF("norm", 6.0 * kDepthModel, 8.0 * kDepthModel);
            layernorm(x, dw.ln_scale, dw.ln_bias, kDepthModel, 1e-6f, h);
        }
        if (dbg && dbg->final_ln) std::copy(h, h + kDepthModel, dbg->final_ln + k * kDepthModel);

        // Only this level's 1024 columns of the 12294: a column slice.
        const size_t c0 = kReserved + (size_t)k * kCodebook;
        float logits[kCodebook];
        {
            const double n = (double)kDepthModel * kCodebook;
            MRT2_PROF("logits", 2 * n, dw.q8_logits[k].rows ? q8_bytes(n) : 4 * n);
            if (dw.q8_logits[k].rows) q8_matvec(dw.q8_logits[k], h, dw.logits_b + c0, logits);
            else vecmat(h, kDepthModel, &dw.logits_packed[(size_t)k * kDepthModel * kCodebook], kCodebook,
                        kCodebook, dw.logits_b + c0, logits);
        }
        if (dbg && dbg->logits) std::copy(logits, logits + kCodebook, dbg->logits + k * kCodebook);

        tokens[k] = (int32_t)c0 + sample_code(logits, s, rng);
        if (dbg && dbg->force) tokens[k] = dbg->force[k];

        // The next step's input: the chosen token, embedded and scaled.
        MRT2_PROF("embed", 1.0 * kModel, 8.0 * kModel);
        const float *row = dw.embedding + (size_t)tokens[k] * kModel;
        for (int i = 0; i < kModel; i++) input[i] = row[i] * 32.0f;
    }
}

} // namespace mrt2
