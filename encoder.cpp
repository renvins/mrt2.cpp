#include "encoder.h"
#include "ops.h"
#include <cassert>
#include <cmath>

namespace mrt2 {

// Token ids: 0..5 are reserved, 6 is the "dropout" id (= no condition),
// real values start at 7. So a stored token is value + 7, and -1 -> 6.
constexpr int kTokenOffset = 7;

// Each channel's vocabulary = its codebook size + 7 (reserved + dropout),
// except the two CFG channels, which were trained without dropout: + 6
// (see the quirk described in encode()).
constexpr int kStyleVocab = 1024 + 7;  // 1031
constexpr int kPitchVocab = 4 + 7;     // 11
constexpr int kDrumVocab = 2 + 7;      // 9
constexpr int kCfgVocab = 41 + 6;      // 47
constexpr int kCfgDrumVocab = 9 + 6;   // 15
constexpr int kRegular = kCondTokens - kStyleTokens; // 132 channels

// The 132 "regular" channels share one table: channel c's rows start at
// the sum of the vocab sizes of the channels before it.
struct RegularLayout {
    int vocab[kRegular];
    int offset[kRegular];
    int rows; // total, 1526 (the table is padded to 1536)
    constexpr RegularLayout() : vocab(), offset(), rows(0) {
        for (int c = 0; c < kRegular; c++) {
            vocab[c] = c < 128 ? kPitchVocab : c == 128 ? kDrumVocab
                     : c < 131 ? kCfgVocab : kCfgDrumVocab;
            offset[c] = rows;
            rows += vocab[c];
        }
    }
};
static constexpr RegularLayout kLayout; // computed at compile time
static_assert(kLayout.rows == 1526, "regular embedding layout");

Conditioning::Conditioning() {
    for (int &s : style) s = -1;
    for (int &p : pitch) p = -1;
}

int discretize_cfg(double value, double step, int max_bin) {
    // Same arithmetic as Google's Python (all in double), so the rounding
    // matches bit for bit.
    // Clamp exactly like Python's max(-1.0, min(7.0, value)): min() keeps
    // 7.0 unless value < 7.0, max() keeps -1.0 unless value > -1.0. A NaN
    // compares false both times, so it becomes 7.0 as in Google's code
    // (a naive clamp would let NaN through, and (int)NaN is undefined
    // behaviour in C++).
    double v = value < 7.0 ? value : 7.0;
    v = v > -1.0 ? v : -1.0;
    const double x = (v + 1.0) / step;
    // Python's round() rounds half to EVEN (6.5 -> 6, 7.5 -> 8), unlike
    // std::round (6.5 -> 7). Written out explicitly so it can't depend on
    // the CPU's rounding mode.
    double r = std::floor(x);
    const double frac = x - r;
    if (frac > 0.5 || (frac == 0.5 && std::fmod(r, 2.0) != 0.0)) r += 1.0;
    int bin = (int)r;
    return bin < 0 ? 0 : bin > max_bin ? max_bin : bin;
}

bool encoder_tokens(const Conditioning &c, int32_t out[kCondTokens], const char **err) {
    // `limit` = number of real values (codebook size); -1 is always allowed.
    auto put = [&](int idx, int value, int limit, const char *what) {
        if (value < -1 || value >= limit) {
            if (err) *err = what;
            return false;
        }
        out[idx] = value + kTokenOffset;
        return true;
    };
    for (int i = 0; i < kStyleTokens; i++)
        if (!put(i, c.style[i], 1024, "style token out of range")) return false;
    for (int p = 0; p < kPitches; p++)
        if (!put(12 + p, c.pitch[p], 4, "piano-roll state out of range")) return false;
    if (!put(140, c.drums, 2, "drum state out of range")) return false;
    // CFG strengths are always set (they are never "dropped").
    out[141] = discretize_cfg(c.cfg_musiccoca, 0.2, 40) + kTokenOffset;
    out[142] = discretize_cfg(c.cfg_notes, 0.2, 40) + kTokenOffset;
    out[143] = discretize_cfg(c.cfg_drums, 1.0, 8) + kTokenOffset;
    return true;
}

EncoderWeights load_encoder(const Weights &w) {
    const std::string br = "params/depthformer/encoder/body/layers_1/";
    const std::string ln = "params/depthformer/encoder/body/encoder_ln/";
    EncoderWeights e;
    e.style_emb = w.f32(br + "branched_mulan_embedder/mulan_embedder/mulan_dequantizer/embedding",
                        {kStyleTokens * kStyleVocab, 768});
    e.adapter = w.f32(br + "branched_mulan_embedder/mulan_embedder/depth_input_adapter/kernel",
                      {768, kEncDim});
    e.regular_emb = w.f32(br + "branched_regular_embedder/regular_embedder/embedding",
                          {1536, kEncDim});
    e.ln_scale = w.f32(ln + "scale", {kEncDim});
    e.ln_bias = w.f32(ln + "bias", {kEncDim});
    return e;
}

void encode(const EncoderWeights &e, const int32_t tokens[kCondTokens], float out[kEncDim],
            EncoderDebug *dbg) {
    // --- MusicCoCa branch: sum the 12 token embeddings, then project ----
    // Level k's tokens live in rows [k * 1031, (k + 1) * 1031).
    float sum[768] = {};
    for (int k = 0; k < kStyleTokens; k++) {
        assert(tokens[k] >= 0 && tokens[k] < kStyleVocab);
        const float *row = e.style_emb + (size_t)(k * kStyleVocab + tokens[k]) * 768;
        for (int i = 0; i < 768; i++) sum[i] += row[i];
    }
    float mulan[kEncDim];
    vecmat(sum, 768, e.adapter, kEncDim, kEncDim, nullptr, mulan);

    // --- regular branch: mean of the 132 channel embeddings ---------------
    //
    // Google quirk, reproduced on purpose: CFG tokens get +7 like every
    // channel, but the CFG vocabularies are only codebook + 6. So the
    // largest CFG bin (strength 7.0) reads one row past its channel: the
    // next channel's first row, or (drums) an untrained padding row of the
    // 1536-row table. Verified against Google's model; we match it rather
    // than "fix" it, so the only requirement is to stay inside the table.
    float regular[kEncDim] = {};
    for (int c = 0; c < kRegular; c++) {
        const int t = tokens[kStyleTokens + c];
        assert(t >= 0 && kLayout.offset[c] + t < 1536);
        const float *row = e.regular_emb + (size_t)(kLayout.offset[c] + t) * kEncDim;
        for (int i = 0; i < kEncDim; i++) regular[i] += row[i];
    }
    for (float &v : regular) v /= kRegular;

    // --- combine (average of the two branches), then LayerNorm ----------
    float combined[kEncDim];
    for (int i = 0; i < kEncDim; i++) combined[i] = (mulan[i] + regular[i]) * 0.5f;
    layernorm(combined, e.ln_scale, e.ln_bias, kEncDim, 1e-6f, out);

    if (dbg) {
        for (int i = 0; i < kEncDim; i++) {
            if (dbg->mulan) dbg->mulan[i] = mulan[i];
            if (dbg->regular) dbg->regular[i] = regular[i];
            if (dbg->combined) dbg->combined[i] = combined[i];
        }
    }
}

} // namespace mrt2
