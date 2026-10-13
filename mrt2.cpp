// mrt2.cpp - generate music with Magenta RealTime 2, entirely in C++.
//
//   mrt2 generate --style techno --seconds 10 -o out.wav
//   mrt2 styles
//
// The pipeline, once per 40 ms frame: conditioning -> encoder -> temporal
// decoder -> depth decoder (12 tokens) -> codebook vector -> streaming
// codec -> 1920 stereo samples at 48 kHz. Nothing is held back, so any
// length works.
//
// The transformer runs with int8 weights by default (quant.h; --fp32 for
// exact float32).
#include "codec.h"
#include "depth.h"
#include "encoder.h"
#include "temporal.h"
#include "threads.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace mrt2;

// Style presets: 12 MusicCoCa tokens each, made with tools/style_tokens.py
// (text -> tokens needs Google's MusicCoCa model, not ported yet).
struct Preset {
    const char *name, *prompt;
    int tokens[12];
};
static const Preset kPresets[] = {
    {"jazz", "jazz piano trio", {555, 402, 72, 424, 482, 932, 954, 153, 33, 374, 519, 779}},
    {"techno", "techno", {117, 790, 167, 916, 301, 30, 644, 623, 187, 197, 577, 129}},
    {"lofi", "lofi hip hop", {827, 97, 6, 449, 914, 694, 731, 110, 698, 613, 850, 823}},
    {"strings", "string quartet", {508, 758, 1008, 336, 902, 379, 290, 713, 971, 379, 765, 78}},
    {"ambient", "ambient synth pads", {369, 693, 55, 893, 843, 790, 227, 800, 813, 540, 32, 415}},
    {"funk", "funk bass and drums", {1003, 399, 668, 68, 515, 886, 373, 739, 655, 708, 784, 936}},
    {"folk", "acoustic folk guitar", {722, 688, 682, 994, 533, 614, 245, 608, 394, 598, 253, 285}},
    {"metal", "heavy metal", {322, 825, 598, 131, 826, 656, 592, 139, 89, 410, 778, 642}},
};

static void usage() {
    std::fprintf(stderr,
                 "usage: mrt2 generate [options] -o out.wav\n"
                 "       mrt2 styles\n"
                 "\noptions:\n"
                 "  -m, --model PATH       mrt2_small.safetensors\n"
                 "                         (default ~/models/magenta-rt-v2/checkpoints/mrt2_small.safetensors)\n"
                 "  --style NAME           a preset from `mrt2 styles` (default jazz)\n"
                 "  --style-tokens A,B,..  12 MusicCoCa tokens 0..1023 (instead of --style)\n"
                 "  -s, --seconds N        length, 1..3600 (default 10)\n"
                 "  --temperature T        sampling temperature, 0 = greedy (default 1.3)\n"
                 "  --top-k K              sample among the K best tokens (default 40)\n"
                 "  --seed N               random seed: same seed, same music (default 0)\n"
                 "  --cfg-style X          how strongly to follow the style, -1..7 (default 3)\n"
                 "  -t, --threads N        CPU threads to use (default: all)\n"
                 "  --fp32                 float32 transformer weights (exact, slower; default int8)\n"
                 "  -o, --out PATH         output .wav (stereo, 48 kHz, 16-bit)\n");
}

static double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

static std::string default_model() {
    const char *home = std::getenv("HOME");
    if (!home) home = std::getenv("USERPROFILE"); // Windows
    return std::string(home ? home : ".") + "/models/magenta-rt-v2/checkpoints/mrt2_small.safetensors";
}

static int generate(int argc, char **argv) {
    std::string model = default_model(), out;
    const Preset *preset = &kPresets[0];
    std::vector<int> custom;
    double seconds = 10;
    Sampling sampling; // temperature 1.3, top-k 40
    uint64_t seed = 0;
    int threads = (int)std::thread::hardware_concurrency();
    bool int8 = true;
    Conditioning cond;

    for (int i = 2; i < argc; i++) {
        const std::string a = argv[i];
        auto value = [&]() -> const char * {
            if (i + 1 >= argc) throw std::runtime_error("missing value after " + a);
            return argv[++i];
        };
        if (a == "-m" || a == "--model") model = value();
        else if (a == "-o" || a == "--out") out = value();
        else if (a == "-s" || a == "--seconds") seconds = std::atof(value());
        else if (a == "--temperature") sampling.temperature = (float)std::atof(value());
        else if (a == "--top-k") sampling.top_k = std::atoi(value());
        else if (a == "--seed") seed = std::strtoull(value(), nullptr, 10);
        else if (a == "--cfg-style") cond.cfg_musiccoca = std::atof(value());
        else if (a == "-t" || a == "--threads") threads = std::atoi(value());
        else if (a == "--fp32") int8 = false;
        else if (a == "--style") {
            const std::string name = value();
            preset = nullptr;
            for (const Preset &p : kPresets)
                if (name == p.name) preset = &p;
            if (!preset) throw std::runtime_error("unknown style '" + name + "' (see `mrt2 styles`)");
        } else if (a == "--style-tokens") {
            std::string list = value();
            for (size_t pos = 0; pos <= list.size();) {
                size_t comma = list.find(',', pos);
                if (comma == std::string::npos) comma = list.size();
                custom.push_back(std::atoi(list.substr(pos, comma - pos).c_str()));
                pos = comma + 1;
            }
            if (custom.size() != 12) throw std::runtime_error("--style-tokens needs exactly 12 numbers");
        } else throw std::runtime_error("unknown option " + a);
    }
    if (out.empty()) throw std::runtime_error("missing -o out.wav");
    if (seconds < 1 || seconds > 3600) throw std::runtime_error("--seconds must be 1..3600");
    if (sampling.top_k < 1 || sampling.top_k > (int)kMaxTopK) throw std::runtime_error("--top-k must be 1..256");
    if (threads < 1) throw std::runtime_error("--threads must be >= 1");
    prefer_performance_cores();
    set_threads(threads);
    for (int k = 0; k < 12; k++) cond.style[k] = custom.empty() ? preset->tokens[k] : custom[k];

    int32_t tokens[kCondTokens];
    const char *bad = nullptr;
    if (!encoder_tokens(cond, tokens, &bad)) throw std::runtime_error(bad);

    auto t0 = std::chrono::steady_clock::now();
    Weights w;
    std::string err;
    if (!w.open(model, err)) throw std::runtime_error(err);
    const EncoderWeights ew = load_encoder(w);
    TemporalWeights tw = load_temporal(w);
    DepthWeights dw = load_depth(w);
    const CodecWeights cw = load_codec(w);
    if (int8) {
        make_int8(tw);
        make_int8(dw);
    }
    std::printf("model loaded in %.0f ms\n", ms_since(t0));
    std::printf("style: %s | %.0f s | temperature %.2f | top-k %d | seed %llu | %d threads | %s\n",
                custom.empty() ? preset->prompt : "custom tokens", seconds, sampling.temperature, sampling.top_k,
                (unsigned long long)seed, threads, int8 ? "int8" : "float32");

    // The conditioning is the same for every frame here, so encode it once.
    float e[256];
    encode(ew, tokens, e);

    const int frames = (int)(seconds * 25 + 0.5);
    TemporalState ts;
    DepthState ds;
    CodecStream cs;
    Rng rng(seed);
    std::vector<int16_t> pcm((size_t)frames * kSamplesPerFrame * 2);
    int32_t prev[kLevels] = {}; // the first frame starts from SOS (all 0)
    float x[kModel], t[kModel], z[256], audio[2 * kSamplesPerFrame];
    double t_tokens = 0, t_codec = 0, worst = 0;
    for (int f = 0; f < frames; f++) {
        auto a = std::chrono::steady_clock::now();
        temporal_input(tw, prev, x);
        temporal_step(tw, ts, x, e, t);
        depth_frame(dw, ds, t, sampling, rng, prev); // becomes the next frame's input
        const double ms_tok = ms_since(a);
        auto b = std::chrono::steady_clock::now();
        dequantize(cw, prev, z);
        decode_frame(cw, cs, z, audio);
        const double ms_codec = ms_since(b);
        for (int i = 0; i < 2 * kSamplesPerFrame; i++) pcm[(size_t)f * 2 * kSamplesPerFrame + i] = to_int16(audio[i]);
        t_tokens += ms_tok;
        t_codec += ms_codec;
        worst = std::max(worst, ms_tok + ms_codec);
        if ((f + 1) % 25 == 0 || f + 1 == frames) {
            std::printf("\r%3d / %d s", (f + 1) / 25, frames / 25);
            std::fflush(stdout);
        }
    }
    if (!write_wav(out.c_str(), pcm, kSampleRate)) throw std::runtime_error("cannot write " + out);

    const double total = t_tokens + t_codec, audio_ms = frames * 40.0;
    std::printf("\nper 40 ms frame: tokens %.1f ms + codec %.1f ms = %.1f ms (slowest frame %.1f ms)\n",
                t_tokens / frames, t_codec / frames, total / frames, worst);
    std::printf("wrote %s: %.1f s of audio in %.1f s (%.2fx real time)\n", out.c_str(), audio_ms / 1000,
                total / 1000, audio_ms / total);
    return 0;
}

int main(int argc, char **argv) {
    if (argc >= 2 && std::strcmp(argv[1], "styles") == 0) {
        for (const Preset &p : kPresets) std::printf("%-8s %s\n", p.name, p.prompt);
        return 0;
    }
    if (argc < 2 || std::strcmp(argv[1], "generate") != 0) {
        usage();
        return 1;
    }
    try {
        return generate(argc, argv);
    } catch (const std::exception &e) {
        std::fprintf(stderr, "mrt2: %s\n", e.what());
        return 1;
    }
}
