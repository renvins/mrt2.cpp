// mrt2.cpp - music with Magenta RealTime 2, entirely in C++.
//
//   mrt2 live --style techno                     play it now; type to steer it
//   mrt2 generate --style techno --seconds 10 -o out.wav
//   mrt2 styles
//
// The pipeline, once per 40 ms frame: conditioning -> encoder -> temporal
// decoder -> depth decoder (12 tokens) -> codebook vector -> streaming
// codec -> 1920 stereo samples at 48 kHz. Nothing is held back, so any
// length works.
//
// By default the transformer runs with int8 weights (quant.h) and the
// codec's first layers with fp16 weights; --fp32 for exact float32.
#include "audio.h"
#include "codec.h"
#include "cpu.h"
#include "depth.h"
#include "encoder.h"
#include "ring.h"
#include "temporal.h"
#include "threads.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
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
                 "usage: mrt2 live [options]               play music now, steer it by typing\n"
                 "       mrt2 generate [options] -o out.wav\n"
                 "       mrt2 styles\n"
                 "\noptions:\n"
                 "  -m, --model PATH       mrt2_small.safetensors\n"
                 "                         (default ~/models/magenta-rt-v2/checkpoints/mrt2_small.safetensors)\n"
                 "  --style NAME           a preset from `mrt2 styles` (default jazz)\n"
                 "  --style-tokens A,B,..  12 MusicCoCa tokens 0..1023 (instead of --style)\n"
                 "  --temperature T        sampling temperature, 0 = greedy (default 1.3)\n"
                 "  --top-k K              sample among the K best tokens (default 40)\n"
                 "  --seed N               random seed: same seed, same music (default 0)\n"
                 "  --cfg-style X          how strongly to follow the style, -1..7 (default 3)\n"
                 "  -t, --threads N        CPU threads to use (default: all)\n"
                 "  --fp32                 exact float32 weights everywhere (slower; default: int8\n"
                 "                         transformer, fp16 for the codec's first layers)\n"
                 "  -o, --out PATH         output .wav (stereo, 48 kHz, 16-bit); for live, a recording\n"
                 "generate only:\n"
                 "  -s, --seconds N        length, 1..3600 (default 10)\n"
                 "live only:\n"
                 "  --buffer-ms N          audio generated ahead, 80..2000 (default 200): more = safer\n"
                 "                         against slow frames, but changes take longer to be heard\n");
}

static double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

static std::string default_model() {
    const char *home = std::getenv("HOME");
    if (!home) home = std::getenv("USERPROFILE"); // Windows
    return std::string(home ? home : ".") + "/models/magenta-rt-v2/checkpoints/mrt2_small.safetensors";
}

static const Preset *find_preset(const std::string &name) {
    for (const Preset &p : kPresets)
        if (name == p.name) return &p;
    return nullptr;
}

// "1,2,3,..." -> 12 style tokens; throws if it isn't exactly 12 numbers.
static std::vector<int> parse_tokens(const std::string &list) {
    std::vector<int> v;
    for (size_t pos = 0; pos <= list.size();) {
        size_t comma = list.find(',', pos);
        if (comma == std::string::npos) comma = list.size();
        v.push_back(std::atoi(list.substr(pos, comma - pos).c_str()));
        pos = comma + 1;
    }
    if (v.size() != 12) throw std::runtime_error("style tokens: need exactly 12 numbers");
    return v;
}

// ---- options ------------------------------------------------------------------------

struct Options {
    std::string model = default_model(), out;
    const Preset *preset = &kPresets[0]; // the style, unless custom tokens are given
    std::vector<int> custom;
    double seconds = 10;
    int buffer_ms = 200;
    Sampling sampling; // temperature 1.3, top-k 40
    uint64_t seed = 0;
    int threads = (int)std::thread::hardware_concurrency();
    bool int8 = true;
    Conditioning cond;
};

static Options parse_options(int argc, char **argv, bool live) {
    Options o;
    for (int i = 2; i < argc; i++) {
        const std::string a = argv[i];
        auto value = [&]() -> const char * {
            if (i + 1 >= argc) throw std::runtime_error("missing value after " + a);
            return argv[++i];
        };
        if (a == "-m" || a == "--model") o.model = value();
        else if (a == "-o" || a == "--out") o.out = value();
        else if (!live && (a == "-s" || a == "--seconds")) o.seconds = std::atof(value());
        else if (live && a == "--buffer-ms") o.buffer_ms = std::atoi(value());
        else if (a == "--temperature") o.sampling.temperature = (float)std::atof(value());
        else if (a == "--top-k") o.sampling.top_k = std::atoi(value());
        else if (a == "--seed") o.seed = std::strtoull(value(), nullptr, 10);
        else if (a == "--cfg-style") o.cond.cfg_musiccoca = std::atof(value());
        else if (a == "-t" || a == "--threads") o.threads = std::atoi(value());
        else if (a == "--fp32") o.int8 = false;
        else if (a == "--style") {
            const std::string name = value();
            o.preset = find_preset(name);
            if (!o.preset) throw std::runtime_error("unknown style '" + name + "' (see `mrt2 styles`)");
        } else if (a == "--style-tokens") o.custom = parse_tokens(value());
        else throw std::runtime_error("unknown option " + a);
    }
    if (!live && o.out.empty()) throw std::runtime_error("missing -o out.wav");
    if (o.seconds < 1 || o.seconds > 3600) throw std::runtime_error("--seconds must be 1..3600");
    if (o.buffer_ms < 80 || o.buffer_ms > 2000) throw std::runtime_error("--buffer-ms must be 80..2000");
    if (o.sampling.top_k < 1 || o.sampling.top_k > (int)kMaxTopK) throw std::runtime_error("--top-k must be 1..256");
    if (o.threads < 1) throw std::runtime_error("--threads must be >= 1");
    for (int k = 0; k < 12; k++) o.cond.style[k] = o.custom.empty() ? o.preset->tokens[k] : o.custom[k];
    return o;
}

// ---- the engine ---------------------------------------------------------------------

// The whole model and its streaming state: set the controls, then each
// frame() call makes the next 40 ms of audio.
struct Engine {
    Weights w;
    EncoderWeights ew;
    TemporalWeights tw;
    DepthWeights dw;
    CodecWeights cw;
    TemporalState ts;
    DepthState ds;
    CodecStream cs;
    Rng rng{0};
    Sampling sampling;
    int32_t prev[kLevels] = {}; // the first frame starts from SOS (all 0)
    float e[256] = {};          // the encoded conditioning
    double tokens_ms = 0, codec_ms = 0; // time spent, in total

    void load(const std::string &path, bool int8) {
        std::string err;
        if (!w.open(path, err)) throw std::runtime_error(err);
        ew = load_encoder(w);
        tw = load_temporal(w);
        dw = load_depth(w);
        cw = load_codec(w);
        if (int8) { // "fast": int8 transformer, and fp16 early codec layers where that helps
            make_int8(tw);
            make_int8(dw);
            if (cpu_prefers_fp16_weights()) make_fp16(cw);
        }
    }

    // Throws if a value is out of range (check with valid() first where a
    // throw would be unwelcome).
    void set_conditioning(const Conditioning &c) {
        int32_t tok[kCondTokens];
        const char *bad = nullptr;
        if (!encoder_tokens(c, tok, &bad)) throw std::runtime_error(bad);
        encode(ew, tok, e); // the same for every frame until the next change
    }

    void frame(float audio[2 * kSamplesPerFrame]) {
        auto a = std::chrono::steady_clock::now();
        float x[kModel], t[kModel], z[256];
        temporal_input(tw, prev, x);
        temporal_step(tw, ts, x, e, t);
        depth_frame(dw, ds, t, sampling, rng, prev); // becomes the next frame's input
        tokens_ms += ms_since(a);
        auto b = std::chrono::steady_clock::now();
        dequantize(cw, prev, z);
        decode_frame(cw, cs, z, audio);
        codec_ms += ms_since(b);
    }
};

static bool valid(const Conditioning &c, std::string &why) {
    int32_t tok[kCondTokens];
    const char *bad = nullptr;
    if (encoder_tokens(c, tok, &bad)) return true;
    why = bad;
    return false;
}

// Model loading shared by both commands; prints what it is doing.
static void start_engine(Engine &eng, const Options &o, const char *what) {
    prefer_performance_cores();
    set_threads(o.threads);
    auto t0 = std::chrono::steady_clock::now();
    eng.load(o.model, o.int8);
    eng.rng = Rng(o.seed);
    eng.sampling = o.sampling;
    eng.set_conditioning(o.cond);
    std::printf("model loaded in %.0f ms\n", ms_since(t0));
    std::printf("%s | style: %s | temperature %.2f | top-k %d | seed %llu | %d threads | %s\n", what,
                o.custom.empty() ? o.preset->prompt : "custom tokens", o.sampling.temperature, o.sampling.top_k,
                (unsigned long long)o.seed, o.threads, !o.int8 ? "float32" : cpu_prefers_fp16_weights() ? "int8 + fp16" : "int8");
}

// ---- mrt2 generate --------------------------------------------------------------------

static int generate(int argc, char **argv) {
    const Options o = parse_options(argc, argv, false);
    Engine eng;
    char what[64];
    std::snprintf(what, sizeof what, "%.0f s", o.seconds);
    start_engine(eng, o, what);

    const int frames = (int)(o.seconds * 25 + 0.5);
    std::vector<int16_t> pcm((size_t)frames * kSamplesPerFrame * 2);
    float audio[2 * kSamplesPerFrame];
    double worst = 0;
    for (int f = 0; f < frames; f++) {
        auto a = std::chrono::steady_clock::now();
        eng.frame(audio);
        worst = std::max(worst, ms_since(a));
        for (int i = 0; i < 2 * kSamplesPerFrame; i++) pcm[(size_t)f * 2 * kSamplesPerFrame + i] = to_int16(audio[i]);
        if ((f + 1) % 25 == 0 || f + 1 == frames) {
            std::printf("\r%3d / %d s", (f + 1) / 25, frames / 25);
            std::fflush(stdout);
        }
    }
    if (!write_wav(o.out.c_str(), pcm, kSampleRate)) throw std::runtime_error("cannot write " + o.out);

    const double total = eng.tokens_ms + eng.codec_ms, audio_ms = frames * 40.0;
    std::printf("\nper 40 ms frame: tokens %.1f ms + codec %.1f ms = %.1f ms (slowest frame %.1f ms)\n",
                eng.tokens_ms / frames, eng.codec_ms / frames, total / frames, worst);
    std::printf("wrote %s: %.1f s of audio in %.1f s (%.2fx real time)\n", o.out.c_str(), audio_ms / 1000,
                total / 1000, audio_ms / total);
    return 0;
}

// ---- mrt2 live ------------------------------------------------------------------------
//
// Three threads:
//   generator  makes frames while the queue holds less than the buffer
//              target, and applies control changes between frames;
//   audio      (the OS's) copies from the queue to the speakers; if the
//              queue runs dry it plays silence and counts a dropout;
//   main       reads commands typed in the terminal.
// The queue (ring.h) is lock-free, so the audio thread never waits. The
// mutex below is only between main and the generator.

struct Live {
    explicit Live(size_t target) : ring(target), target(target) {}
    SpscRing ring;
    const size_t target;             // floats to keep queued (the buffer)
    std::atomic<bool> quit{false};
    std::atomic<long> dropouts{0};   // audio callbacks that found the queue short
    std::atomic<long> frames{0};     // frames generated
    std::atomic<double> worst_ms{0}; // slowest frame so far
    std::atomic<double> tokens_ms{0}, codec_ms{0}; // totals, copied from the engine after each frame
    std::mutex mu;                   // guards the three fields below
    Conditioning cond;
    Sampling sampling;
    bool changed = false;
};

static void live_fill(void *ctx, float *out, int frames) {
    Live &L = *static_cast<Live *>(ctx);
    const size_t want = (size_t)frames * 2, got = L.ring.read(out, want);
    if (got < want) {
        std::fill(out + got, out + want, 0.0f); // silence rather than garbage
        L.dropouts.fetch_add(1, std::memory_order_relaxed);
    }
}

static void live_help() {
    std::printf("type a style and press Enter:");
    for (const Preset &p : kPresets) std::printf(" %s", p.name);
    std::printf("\nor:  tokens A,B,...  (12 style tokens)   cfg X  (style strength -1..7)\n"
                "     temp T  (temperature)   topk K   stats   help   q  (quit)\n");
}

static int live(int argc, char **argv) {
    const Options o = parse_options(argc, argv, true);
    Engine eng;
    start_engine(eng, o, "live");

    const int ahead = (o.buffer_ms + 39) / 40; // frames generated ahead of the speakers
    Live L((size_t)ahead * 2 * kSamplesPerFrame);
    L.cond = o.cond;
    L.sampling = o.sampling;
    std::vector<int16_t> recording;
    const bool record = !o.out.empty();

    std::thread generator([&] {
        float audio[2 * kSamplesPerFrame];
        long seen_dropouts = 0, next_report = 0;
        while (!L.quit.load()) {
            if (L.ring.size() + 2 * kSamplesPerFrame > L.target) {
                std::this_thread::sleep_for(std::chrono::milliseconds(2)); // queue full: the speakers are behind us
                continue;
            }
            {
                std::lock_guard<std::mutex> lock(L.mu);
                if (L.changed) { // checked by the main thread, so this can't throw
                    eng.set_conditioning(L.cond);
                    eng.sampling = L.sampling;
                    L.changed = false;
                }
            }
            auto a = std::chrono::steady_clock::now();
            eng.frame(audio);
            const double ms = ms_since(a);
            L.ring.write(audio, 2 * kSamplesPerFrame); // fits: checked above
            if (ms > L.worst_ms.load()) L.worst_ms.store(ms);
            L.tokens_ms.store(eng.tokens_ms); // the main thread reads these copies, never eng's
            L.codec_ms.store(eng.codec_ms);
            L.frames.fetch_add(1);
            if (record)
                for (float v : audio) recording.push_back(to_int16(v));
            // Report dropouts at most every 5 s (125 frames), so a slow
            // machine gets one line now and then, not one per frame.
            const long d = L.dropouts.load();
            if (d > seen_dropouts && L.frames.load() >= next_report) {
                std::printf("(%ld dropouts: generation is falling behind the speakers; try --buffer-ms %d"
                            " or a faster setting)\n", d - seen_dropouts, 2 * o.buffer_ms);
                std::fflush(stdout);
                seen_dropouts = d;
                next_report = L.frames.load() + 125;
            }
        }
    });

    // Fill the buffer before the sound starts, so the first frames are safe.
    while (L.ring.size() + 2 * kSamplesPerFrame <= L.target) std::this_thread::sleep_for(std::chrono::milliseconds(5));
    L.dropouts.store(0);
    AudioOut out;
    std::string err;
    if (!out.start(kSampleRate, live_fill, &L, err)) {
        L.quit.store(true);
        generator.join();
        throw std::runtime_error(err);
    }
    std::printf("playing (%d ms buffered: a change is heard after about that long)\n", ahead * 40);
    live_help();

    std::string line;
    while (std::printf("> "), std::fflush(stdout), std::getline(std::cin, line)) {
        // "cmd arg": the first word, and the rest without leading spaces.
        const size_t sp = line.find(' ');
        const std::string cmd = line.substr(0, sp);
        std::string arg = sp == std::string::npos ? "" : line.substr(sp + 1);
        arg.erase(0, arg.find_first_not_of(' '));
        if (cmd.empty()) continue;
        if (cmd == "q" || cmd == "quit" || cmd == "exit") break;
        if (cmd == "help" || cmd == "?") {
            live_help();
            continue;
        }
        if (cmd == "stats") {
            const long n = L.frames.load();
            const double tok = L.tokens_ms.load(), codec = L.codec_ms.load();
            std::printf("%ld frames (%.1f s): %.1f ms per 40 ms frame on average (tokens %.1f + codec %.1f),"
                        " slowest %.1f ms, %ld dropouts\n",
                        n, n * 0.04, n ? (tok + codec) / n : 0.0, n ? tok / n : 0.0, n ? codec / n : 0.0,
                        L.worst_ms.load(), L.dropouts.load());
            continue;
        }
        // Everything else changes the controls: build the new ones, check
        // them here, and hand them to the generator.
        Conditioning c;
        Sampling s;
        {
            std::lock_guard<std::mutex> lock(L.mu);
            c = L.cond;
            s = L.sampling;
        }
        std::string msg;
        try {
            if (const Preset *p = find_preset(cmd)) {
                for (int k = 0; k < 12; k++) c.style[k] = p->tokens[k];
                msg = std::string("style: ") + p->prompt;
            } else if (cmd == "tokens") {
                const std::vector<int> v = parse_tokens(arg);
                for (int k = 0; k < 12; k++) c.style[k] = v[k];
                msg = "style: custom tokens";
            } else if (cmd == "cfg") {
                c.cfg_musiccoca = std::stod(arg);
                msg = "style strength " + arg;
            } else if (cmd == "temp") {
                s.temperature = std::stof(arg);
                if (s.temperature < 0) throw std::runtime_error("temperature must be >= 0");
                msg = "temperature " + arg;
            } else if (cmd == "topk") {
                s.top_k = std::stoi(arg);
                if (s.top_k < 1 || s.top_k > (int)kMaxTopK) throw std::runtime_error("top-k must be 1..256");
                msg = "top-k " + arg;
            } else {
                std::printf("unknown command '%s' (type help)\n", cmd.c_str());
                continue;
            }
        } catch (const std::exception &e) { // std::stod & co. throw on non-numbers
            std::printf("not changed: %s\n", e.what());
            continue;
        }
        std::string why;
        if (!valid(c, why)) {
            std::printf("not changed: %s\n", why.c_str());
            continue;
        }
        {
            std::lock_guard<std::mutex> lock(L.mu);
            L.cond = c;
            L.sampling = s;
            L.changed = true;
        }
        std::printf("-> %s\n", msg.c_str());
    }

    out.stop();
    L.quit.store(true);
    generator.join();
    const long n = L.frames.load();
    std::printf("\nstopped after %.1f s: %.1f ms per 40 ms frame on average, slowest %.1f ms, %ld dropouts\n",
                n * 0.04, n ? (eng.tokens_ms + eng.codec_ms) / n : 0.0, L.worst_ms.load(), L.dropouts.load());
    if (record) {
        if (!write_wav(o.out.c_str(), recording, kSampleRate)) throw std::runtime_error("cannot write " + o.out);
        std::printf("recording: %s (%.1f s)\n", o.out.c_str(), recording.size() / 2.0 / kSampleRate);
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc >= 2 && std::strcmp(argv[1], "styles") == 0) {
        for (const Preset &p : kPresets) std::printf("%-8s %s\n", p.name, p.prompt);
        return 0;
    }
    const bool is_live = argc >= 2 && std::strcmp(argv[1], "live") == 0;
    if (argc < 2 || (!is_live && std::strcmp(argv[1], "generate") != 0)) {
        usage();
        return 1;
    }
    try {
        return is_live ? live(argc, argv) : generate(argc, argv);
    } catch (const std::exception &e) {
        std::fprintf(stderr, "mrt2: %s\n", e.what());
        return 1;
    }
}
