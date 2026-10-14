// audio.h - play stereo float audio through the operating system's own
// audio API, with no libraries: CoreAudio on macOS, WASAPI on Windows,
// ALSA on Linux. The OS converts our 48 kHz to the device's rate.
//
//   AudioOut out;
//   out.start(48000, fill, &state, err);  // fill() now runs on the OS's audio thread
//   ...
//   out.stop();
//
// The OS calls fill(ctx, out, frames) whenever the device needs more
// sound: write `frames` stereo frames (2 floats each, left then right,
// range -1..1) into `out`. It runs on a real-time thread, so it must be
// quick and must never wait: no locks, no memory allocation, no I/O. In
// practice it only copies from a SpscRing (ring.h).
#pragma once
#include <string>

namespace mrt2 {

class AudioOut {
public:
    using Fill = void (*)(void *ctx, float *out, int frames);

    AudioOut() = default;
    ~AudioOut() { stop(); }
    AudioOut(const AudioOut &) = delete;
    AudioOut &operator=(const AudioOut &) = delete;

    // Open the default output device and start calling fill. Returns false
    // (with a message in err) if there is no device or the format is refused.
    bool start(int sample_rate, Fill fill, void *ctx, std::string &err);

    // Stop calling fill and release the device. Safe to call twice.
    void stop();

    struct Impl; // per operating system, in audio.cpp
private:
    Impl *impl_ = nullptr;
};

} // namespace mrt2
