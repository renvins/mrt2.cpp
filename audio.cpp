#include "audio.h"

// One implementation per operating system. Each provides AudioOut::Impl
// plus start() and stop(); only one of the three is compiled.

#if defined(__APPLE__)
// ---- macOS: CoreAudio's default output unit ---------------------------------------
// The "default output" audio unit follows the system's output device and
// converts our format to the device's; it calls render() for each buffer.
#include <AudioToolbox/AudioToolbox.h>

namespace mrt2 {

struct AudioOut::Impl {
    AudioComponentInstance unit = nullptr;
    Fill fill = nullptr;
    void *ctx = nullptr;
};

static OSStatus render(void *ref, AudioUnitRenderActionFlags *, const AudioTimeStamp *, UInt32, UInt32 frames,
                       AudioBufferList *io) {
    auto *im = static_cast<AudioOut::Impl *>(ref);
    // One interleaved buffer (we asked for packed, interleaved stereo).
    im->fill(im->ctx, static_cast<float *>(io->mBuffers[0].mData), (int)frames);
    return noErr;
}

bool AudioOut::start(int sample_rate, Fill fill, void *ctx, std::string &err) {
    stop();
    auto fail = [&](const char *what, OSStatus s) {
        err = std::string("CoreAudio: ") + what + " failed (" + std::to_string((int)s) + ")";
        stop();
        return false;
    };
    impl_ = new Impl{nullptr, fill, ctx};
    AudioComponentDescription d = {};
    d.componentType = kAudioUnitType_Output;
    d.componentSubType = kAudioUnitSubType_DefaultOutput;
    d.componentManufacturer = kAudioUnitManufacturer_Apple;
    AudioComponent comp = AudioComponentFindNext(nullptr, &d);
    if (!comp) return fail("finding the default output", 0);
    OSStatus s = AudioComponentInstanceNew(comp, &impl_->unit);
    if (s) return fail("opening the default output", s);

    AudioStreamBasicDescription f = {};
    f.mSampleRate = sample_rate;
    f.mFormatID = kAudioFormatLinearPCM;
    f.mFormatFlags = kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked; // interleaved
    f.mChannelsPerFrame = 2;
    f.mBitsPerChannel = 32;
    f.mBytesPerFrame = 8;
    f.mFramesPerPacket = 1;
    f.mBytesPerPacket = 8;
    s = AudioUnitSetProperty(impl_->unit, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Input, 0, &f, sizeof f);
    if (s) return fail("setting the format", s);
    AURenderCallbackStruct cb = {render, impl_};
    s = AudioUnitSetProperty(impl_->unit, kAudioUnitProperty_SetRenderCallback, kAudioUnitScope_Input, 0, &cb,
                             sizeof cb);
    if (s) return fail("setting the callback", s);
    if ((s = AudioUnitInitialize(impl_->unit))) return fail("initializing", s);
    if ((s = AudioOutputUnitStart(impl_->unit))) return fail("starting", s);
    return true;
}

void AudioOut::stop() {
    if (!impl_) return;
    if (impl_->unit) {
        AudioOutputUnitStop(impl_->unit); // returns after the last callback finished
        AudioUnitUninitialize(impl_->unit);
        AudioComponentInstanceDispose(impl_->unit);
    }
    delete impl_;
    impl_ = nullptr;
}

} // namespace mrt2

#elif defined(_WIN32)
// ---- Windows: WASAPI, shared mode, event-driven ----------------------------------------
// A thread of ours waits for the device's "need more" event, then asks
// how much room the buffer has and fills it. AUTOCONVERTPCM lets Windows
// convert our 48 kHz float to the device's mix format.
// NOT YET TESTED on Windows (written on macOS).
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <audioclient.h>
#include <mmdeviceapi.h>
#include <atomic>
#include <future>
#include <thread>
#if defined(_MSC_VER)
#pragma comment(lib, "ole32.lib")
#endif

namespace mrt2 {

struct AudioOut::Impl {
    Fill fill = nullptr;
    void *ctx = nullptr;
    std::thread thread;
    std::atomic<bool> quit{false};
};

// KSDATAFORMAT_SUBTYPE_IEEE_FLOAT, written out to avoid needing ksuser.lib.
static const GUID kFloatFormat = {0x00000003, 0x0000, 0x0010, {0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71}};

template <class T> static void release(T *&p) {
    if (p) p->Release();
    p = nullptr;
}

// The whole device lifetime runs on our thread (COM objects stay on the
// thread that made them). `ready` reports "" on success or the error.
static void wasapi_thread(AudioOut::Impl *im, int sample_rate, std::promise<std::string> ready) {
    IMMDeviceEnumerator *en = nullptr;
    IMMDevice *dev = nullptr;
    IAudioClient *client = nullptr;
    IAudioRenderClient *render = nullptr;
    HANDLE event = nullptr;
    bool told = false;
    auto finish = [&](const char *error) {
        if (!told) ready.set_value(error ? std::string("WASAPI: ") + error : std::string());
        told = true;
    };
    const HRESULT com = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    do {
        if (FAILED(CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL, __uuidof(IMMDeviceEnumerator),
                                    (void **)&en))) { finish("no device enumerator"); break; }
        if (FAILED(en->GetDefaultAudioEndpoint(eRender, eConsole, &dev))) { finish("no default output device"); break; }
        if (FAILED(dev->Activate(__uuidof(IAudioClient), CLSCTX_ALL, nullptr, (void **)&client))) {
            finish("cannot open the device");
            break;
        }
        WAVEFORMATEXTENSIBLE fmt = {};
        fmt.Format.wFormatTag = WAVE_FORMAT_EXTENSIBLE;
        fmt.Format.nChannels = 2;
        fmt.Format.nSamplesPerSec = (DWORD)sample_rate;
        fmt.Format.wBitsPerSample = 32;
        fmt.Format.nBlockAlign = 8;
        fmt.Format.nAvgBytesPerSec = (DWORD)sample_rate * 8;
        fmt.Format.cbSize = sizeof(WAVEFORMATEXTENSIBLE) - sizeof(WAVEFORMATEX);
        fmt.Samples.wValidBitsPerSample = 32;
        fmt.dwChannelMask = SPEAKER_FRONT_LEFT | SPEAKER_FRONT_RIGHT;
        fmt.SubFormat = kFloatFormat;
        const DWORD flags = AUDCLNT_STREAMFLAGS_EVENTCALLBACK | AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM |
                            AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY;
        const REFERENCE_TIME buffer = 200000; // 20 ms, in units of 100 ns
        if (FAILED(client->Initialize(AUDCLNT_SHAREMODE_SHARED, flags, buffer, 0, (WAVEFORMATEX *)&fmt, nullptr))) {
            finish("the device refused 48 kHz stereo float");
            break;
        }
        event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        UINT32 frames = 0;
        if (!event || FAILED(client->SetEventHandle(event)) || FAILED(client->GetBufferSize(&frames)) ||
            FAILED(client->GetService(__uuidof(IAudioRenderClient), (void **)&render)) || FAILED(client->Start())) {
            finish("cannot start the stream");
            break;
        }
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);
        finish(nullptr);
        while (!im->quit.load()) {
            WaitForSingleObject(event, 100);
            UINT32 pad = 0;
            if (FAILED(client->GetCurrentPadding(&pad))) break; // device gone
            const UINT32 n = frames - pad;
            BYTE *data = nullptr;
            if (n == 0 || FAILED(render->GetBuffer(n, &data))) continue;
            im->fill(im->ctx, reinterpret_cast<float *>(data), (int)n);
            render->ReleaseBuffer(n, 0);
        }
        client->Stop();
    } while (false);
    finish("unknown error"); // no-op if already reported
    if (event) CloseHandle(event);
    release(render);
    release(client);
    release(dev);
    release(en);
    if (SUCCEEDED(com)) CoUninitialize();
}

bool AudioOut::start(int sample_rate, Fill fill, void *ctx, std::string &err) {
    stop();
    impl_ = new Impl;
    impl_->fill = fill;
    impl_->ctx = ctx;
    std::promise<std::string> ready;
    std::future<std::string> result = ready.get_future();
    impl_->thread = std::thread(wasapi_thread, impl_, sample_rate, std::move(ready));
    err = result.get();
    if (!err.empty()) {
        stop();
        return false;
    }
    return true;
}

void AudioOut::stop() {
    if (!impl_) return;
    impl_->quit.store(true);
    if (impl_->thread.joinable()) impl_->thread.join();
    delete impl_;
    impl_ = nullptr;
}

} // namespace mrt2

#elif defined(__linux__)
// ---- Linux: ALSA (libasound, part of every desktop Linux) ----------------------------
// A thread of ours fills 10 ms at a time; snd_pcm_writei() blocks until the
// device has room, which paces the loop. "default" goes through the
// system's sound server (PulseAudio / PipeWire) when there is one.
// NOT YET TESTED on Linux (written on macOS).
#include <alsa/asoundlib.h>
#include <atomic>
#include <thread>

namespace mrt2 {

struct AudioOut::Impl {
    Fill fill = nullptr;
    void *ctx = nullptr;
    snd_pcm_t *pcm = nullptr;
    std::thread thread;
    std::atomic<bool> quit{false};
};

bool AudioOut::start(int sample_rate, Fill fill, void *ctx, std::string &err) {
    stop();
    impl_ = new Impl;
    impl_->fill = fill;
    impl_->ctx = ctx;
    int r = snd_pcm_open(&impl_->pcm, "default", SND_PCM_STREAM_PLAYBACK, 0);
    if (r == 0)
        r = snd_pcm_set_params(impl_->pcm, SND_PCM_FORMAT_FLOAT_LE, SND_PCM_ACCESS_RW_INTERLEAVED, 2,
                               (unsigned)sample_rate, 1 /* allow resampling */, 60000 /* latency, us */);
    if (r < 0) {
        err = std::string("ALSA: ") + snd_strerror(r);
        stop();
        return false;
    }
    Impl *im = impl_;
    impl_->thread = std::thread([im, sample_rate] {
        const int n = sample_rate / 100; // 10 ms
        float buf[2 * 4800];
        while (!im->quit.load()) {
            im->fill(im->ctx, buf, n);
            snd_pcm_sframes_t w = snd_pcm_writei(im->pcm, buf, (snd_pcm_uframes_t)n);
            if (w < 0) snd_pcm_recover(im->pcm, (int)w, 1); // underrun or suspend: restart quietly
        }
    });
    return true;
}

void AudioOut::stop() {
    if (!impl_) return;
    impl_->quit.store(true);
    if (impl_->thread.joinable()) impl_->thread.join();
    if (impl_->pcm) {
        snd_pcm_drop(impl_->pcm);
        snd_pcm_close(impl_->pcm);
    }
    delete impl_;
    impl_ = nullptr;
}

} // namespace mrt2

#else
// ---- anything else: no audio output yet ----------------------------------------------
namespace mrt2 {
struct AudioOut::Impl {};
bool AudioOut::start(int, Fill, void *, std::string &err) {
    err = "no audio output on this system yet (macOS, Windows and Linux are supported)";
    return false;
}
void AudioOut::stop() {}
} // namespace mrt2
#endif
