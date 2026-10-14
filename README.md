# mrt2.cpp

**Google's [Magenta RealTime 2](https://magenta.withgoogle.com/magenta-realtime-2) live music model, running in real time on an ordinary CPU.** Pure C++20, no dependencies, no Python, no GPU. Windows, Linux and macOS.

Type a style, music starts playing; type another one and the music follows, while it plays.

> Demo video: coming soon.

## Quick start

```sh
git clone https://github.com/renvins/mrt2.cpp && cd mrt2.cpp
cmake -S . -B build && cmake --build build --config Release
scripts/download_weights.sh             # 1.1 GB, Google's weights (Windows: scripts\download_weights.ps1)
build/mrt2 live --style techno          # Windows: build\Release\mrt2.exe live --style techno
```

On macOS and Linux, plain `make` works too (it puts `mrt2` in the top folder). Linux needs the ALSA headers to build: `sudo apt install libasound2-dev`.

## Playing

While `mrt2 live` plays, type a command and press Enter:

| Command | What it does |
|---|---|
| `jazz`, `techno`, `lofi`, `strings`, `ambient`, `funk`, `folk`, `metal` | switch style (it follows within ~0.2 s) |
| `cfg 5` | follow the style more (−1..7, default 3) |
| `temp 1.0` | less random (default 1.3) |
| `tokens a,b,...` | your own 12 style tokens |
| `stats` | speed and dropouts so far |
| `q` | stop (with `-o file.wav`, the session is saved) |

Or render a file: `mrt2 generate --style lofi --seconds 60 -o lofi.wav`. `mrt2` with no arguments lists every option.

## Speed

Milliseconds to generate 40 ms of music (lower is better; under 40 = real time):

| Machine | ms per 40 ms | Real time |
|---|---|---|
| Apple M4 MacBook (10 cores) | 14 | 2.8× |
| GitHub runner, Linux ARM64 (4 cores) | 33 | 1.2× |
| GitHub runner, Linux x64 (AMD EPYC 9V74, 2 cores) | 39 | 1.0× |
| GitHub runner, Linux x64 (AMD EPYC 7763, 2 cores) | 43 | 0.94× |
| GitHub runner, Windows x64 (AMD EPYC 7763, 2 cores) | 45 | 0.89× |
| GitHub runner, Windows ARM64 (4 cores) | 60 | 0.67× |

GitHub's runners are small shared virtual machines (2–4 cores, slow memory): roughly a weak laptop. A normal 4–8 core PC is several times faster. Numbers are measured on every push by CI (`tools/bench`); help us fill this table with your machine: run `build/bench <model path>` and open an issue.

## How it works

Every 40 ms the model makes 12 tokens and the codec turns them into 1920 stereo samples at 48 kHz:

```
style + controls -> encoder -> temporal transformer (12 layers) -> depth transformer (12 steps)
                 -> 12 tokens -> SpectroStream codec (2D convs, streaming) -> inverse FFT -> audio
```

- **Checked against Google.** Tests compare every stage with Google's own implementation: in exact mode (`--fp32`), the same 600 tokens out of 600 and audio within one 16-bit step. The default fast mode is checked too: its sampling distribution stays within 0.004 nats of float32, and its audio 76 dB above the difference. `ctest` runs them all (the reference data is in the `test-data` release).
- **Fast, the boring way.** SIMD kernels for AVX2 and ARM NEON chosen at run time (one binary for every PC), int8 transformer weights (as Google's own Mac engine does), fp16 for the codec's first layers (on x86 and Apple chips, where it is faster), Winograd 3×3 convolutions, a streaming codec, a lock-free audio queue. `--fp32` gives exact float32 everywhere.
- **No dependencies.** Your OS's own audio API (CoreAudio, WASAPI, ALSA) and the C++ standard library. That's it.

## Status

Working: live playback and steering, file rendering, all five platforms built and tested on every push (including address, undefined-behaviour and thread sanitizers).

Not yet: live audio on Windows and Linux is new (tested on macOS; reports welcome). Next: a CLAP/VST3 plugin for DAWs, MIDI input (the model takes notes and drums too), and text prompts ("warm jazz piano") instead of presets.

## License and credits

Code: Apache-2.0. The model and its weights are Google's [Magenta RealTime 2](https://github.com/magenta/magenta-realtime) (weights CC-BY 4.0); this project is an independent reimplementation, not affiliated with Google.
