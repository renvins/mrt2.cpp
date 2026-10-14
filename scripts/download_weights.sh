#!/usr/bin/env bash
# Download the mrt2_small model (1.1 GB; Google's Magenta RealTime 2 weights,
# CC-BY 4.0) to where mrt2 looks for it by default:
#   ~/models/magenta-rt-v2/checkpoints/mrt2_small.safetensors
# Usage: scripts/download_weights.sh [directory]
set -euo pipefail

URL=https://huggingface.co/google/magenta-realtime-2/resolve/main/checkpoints/mrt2_small.safetensors
SHA256=5dd1cbc7c606c512c21de0bcb04d4818bf0a3b873d7cbb9d1556d67d3b034de3
DIR=${1:-$HOME/models/magenta-rt-v2/checkpoints}
FILE="$DIR/mrt2_small.safetensors"

sha() { if command -v sha256sum > /dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -d' ' -f1; }

if [ -f "$FILE" ] && [ "$(sha "$FILE")" = "$SHA256" ]; then
    echo "already there: $FILE"
    exit 0
fi
mkdir -p "$DIR"
echo "downloading mrt2_small (1.1 GB) to $FILE"
curl -fL --retry 3 -C - -o "$FILE.part" "$URL" # -C -: resumes an interrupted download
echo "checking the file's fingerprint (sha256)..."
if [ "$(sha "$FILE.part")" != "$SHA256" ]; then
    echo "error: the download is damaged (wrong sha256); delete $FILE.part and try again" >&2
    exit 1
fi
mv "$FILE.part" "$FILE"
echo "done: $FILE"
