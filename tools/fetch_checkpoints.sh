#!/bin/sh
# fetch_checkpoints.sh: download Andrej Karpathy's public llama2.c checkpoints into checkpoints/ and
# check their SHA-256. They are not committed to this repo (MIT, see THIRD_PARTY.md).
#   sh tools/fetch_checkpoints.sh          stories260K.bin + tok512.bin        (about 1 MB, for TONER-1 and TONER-1-V)
#   sh tools/fetch_checkpoints.sh --pro    also stories15M.bin + tokenizer.bin (about 61 MB, for TONER-1 Pro)
set -e
cd "$(dirname "$0")/.."
mkdir -p checkpoints
HF=https://huggingface.co/karpathy/tinyllamas/resolve/main
GH=https://raw.githubusercontent.com/karpathy/llama2.c/master

sha() { (shasum -a 256 "$1" 2>/dev/null || sha256sum "$1") | cut -d' ' -f1; }
get() { # url file sha256
  if [ -f "checkpoints/$2" ] && [ "$(sha "checkpoints/$2")" = "$3" ]; then
    echo "checkpoints/$2: already present, SHA-256 OK"; return
  fi
  echo "checkpoints/$2: downloading $1"
  curl -fsSL --proto '=https' --retry 3 -o "checkpoints/$2.part" "$1"
  got=$(sha "checkpoints/$2.part")
  if [ "$got" != "$3" ]; then
    rm -f "checkpoints/$2.part"
    echo "checkpoints/$2: SHA-256 mismatch (got $got, want $3)" >&2; exit 1
  fi
  mv "checkpoints/$2.part" "checkpoints/$2"
  echo "checkpoints/$2: downloaded, SHA-256 OK"
}

get $HF/stories260K/stories260K.bin stories260K.bin b0a507e7ad0f626624f17112325e66691f9076d622e1d3274d103d00299f2696
get $HF/stories260K/tok512.bin      tok512.bin      037cb335abb25d1fa9e8ecae30ed2a3a8ace9302862ebcdc05d51a6bbb10c312
if [ "$1" = "--pro" ]; then
  get $HF/stories15M.bin            stories15M.bin  cd590644d963867a2b6e5a1107f51fad663c41d79c149fbecbbb1f95fa81f49a
  get $GH/tokenizer.bin             tokenizer.bin   50a52ef822ee9e83de5ce9d0be0a025a773d019437f58b5ff9dcafb063ece361
fi
