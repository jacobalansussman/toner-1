#!/bin/sh
# fetch_checkpoints.sh: download Andrej Karpathy's public llama2.c files into checkpoints/ and check their
# SHA-256. They are MIT-licensed and not committed to this repo (see THIRD_PARTY.md).
#   sh tools/fetch_checkpoints.sh          tok512.bin (6 KB): the 512-token vocabulary that TONER-1 and
#                                          TONER-1-V print with. This is all the print-job build needs.
#   sh tools/fetch_checkpoints.sh --train  also stories260K.pt, stories260K.bin and tok512.model (about 2 MB):
#                                          the base checkpoint and tokenizer for the fine-tune in training/
# TONER-1's fine-tuned weights and everything TONER-1 Pro needs are committed (weights/, training/tokenizer/).
set -e
cd "$(dirname "$0")/.."
mkdir -p checkpoints
HF=https://huggingface.co/karpathy/tinyllamas/resolve/main/stories260K

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

get $HF/tok512.bin          tok512.bin      037cb335abb25d1fa9e8ecae30ed2a3a8ace9302862ebcdc05d51a6bbb10c312
case "${1:-}" in
  --train)
    get $HF/stories260K.pt  stories260K.pt  eec953f9d0f139e894ef8996302680e64b24813c7a98425424f5c85f7cf4abb1
    get $HF/stories260K.bin stories260K.bin b0a507e7ad0f626624f17112325e66691f9076d622e1d3274d103d00299f2696
    get $HF/tok512.model    tok512.model    dfff07d929db979913f166ec94a6f5ecad4c70cfed8eb5c9cbe7e464455e46f5 ;;
  "") ;;
  *) echo "usage: sh tools/fetch_checkpoints.sh [--train]" >&2; exit 2 ;;
esac
