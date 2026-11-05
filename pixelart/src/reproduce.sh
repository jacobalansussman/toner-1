#!/bin/sh
# Reproduce TONER-1-V end to end: data, training, sampling, judge, export, and the Ghostscript checks.
# Run from anywhere; nothing is sent to any printer.
#   sh pixelart/src/reproduce.sh [path/to/engine2.ps] [path/to/story weights.ps]
# Needs Python with torch, numpy and pillow (PYTHON=..., default python3), Ghostscript (GS=..., default gs),
# about 2 GB of free disk for data/, and the story weights from tools/export.py (build/weights.ps) for the
# last check. Training takes about 12 minutes on an M4 (MPS); a different machine or torch version
# will not give a bit-identical checkpoint, so the shipped one is in ckpt/.
set -e
ROOT=$(cd "$(dirname "$0")/.." && pwd)
REPO=$(cd "$ROOT/.." && pwd)
PY=${PYTHON:-python3}
GS=${GS:-gs}
ENGINE=${1:-$REPO/engine/engine2.ps}
STORY=${2:-$REPO/build/weights.ps}
NAME=d48L4_16k   # final model; the 230K baseline was: d64L5 --steps 8000 (default shape)
mkdir -p "$ROOT/data" && cd "$ROOT/data"                 # never run Python from inside data/raw
$PY -I "$ROOT/src/fetch_data.py" "$ROOT" 10000           # Range-request the first 10,000 drawings/class
$PY -I "$ROOT/src/explore_threshold.py" "$ROOT"          # out/prep_variants.png (threshold choice)
$PY -I "$ROOT/src/preprocess.py" "$ROOT"                 # data/proc/{train,val}.npy
$PY -I "$ROOT/src/train.py" "$ROOT" $NAME --dim 48 --layers 4 --heads 6 --kv 3 --hidden 128 --steps 16000 --eval_every 2000  # ckpt/$NAME.pt
$PY -I "$ROOT/src/sample.py" "$ROOT" $NAME --temps 1.0 0.8            # float weights
$PY -I "$ROOT/src/sample.py" "$ROOT" $NAME --temps 1.0 0.9 0.8 --quant    # int8 weights
$PY -I "$ROOT/src/classifier.py" "$ROOT" train                         # judge CNN on real 16x16 images
$PY -I "$ROOT/src/sample.py" "$ROOT" $NAME --temps 0.9 --n 64 --seed 1 --tag _n64 --no_sheet --quant
$PY -I "$ROOT/src/classifier.py" "$ROOT" score "$ROOT/out/samples_${NAME}_int8_n64_t09.npy"
$PY -I "$ROOT/src/export.py" "$ROOT" $NAME               # out/doodle_weights.ps, doodle_meta.json, reference_logits.json
$PY -I "$ROOT/src/check_ps.py" "$ROOT" "$ENGINE" "$ROOT/out/ps_check" "$GS"            # engine2.ps forward in Ghostscript vs reference
$PY -I "$ROOT/src/check_ps.py" "$ROOT" "$ENGINE" "$ROOT/out/ps_check" "$GS" "$STORY"   # mode B: next to the story model's weights
echo "To ship the new weights: cp $ROOT/out/doodle_weights.ps $REPO/engine/ && sh $REPO/tools/build.sh"
