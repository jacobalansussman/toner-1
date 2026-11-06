#!/bin/sh
# rebuild.sh: rebuild all three print jobs from a fresh clone and check them byte for byte against print-jobs/.
#   sh tools/rebuild.sh            (needs sh, curl and Python 3; standard library only; about a minute)
# 1. tools/fetch_checkpoints.sh downloads tok512.bin, Karpathy's 512-token vocabulary (SHA-256 checked).
# 2. tools/l2c_safetensors.py turns the committed weights/*/model.safetensors into llama2.c .bin files.
# 3. tools/export.py writes the int8 PostScript weights, checked by MD5.
# 4. tools/build.sh assembles the jobs and compares them with print-jobs/.
# Everything is written to build/ and checkpoints/, which git ignores. Nothing is printed or sent anywhere.
set -e
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
md5of() { $PY -I -c 'import hashlib,sys; print(hashlib.md5(open(sys.argv[1],"rb").read()).hexdigest())' "$1"; }
want() { # file md5
  got=$(md5of "$1")
  if [ "$got" = "$2" ]; then echo "$1: MD5 $got OK"; else echo "$1: MD5 $got, expected $2" >&2; exit 1; fi
}
sh tools/fetch_checkpoints.sh
mkdir -p build
$PY -I tools/l2c_safetensors.py to-bin weights/TONER-1 build/toner1.bin
$PY -I tools/l2c_safetensors.py to-bin weights/TONER-1-Pro build/toner1_pro.bin
$PY -I tools/export.py build/toner1.bin checkpoints/tok512.bin build/weights.ps
want build/weights.ps 838ba89393e424c3fc7cff15e9d28f16
$PY -I tools/export.py build/toner1_pro.bin training/tokenizer/tok4096.bin build/weights_pro.ps
want build/weights_pro.ps 70b3c06decb0c88672123294f94c2edb
sh tools/build.sh
