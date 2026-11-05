#!/bin/sh
# build_results.sh: score the blind identification test and build its one-page results handout.
#   ../eval_key/blind_guesses.json  the evaluator's guesses (an AI model that was not told the labels)
#   ../eval_key/blind_results.json  scores, written by src/score_blind.py
#   ../eval_key/blind_grid.log      Ghostscript stdout of the blind grid job (the doodles as patch tokens)
#   build/results/results_data.ps   guesses + the blind grid's patch tokens, written by src/make_results_data.py
#   build/results/blind_results.ps  the print job: doodle weights + engine + house style + key + data + page
#   build/results/blind_results.pdf one Ghostscript run (pdfwrite) = docs/TONER-1-V/06_blind_test_results.pdf
# Run tools/build.sh first (it writes build/doodle_engine.ps). Needs Python 3 (PYTHON=..., standard library
# only) and Ghostscript (GS=...). Local Ghostscript only: nothing is printed or sent to any device.
set -e
cd "$(dirname "$0")"
GS=${GS:-gs}
PY=${PYTHON:-python3}
REPO=../..
O=$REPO/build/results
mkdir -p $O
$PY -I src/score_blind.py .. ../eval_key/blind_guesses.json
$PY -I src/make_results_data.py .. $O/results_data.ps
{ printf '%%!PS-Adobe-3.0\n%%%%Title: TONER-1-V blind identification test\n'
  printf '%%%%Creator: TONER-1-V (doodle_weights.ps + doodle_engine.ps + results_page.ps)\n'
  printf '%%%%BoundingBox: 0 0 612 792\n%%%%LanguageLevel: 2\n'
  printf '%%%%DocumentNeededResources: font Helvetica Helvetica-Bold Times-Roman Times-Italic Courier\n%%%%EndComments\n'
  tail -n +3 $REPO/engine/doodle_weights.ps
  cat $REPO/build/doodle_engine.ps $REPO/engine/house.ps ../eval_key/blind_order.ps $O/results_data.ps src/results_page.ps
  printf '%%%%EOF\n'; } > $O/blind_results.ps
$GS -q -dNOPAUSE -dBATCH -sDEVICE=pdfwrite -dPDFSETTINGS=/prepress -o $O/blind_results.pdf $O/blind_results.ps > $O/blind_results.log 2>&1
# the doodles must be the blind grid's, token for token
grep '^D ' $O/blind_results.log > $O/d_results.tmp; grep '^D ' ../eval_key/blind_grid.log > $O/d_grid.tmp
cmp $O/d_results.tmp $O/d_grid.tmp && echo "doodles: identical to the blind grid (36 x 64 patch tokens)"
rm -f $O/d_results.tmp $O/d_grid.tmp
grep -E '^(RESULT|PERCLASS|match|36 doodles|VM)' $O/blind_results.log
ls -l $O/blind_results.ps $O/blind_results.pdf
