#!/bin/sh
# make_pages.sh: re-render the PDFs in docs/ into build/pages/, one genuine Ghostscript run of a print
# job per PDF (pdfwrite, /prepress, fonts embedded), one at a time so the timings printed on the pages
# are not inflated by each other. Local Ghostscript only: nothing is printed or sent to any device.
# Writes build/pages/runs.tsv (one row per run) and a .log per page (the job's stdout).
#   sh tools/make_pages.sh          everything except the Pro page
#   PRO=1 sh tools/make_pages.sh    also the Pro page (needs build/TONER-1-Pro_print-job.ps; about 2 minutes)
# Run tools/build.sh first. Sampled stories depend on the interpreter's rand, so they match the
# shipped pages only under the same Ghostscript (10.07.1 was used).
set -e
cd "$(dirname "$0")/.."
GS=${GS:-gs}
B=build; P=build/pages
mkdir -p $P
printf 'pdf\tjob\tseed\ttemp\tgs_stdout_last_line\tgs_command\n' > $P/runs.tsv
run() { # out.pdf job seed temp
  if [ -n "$3" ]; then opts="-dSEED=$3 -dTEMP=$4"; else opts=""; fi
  cmd="$GS -q -dNOPAUSE -dBATCH -sDEVICE=pdfwrite -dPDFSETTINGS=/prepress $opts -o $P/$1 $2"
  $cmd > "$P/${1%.pdf}.log" 2>&1
  last=$(grep -E 'tok/s|doodles,' "$P/${1%.pdf}.log" | tail -n 1)
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "${3:--}" "${4:--}" "$last" "$cmd" >> $P/runs.tsv
  echo "$1: $last"
}
runv() { # out.pdf seed  (TONER-1-V job: story TEMP 0.8, doodle DTEMP 0.9, doodle seed = SEED)
  cmd="$GS -q -dNOPAUSE -dBATCH -sDEVICE=pdfwrite -dPDFSETTINGS=/prepress -dSEED=$2 -o $P/$1 $B/TONER-1-V_print-job.ps"
  $cmd > "$P/${1%.pdf}.log" 2>&1
  r=$(grep 'router:' "$P/${1%.pdf}.log" | sed 's/^TONER-1-V | [0-9]* params | //')
  printf '%s\t%s\t%s\t-\t%s\t%s\n' "$1" "$B/TONER-1-V_print-job.ps" "$2" "$r" "$cmd" >> $P/runs.tsv
  echo "$1: $r"
}

# ---- TONER-1 (docs/TONER-1/)
run 01_explainer.pdf docs/TONER-1/src/explainer.ps
run 02_model-card-and-benchmarks.pdf docs/TONER-1/src/modelcard.ps
run 03_story_seed20_t0.8.pdf $B/TONER-1_print-job.ps 20 0.8
run 04_story_seed40_t0.8.pdf $B/TONER-1_print-job.ps 40 0.8
run 05_story_seed22_t0.8.pdf $B/TONER-1_print-job.ps 22 0.8
run 06_story_greedy_t0.pdf   $B/TONER-1_print-job.ps 42 0
run 07_story_seed6_t1.4.pdf  $B/TONER-1_print-job.ps 6 1.4
if [ -n "$PRO" ]; then run 08_story_pro-15M_seed42_t0.8.pdf $B/TONER-1-Pro_print-job.ps 42 0.8; fi

# ---- TONER-1-V (docs/TONER-1-V/)
run V01_pareto_frontier.pdf docs/TONER-1-V/src/pareto.ps
run V02_sample_sheet.pdf $B/sample_sheet.ps
runv V03_story_illustrated_seed7.pdf 7
runv V04_story_illustrated_seed14.pdf 14
runv V05_story_illustrated_seed25.pdf 25
# the blind grid that the evaluator saw (unlabeled; the key is in pixelart/eval_key/)
run blind_grid.pdf $B/blind_grid.ps
# 06_blind_test_results.pdf is built by pixelart/results/build_results.sh
ls -l $P/*.pdf
