#!/bin/sh
# build.sh: assemble the print jobs from their parts into build/, then compare each with the shipped
# copy in print-jobs/. It only writes files; nothing is printed or sent anywhere.
#
#   build/TONER-1_print-job.ps      DSC header + build/weights.ps (tools/export.py) + a MODELNAME line + engine/engine2.ps
#   build/TONER-1-Pro_print-job.ps  DSC header + build/weights_pro.ps + a MODELNAME line + engine/engine2.ps
#                                   (only if build/weights_pro.ps exists; about 24.9 MB)
#   build/TONER-1-V_print-job.ps    story weights (line-wrapped by tools/wrapws.py) + engine2.ps (its own page
#                                   turned off) + engine/doodle_weights.ps + doodle engine + engine/story_page.ps
#   build/doodle_engine.ps          engine/doodle_engine.in.ps with engine2.ps's forward pass spliced in
#   build/sample_sheet.ps           18 classes x 4 doodles (doodle model only)
#   build/blind_grid.ps             the 36-cell blind test grid (contains the answer key)
#
# usage (from anywhere):  sh tools/build.sh      (sh tools/rebuild.sh runs every step from a fresh clone)
# needs: build/weights.ps, and build/weights_pro.ps for Pro (see README, "Reproduce")
set -e
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
B=build
E=engine
SW=$B/weights.ps            # TONER-1 int8 weights + vocab (stories260K, printer fine-tune)
PW=$B/weights_pro.ps        # TONER-1 Pro int8 weights + vocab (our 12M model and tokenizer)
DW=$E/doodle_weights.ps     # TONER-1-V int8 weights: defines only /DOODLE
[ -f "$SW" ] || { echo "missing $SW: run sh tools/rebuild.sh, or the steps in the README" >&2; exit 1; }
mkdir -p $B

# ---------------- TONER-1 and Pro: header + weights + a MODELNAME line (the page footer) + engine2.ps
mk() { # title weights out modelname
  { printf '%%!PS-Adobe-3.0\n%%%%Title: %s\n%%%%Creator: TONER-1 export.py + engine2.ps\n%%%%BoundingBox: 0 0 612 792\n%%%%LanguageLevel: 2\n%%%%EndComments\n' "$1"
    tail -n +3 "$2"; printf '/MODELNAME (%s) def\n' "$4"; cat $E/engine2.ps; printf '%%%%EOF\n'; } > "$3"
}
mk 'TONER-1 (stories260K printer fine-tune, int8, PostScript inference)' $SW $B/TONER-1_print-job.ps 'stories260K printer-tuned'
if [ -f $PW ]; then
  mk 'TONER-1 Pro (12M int8, PostScript inference)' $PW $B/TONER-1-Pro_print-job.ps 'TONER-1 Pro 12M'
else
  echo "note: no $PW, so TONER-1 Pro is not built (see README, \"Reproduce\")"
fi

# ---------------- TONER-1-V
sh tools/mk_engine.sh $E/engine2.ps $E/doodle_engine.in.ps $B/doodle_engine.ps
# the spliced section must be byte-identical to engine2.ps's
awk '/^\/E 2.718281828 def/{p=1} p{print} /^  logits x EMB mv/{getline; print; exit}' $E/engine2.ps > $B/e2.tmp
awk '/^\/E 2.718281828 def/{p=1} p{print} /^  logits x EMB mv/{getline; print; exit}' $B/doodle_engine.ps > $B/de.tmp
cmp $B/e2.tmp $B/de.tmp && rm -f $B/e2.tmp $B/de.tmp

hdr() { # title
  printf '%%!PS-Adobe-3.0\n%%%%Title: %s\n%%%%Creator: TONER-1-V (doodle_weights.ps + doodle_engine.ps)\n' "$1"
  printf '%%%%BoundingBox: 0 0 612 792\n%%%%LanguageLevel: 2\n%%%%DocumentNeededResources: font Helvetica Helvetica-Bold Times-Roman Times-Italic Courier\n%%%%EndComments\n'
}
# sample sheet and blind grid: doodle model only
{ hdr 'TONER-1-V sample sheet'; tail -n +3 $DW; cat $B/doodle_engine.ps $E/house.ps $E/sample_sheet_page.ps; printf '%%%%EOF\n'; } > $B/sample_sheet.ps
{ hdr 'TONER-1-V blind grid'; tail -n +3 $DW; cat $B/doodle_engine.ps $E/house.ps pixelart/eval_key/blind_order.ps $E/blind_grid_page.ps; printf '%%%%EOF\n'; } > $B/blind_grid.ps

# the combined print job: story first (engine2.ps generates into TEXT; PAGE false skips its own page),
# then the doodle model, the router and the illustrated page
{ printf '%%!PS-Adobe-3.0\n%%%%Title: TONER-1-V (stories260K printer fine-tune story + TONER-1-V doodle, int8, PostScript inference)\n'
  printf '%%%%Creator: TONER-1 export.py + engine2.ps; TONER-1-V export.py + doodle_engine.ps + story_page.ps\n'
  printf '%%%%BoundingBox: 0 0 612 792\n%%%%LanguageLevel: 2\n'
  printf '%%%%DocumentNeededResources: font Helvetica Helvetica-Bold Times-Roman Times-Italic Courier\n%%%%EndComments\n'
  tail -n +3 $SW | $PY -I tools/wrapws.py; printf '\n'
  printf '%% ---- the illustrated page below replaces engine2.ps'"'"'s own page ----\n/PAGE false def\n'
  cat $E/engine2.ps
  tail -n +3 $DW
  cat $B/doodle_engine.ps $E/story_page.ps
  printf '%%%%EOF\n'; } > $B/TONER-1-V_print-job.ps

# ---------------- compare with the shipped jobs
st=0
for j in TONER-1_print-job.ps TONER-1-Pro_print-job.ps TONER-1-V_print-job.ps; do
  [ -f $B/$j ] || continue
  if cmp -s $B/$j print-jobs/$j; then echo "$j: byte-identical to print-jobs/$j"
  else echo "$j: DIFFERS from print-jobs/$j"; st=1; fi
done
ls -l $B/*.ps
exit $st
