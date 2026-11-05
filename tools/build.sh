#!/bin/sh
# build.sh: assemble the print jobs from their parts into build/, then compare each with the shipped
# copy in print-jobs/. It only writes files; nothing is printed or sent anywhere.
#
#   build/TONER-1_print-job.ps      DSC header + build/weights.ps (tools/export.py) + engine/engine2.ps
#   build/TONER-1-V_print-job.ps    story weights (line-wrapped by tools/wrapws.py) + engine2.ps (its own page
#                                   turned off) + engine/doodle_weights.ps + doodle engine + engine/story_page.ps
#   build/doodle_engine.ps          engine/doodle_engine.in.ps with engine2.ps's forward pass spliced in
#   build/sample_sheet.ps           18 classes x 4 doodles (doodle model only)
#   build/blind_grid.ps             the 36-cell blind test grid (contains the answer key)
#   build/TONER-1-Pro_print-job.ps  only if build/weights15M.ps exists (about 31.6 MB)
#
# usage (from anywhere):  sh tools/build.sh
# needs: build/weights.ps (python3 tools/export.py checkpoints/stories260K.bin checkpoints/tok512.bin build/weights.ps)
set -e
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
B=build
E=engine
SW=$B/weights.ps            # stories260K int8 weights + vocab
DW=$E/doodle_weights.ps     # TONER-1-V int8 weights: defines only /DOODLE
[ -f "$SW" ] || { echo "missing $SW: run tools/fetch_checkpoints.sh and tools/export.py first (see README)" >&2; exit 1; }
mkdir -p $B

# ---------------- TONER-1 (and Pro): header + weights + engine2.ps
mk() { # title weights out
  { printf '%%!PS-Adobe-3.0\n%%%%Title: %s\n%%%%Creator: TONER-1 export.py + engine2.ps\n%%%%BoundingBox: 0 0 612 792\n%%%%LanguageLevel: 2\n%%%%EndComments\n' "$1"
    tail -n +3 "$2"; cat $E/engine2.ps; printf '%%%%EOF\n'; } > "$3"
}
mk 'TONER-1 (stories260K int8, PostScript inference)' $SW $B/TONER-1_print-job.ps
if [ -f $B/weights15M.ps ]; then
  mk 'TONER-1 Pro (stories15M int8, PostScript inference)' $B/weights15M.ps $B/TONER-1-Pro_print-job.ps
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
{ printf '%%!PS-Adobe-3.0\n%%%%Title: TONER-1-V (stories260K story + TONER-1-V doodle, int8, PostScript inference)\n'
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
for j in TONER-1_print-job.ps TONER-1-V_print-job.ps; do
  if cmp -s $B/$j print-jobs/$j; then echo "$j: byte-identical to print-jobs/$j"
  else echo "$j: DIFFERS from print-jobs/$j"; fi
done
if [ -f $B/TONER-1-Pro_print-job.ps ]; then
  # the Pro job is not in this repo (31.6 MB); this is the MD5 of the reference build
  want=245f132c2474045f88b4611f70531d55
  got=$( (md5sum $B/TONER-1-Pro_print-job.ps 2>/dev/null || md5 -r $B/TONER-1-Pro_print-job.ps) | cut -d' ' -f1)
  if [ "$got" = "$want" ]; then echo "TONER-1-Pro_print-job.ps: MD5 $got matches the reference Pro build"
  else echo "TONER-1-Pro_print-job.ps: MD5 $got (reference Pro build: $want)"; fi
fi
ls -l $B/*.ps
