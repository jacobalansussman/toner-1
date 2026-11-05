#!/bin/sh
# render_check.sh SEED TEMP NAME [job]  ->  build/check/NAME-1.png (110 dpi) + build/check/NAME.log
# A quick look at one story page in local Ghostscript. Nothing is printed or sent to any device.
# job defaults to print-jobs/TONER-1_print-job.ps
set -e
cd "$(dirname "$0")/.."
[ -n "$3" ] || { echo "usage: sh tools/render_check.sh SEED TEMP NAME [job]" >&2; exit 2; }
job=${4:-print-jobs/TONER-1_print-job.ps}
mkdir -p build/check
${GS:-gs} -q -dNOPAUSE -dBATCH -sDEVICE=png16m -r110 -dSEED=$1 -dTEMP=$2 -o build/check/$3-%d.png $job > build/check/$3.log 2>&1
tail -n 2 build/check/$3.log
ls build/check/$3-*.png
