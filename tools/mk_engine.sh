#!/bin/sh
# mk_engine.sh ENGINE2 IN OUT: splice engine2.ps's forward pass (from "/E 2.718281828 def" through the
# end of /forward, byte for byte) into the doodle engine template at the %%ENGINE2-FORWARD%% line.
set -e
E2=$1; IN=$2; OUT=$3
awk '/^\/E 2.718281828 def/{p=1} p{print} /^  logits x EMB mv/{getline; print; exit}' "$E2" > "$OUT.slice"
tail -n 1 "$OUT.slice" | grep -q '^} bind def$'
awk -v f="$OUT.slice" '$0=="%%ENGINE2-FORWARD%%"{while((getline l < f)>0) print l; next} {print}' "$IN" > "$OUT"
rm -f "$OUT.slice"
