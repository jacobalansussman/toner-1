#!/usr/bin/env python3
"""cost_check.py: look up every logged generation id (requests >= FIRST_REQ) with GET /api/v1/generation
and sum total_cost + upstream_inference_cost. Writes logs/cost_check_bulk.json. No key material is printed.
Usage: python -I cost_check.py [FIRST_REQ]
"""
import concurrent.futures as cf
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import TT as TRAIN  # noqa: E402
import or_free  # noqa: E402

first = int(sys.argv[1]) if len(sys.argv) > 1 else 133
log = [json.loads(l) for l in open(os.path.join(TRAIN, "logs/gen_requests.jsonl"))]
ids = [(r["req"], r["generation_id"]) for r in log if r["req"] >= first and r.get("generation_id")]


def look(item):
    req, gid = item
    for k in range(5):
        st, d, _ = or_free.http("GET", f"/generation?id={gid}", timeout=30)
        if st == 200:
            dd = d.get("data", {})
            return req, gid, float(dd.get("total_cost") or 0), float(dd.get("upstream_inference_cost") or 0), \
                dd.get("model"), dd.get("is_byok")
        time.sleep(2 + 2 * k)
    return req, gid, None, None, None, None


t0 = time.time()
with cf.ThreadPoolExecutor(3) as ex:
    res = list(ex.map(look, ids))
found = [r for r in res if r[2] is not None]
out = {"measured_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "first_req": first, "generation_ids": len(ids),
       "looked_up": len(found), "not_found": [r[1] for r in res if r[2] is None],
       "sum_total_cost": sum(r[2] for r in found), "sum_upstream_inference_cost": sum(r[3] for r in found),
       "nonzero": [r[:4] for r in found if r[2] or r[3]],
       "models": sorted({r[4] for r in found if r[4]}), "byok_any": any(r[5] for r in found),
       "seconds": round(time.time() - t0)}
json.dump(out, open(os.path.join(TRAIN, "logs/cost_check_bulk.json"), "w"), indent=1)
print(json.dumps({k: (v if k != "not_found" else len(v)) for k, v in out.items()}, indent=1))
