#!/usr/bin/env python3
"""or_free.py: minimal OpenRouter client for FREE models only (toner1-train).

The key is read inside this process only: from the environment variable OPENROUTER_API_KEY, or else by
regex sk-or-v1-[0-9a-f]{20,} from a key file ($OPENROUTER_KEY_FILE, default training/.env, which git ignores).
It is never printed, logged, written or passed on a command line; every error string is scrubbed.

Hard $0 guards:
  * assert_free(model) refuses any model whose /models pricing (prompt, completion, request, image,
    web_search, internal_reasoning) is not all zero, or whose id does not end in ":free".
  * chat() raises CostError if a response reports usage.cost > 0.

CLI:  python -I or_free.py usage      -> prints usage numbers from GET /key (no key material)
"""
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

KEY_FILE = os.environ.get("OPENROUTER_KEY_FILE") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
KEY_RE = re.compile(r"sk-or-v1-[0-9a-f]{20,}")
API = "https://openrouter.ai/api/v1"
TITLE = "toner1-train printer stories"

_KEY = None


class CostError(RuntimeError):
    pass


def scrub(s):
    s = str(s)
    if _KEY:
        s = s.replace(_KEY, "[REDACTED]")
    return re.sub(r"sk-or-[A-Za-z0-9-]{8,}", "[REDACTED]", s)


def load_key():
    global _KEY
    if _KEY:
        return
    env = os.environ.get("OPENROUTER_API_KEY", "")
    m = KEY_RE.search(env)
    if not m:
        try:
            with open(KEY_FILE, encoding="utf-8", errors="ignore") as fh:
                m = KEY_RE.search(fh.read())
        except OSError as e:
            raise SystemExit(f"or_free: no OPENROUTER_API_KEY and cannot read the key file ({e.__class__.__name__})")
    if not m:
        raise SystemExit("or_free: no OpenRouter key (sk-or-v1-...) in OPENROUTER_API_KEY or the key file")
    _KEY = m.group(0)


def http(method, path, body=None, timeout=60, auth=True):
    req = urllib.request.Request(API + path, method=method)
    if auth:
        load_key()
        req.add_header("Authorization", "Bearer " + _KEY)
    req.add_header("X-Title", TITLE)
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, data=data, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode()), dict(r.headers)
    except urllib.error.HTTPError as e:
        try:
            detail = e.read().decode()[:1500]
        except Exception:
            detail = ""
        return e.code, {"error": scrub(detail)}, dict(e.headers or {})
    except Exception as e:  # network errors, timeouts, bad JSON
        return 0, {"error": scrub(f"{e.__class__.__name__}: {e}")}, {}


SAFE_KEY_FIELDS = ("usage", "usage_daily", "usage_weekly", "usage_monthly", "limit", "limit_remaining",
                   "limit_reset", "is_free_tier", "free_model_daily_requests", "rate_limit")


def key_info():
    """Selected numeric fields of GET /key (never the label, which can echo part of the key)."""
    st, d, _ = http("GET", "/key", timeout=30)
    if st != 200 or "data" not in d:
        raise SystemExit(f"or_free: GET /key failed (HTTP {st}): {scrub(d.get('error', ''))[:300]}")
    out = {k: d["data"].get(k) for k in SAFE_KEY_FIELDS if k in d["data"]}
    return json.loads(scrub(json.dumps(out)))


def model_record(model):
    st, d, _ = http("GET", "/models", timeout=60, auth=False)
    if st != 200:
        raise SystemExit(f"or_free: GET /models failed (HTTP {st})")
    for m in d.get("data", []):
        if m.get("id") == model:
            return m
    return None


def assert_free(model):
    if not model.endswith(":free"):
        raise SystemExit(f"or_free: refusing {model}: id does not end in ':free'")
    m = model_record(model)
    if m is None:
        raise SystemExit(f"or_free: {model} is not listed on /models")
    p = m.get("pricing") or {}
    nonzero = {k: v for k, v in p.items() if k != "discount" and float(v or 0) != 0}
    if nonzero or float(p.get("prompt", 1)) != 0 or float(p.get("completion", 1)) != 0:
        raise SystemExit(f"or_free: refusing {model}: non-zero pricing {nonzero or p}")
    return m


def chat(model, messages, max_tokens=4000, temperature=1.0, extra=None, timeout=300):
    """One chat completion. Returns (status, json, headers). Raises CostError on any non-zero cost."""
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature,
            "usage": {"include": True}}
    if extra:
        body.update(extra)
    st, d, h = http("POST", "/chat/completions", body, timeout=timeout)
    if st == 200:
        cost = (d.get("usage") or {}).get("cost")
        if cost not in (None, 0, 0.0) and float(cost) != 0.0:
            raise CostError(f"non-zero usage.cost {cost} reported for {model}")
    return st, d, h


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "usage":
        info = key_info()
        info["_measured_at"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
        print(json.dumps(info, indent=1))
    else:
        print(__doc__)
