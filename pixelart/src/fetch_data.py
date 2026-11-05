"""Fetch the first N drawings of each Quick, Draw! category with HTTP Range requests.

Data: Google "Quick, Draw!" numpy bitmaps, CC BY 4.0,
https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/<category>.npy

Each file is a .npy of shape (n, 784) uint8 (~100 MB). We download only the first
1 KiB to parse the header (magic, version, header length, dict), then only the
bytes for the first N rows. Raw bytes go to data/raw/ and are treated as untrusted:
the header is parsed with ast.literal_eval (no pickle), dtype/shape are validated,
and the payload is only ever read with numpy.frombuffer as uint8.

usage: python -I src/fetch_data.py <pixelart_root> [N]
"""
import ast, json, os, struct, subprocess, sys, urllib.parse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # -I drops the script dir

from classes import CLASSES, slug

BASE = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"
ROW = 784


def curl_range(url, a, b, out):
    r = subprocess.run(["curl", "-sS", "--fail", "--proto", "=https", "-r", f"{a}-{b}",
                        "-o", out, "-w", "%{http_code}", url], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"curl failed for {url}: {r.stderr.strip()}")
    return r.stdout.strip()


def parse_header(buf):
    if buf[:6] != b"\x93NUMPY":
        raise ValueError("bad magic")
    major, minor = buf[6], buf[7]
    if major == 1:
        hlen = struct.unpack("<H", buf[8:10])[0]; off = 10
    elif major in (2, 3):
        hlen = struct.unpack("<I", buf[8:12])[0]; off = 12
    else:
        raise ValueError(f"unsupported npy version {major}.{minor}")
    if off + hlen > len(buf):
        raise ValueError("header longer than probe")
    d = ast.literal_eval(buf[off:off + hlen].decode("latin1"))
    if not isinstance(d, dict) or set(d) != {"descr", "fortran_order", "shape"}:
        raise ValueError("unexpected header dict")
    if d["descr"] != "|u1" or d["fortran_order"] is not False:
        raise ValueError(f"unexpected dtype/order {d}")
    shp = d["shape"]
    if not (isinstance(shp, tuple) and len(shp) == 2 and shp[1] == ROW and shp[0] > 0):
        raise ValueError(f"unexpected shape {shp}")
    return dict(version=f"{major}.{minor}", data_offset=off + hlen, shape=list(shp))


def main():
    root = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
    raw = os.path.join(root, "data", "raw")
    os.makedirs(raw, exist_ok=True)
    manifest = {}
    for c in CLASSES:
        url = BASE + urllib.parse.quote(c + ".npy")
        probe = os.path.join(raw, slug(c) + ".probe")
        code = curl_range(url, 0, 1023, probe)
        if code != "206":
            raise RuntimeError(f"{c}: server did not honour Range (HTTP {code})")
        hdr = parse_header(open(probe, "rb").read())
        take = min(n, hdr["shape"][0])
        a = hdr["data_offset"]; b = a + take * ROW - 1
        out = os.path.join(raw, slug(c) + ".u8")
        if not (os.path.exists(out) and os.path.getsize(out) == take * ROW):
            code = curl_range(url, a, b, out)
            if code != "206":
                raise RuntimeError(f"{c}: HTTP {code} on data range")
        got = os.path.getsize(out)
        if got != take * ROW:
            raise RuntimeError(f"{c}: got {got} bytes, wanted {take * ROW}")
        manifest[c] = dict(url=url, header=hdr, rows=take, bytes=got, file=os.path.basename(out))
        print(f"{c:12s} total={hdr['shape'][0]:7d} offset={hdr['data_offset']} took={take}", flush=True)
    json.dump(manifest, open(os.path.join(raw, "manifest.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
