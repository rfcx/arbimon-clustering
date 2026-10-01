#!/usr/bin/env python3
"""Tests for clustering_prewarm.py (no DB, no S3; a local HTTP stub plays media-prewarm).

  1. KEY PARITY: for 110 REAL prod PM/AED rows, the item this module builds, run
     through the media-prewarm-worker filename grammar, equals the filename the
     DEPLOYED arbimon-legacy reader requests (prewarm-reader-golden.json; same
     fixture as rfcx/arbimon-jobs-analysis + rfcx/arbimon-jobs-internal).
  2. PAGE ORDER + SCOPE: POSTs exactly the clustering's aed_ids, in the given
     (lda.json) order, nothing else from the AED job; ref=clustering:<id>.
  3. SPLIT: <= 200 items per POST.
  4. DEFAULT OFF: no MEDIA_PREWARM_API_URL -> 0, no request.
  5. RULE 1: a dead endpoint / a raising row fetcher -> returns 0, never raises.
  6. run_after_cluster with no `ids` (empty-result path) -> 0.
Exit 1 on any failure.
"""
import datetime as dt
import http.server
import json
import math
import os
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.pop("MEDIA_PREWARM_API_URL", None)
import clustering_prewarm as CP  # noqa: E402

FAILS = []
def check(name, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + name + ("" if ok else f"  [{detail}]"))
    if not ok:
        FAILS.append(name)

# ---- 1. key parity on real rows -------------------------------------------------
def consumer_filename(it):
    lo, hi = min(it["freq_min"], it["freq_max"]), max(it["freq_min"], it["freq_max"])
    sr = it.get("sample_rate")
    if sr and hi > sr / 2.0:
        hi = sr / 2.0
    lo = max(0.0, lo)
    f = lambda x: int(math.floor(x) + 1) if (x - math.floor(x)) >= 0.5 else int(math.floor(x))  # consumer half-up (jobs-internal #12)
    return (f"{it['external_id']}_t{it['start']}Z.{it['end']}Z_r{f(lo)}.{f(hi)}"
            f"_g1_fspec_mtrue_d400.400_wdolph_z120.png")

G = json.load(open(os.path.join(HERE, "prewarm-reader-golden.json")))
bad = []
for c in G["cases"]:
    d = dt.datetime.strptime(c["datetime_utc"], "%Y-%m-%dT%H:%M:%S.%fZ")
    it = CP.roi_item(c["det"]["x1"], c["det"]["x2"], c["det"]["y1"], c["det"]["y2"], d,
                     c["sample_rate"], CP.stream_id(c["rec_uri"], c["site_external_id"]))
    if it is None or consumer_filename(it) != c["reader_filename_400x400"]:
        bad.append((c["id"], it and consumer_filename(it), c["reader_filename_400x400"]))
check(f"key parity: {len(G['cases'])} real rows == deployed reader filename", not bad, bad[:2])
check("stream_id: uri-first", CP.stream_id("2024/04/11/aI7JpY9VIRvJ/x.flac", "other") == "aI7JpY9VIRvJ")
check("stream_id: project_ uri falls back", CP.stream_id("project_1/site_2/x.wav", "ext") == "ext")
check("stream_id: nothing -> None", CP.stream_id(None, None) is None)

# ---- stub media-prewarm -----------------------------------------------------------
BODIES = []
class H(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        b = json.loads(self.rfile.read(n))
        BODIES.append(b)
        out = json.dumps({"accepted": len(b["requests"])}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out))); self.end_headers(); self.wfile.write(out)
    def log_message(self, *a):
        pass
srv = http.server.HTTPServer(("127.0.0.1", 0), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()
URL = f"http://127.0.0.1:{srv.server_address[1]}"

base = dt.datetime(2024, 4, 11, 5, 0, 0)
ROWS = [(1000 + i, 0.010666667 + i, 1.5 + i, 100.0, 2000.5, base, 48000,
         "2024/04/11/aI7JpY9VIRvJ/x.flac", "aI7JpY9VIRvJ") for i in range(450)]
order = [1000 + i for i in range(449, -1, -3)]   # 150 ids, reverse-strided: NOT row order

# ---- 4. default off -----------------------------------------------------------
n = CP.prewarm_clustering(7, 8, order, fetch_rows=lambda a: ROWS, log=lambda m: None)
check("default OFF: no URL -> 0", n == 0 and not BODIES)

os.environ["MEDIA_PREWARM_API_URL"] = URL
# ---- 2/3. scope, order, split ------------------------------------------------------
n = CP.prewarm_clustering(170817, 170814, order, project_id=9, user_id=5,
                          fetch_rows=lambda a: ROWS, log=lambda m: None)
sent = [it["roi"] for b in BODIES for it in b["requests"]]
starts = [s["start"] for s in sent]
want_starts = [CP.roi_item(r[1], r[2], r[3], r[4], r[5], r[6], "aI7JpY9VIRvJ")["start"]
               for aid in order for r in ROWS if r[0] == aid]
check("accepted == clustered ids (150)", n == 150, n)
check("only the run's aed_ids, in lda order", starts == want_starts)
check("<= 200 items per POST", all(len(b["requests"]) <= 200 for b in BODIES), [len(b["requests"]) for b in BODIES])
check("ref = clustering:<job>", all(b["ref"] == "clustering:170817" for b in BODIES))
check("origin allowed by the service", all(b["origin"] in ("pm", "aed", "visualizer-open", "backfill", "admin") for b in BODIES))
check("project/user carried", all(b["project"] == 9 and b["user"] == 5 for b in BODIES))
check("2000.5 Hz rounds half-UP", sent[0]["freq_max"] == 2001.0, sent[0]["freq_max"])
check("1.5 s start (+ .010666667 base) rounds to ms", sent[-1]["start"].endswith("011"), sent[-1]["start"])

# ---- 5. rule 1 ---------------------------------------------------------------------
os.environ["MEDIA_PREWARM_API_URL"] = "http://127.0.0.1:9"
try:
    n = CP.prewarm_clustering(1, 2, order[:3], fetch_rows=lambda a: ROWS, log=lambda m: None)
    check("dead endpoint -> 0, no raise", n == 0)
except Exception as e:
    check("dead endpoint -> 0, no raise", False, e)
def boom(_a):
    raise RuntimeError("db down")
try:
    check("raising fetcher -> 0, no raise", CP.prewarm_clustering(1, 2, order, fetch_rows=boom, log=lambda m: None) == 0)
except Exception as e:
    check("raising fetcher -> 0, no raise", False, e)
# ---- 6. empty-result path ------------------------------------------------------------
check("run_after_cluster without ids -> 0", CP.run_after_cluster(1, 2, {}, log=lambda m: None) == 0)

srv.shutdown()
print(f"RESULT: {'FAIL' if FAILS else 'PASS'} ({len(FAILS)} failure(s))")
sys.exit(1 if FAILS else 0)