#!/usr/bin/env python3
"""Pre-warm the thumbnails of a COMPLETED clustering run (rfcx-local 2026-10-01).

WHY HERE, NOT AT AED COMPLETION: the AED review page (SPA
`aed-clustering/detail/:clusteringJobId`; legacy clustering grid) can show ONLY
the aed_ids a clustering run kept -- cluster.py drops DBSCAN noise and caps each
cluster at `Max. Cluster Size`. Measured: <= 10-16 % of AED ROIs are ever
displayable, so warming at AED completion spent >= 84 % of its renders on
images no page can request. This module warms exactly that run's aed_ids, in
the page's order (the `_lda.json` order the page reads), 400x400 thumbnails
only. rfcx-local runbooks/FINDING-2026-10-01-aed-roi-review-surfaces-vs-prewarm.md.

CACHE-KEY PARITY: media-api caches by the EXACT filename. The fields built here
are byte-identical to rfcx/arbimon-jobs-analysis roi_prewarm_publish.
build_message() (rounded ms + half-up Hz, matching the arbimon-legacy reader);
test_clustering_prewarm.py pins real rows to the DEPLOYED reader's filename
(prewarm-reader-golden.json, same fixture as that repo).

RULE 1 (same as every producer): A PRE-WARM FAILURE NEVER FAILS THE CLUSTERING
JOB. Every entry point swallows everything and returns 0. DEFAULT OFF: with
MEDIA_PREWARM_API_URL unset nothing is sent.
"""
from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.request
from datetime import datetime as _datetime, timedelta as _timedelta, timezone as _timezone

ORIGIN = os.environ.get("CLUSTERING_PREWARM_ORIGIN", "aed")   # service ORIGINS today: pm,aed,visualizer-open,backfill,admin
SERVICE_MAX_REQUESTS = 250
MAX_ITEMS = 200            # == media-prewarm per-message BATCH (no runt messages)
_EPOCH = _datetime(1970, 1, 1)

def api_url():
    v = os.environ.get("MEDIA_PREWARM_API_URL", "").strip()
    return v.rstrip("/") or None

def _round_half_up(x: float) -> int:
    f = math.floor(x)
    return int(f + 1) if (x - f) >= 0.5 else int(f)

def _as_stored_real(x: float) -> float:
    try:
        import numpy as _np
        return float(str(_np.float32(x)))
    except Exception:
        return float(x)

def _utc_ms(d) -> float:
    if d.tzinfo is None:
        d = d.replace(tzinfo=_timezone.utc)
    return d.timestamp() * 1000.0

def _stamp(ms: int) -> str:
    d = _EPOCH + _timedelta(milliseconds=int(ms))
    return (f"{d.year:04d}{d.month:02d}{d.day:02d}T"
            f"{d.hour:02d}{d.minute:02d}{d.second:02d}{d.microsecond // 1000:03d}")

def stream_id(rec_uri, site_external_id):
    """arbimon-legacy mediaStreamId(): uri segment first, site external_id fallback."""
    if isinstance(rec_uri, str) and not rec_uri.startswith("project_"):
        parts = rec_uri.split("/")
        if len(parts) == 5 and len(parts[0]) == 4 and parts[3] and parts[3] != "undefined":
            return parts[3]
    if site_external_id and site_external_id != "undefined":
        return site_external_id
    return None

def roi_item(time_min, time_max, freq_min, freq_max, datetime_utc, sample_rate, sid):
    """One `roi` item for media-prewarm; None if not renderable."""
    if sid is None or datetime_utc is None:
        return None
    t1, t2 = _as_stored_real(float(time_min)), _as_stored_real(float(time_max))
    lo, hi = min(t1, t2), max(t1, t2)
    base = _utc_ms(datetime_utc)
    f1, f2 = _as_stored_real(float(freq_min)), _as_stored_real(float(freq_max))
    fmin, fmax = max(0.0, min(f1, f2)), max(f1, f2)
    if sample_rate and fmax > float(sample_rate) / 2.0:
        fmax = float(sample_rate) / 2.0
    return {
        "external_id": sid,
        "start": _stamp(_round_half_up(base + lo * 1000.0)),
        "end": _stamp(_round_half_up(base + hi * 1000.0)),
        "freq_min": float(_round_half_up(fmin)),
        "freq_max": float(_round_half_up(fmax)),
        "sample_rate": sample_rate,
    }

_ROWS_SQL = (
    "SELECT A.aed_id, A.time_min, A.time_max, A.frequency_min, A.frequency_max, "
    "R.datetime_utc, R.sample_rate, R.uri, S.external_id "
    "FROM audio_event_detections_clustering A "
    "JOIN recordings R ON R.recording_id = A.recording_id "
    "JOIN sites S ON S.site_id = R.site_id "
    "WHERE A.job_id = :aed_job_id")

def _post(base, body, timeout):
    req = urllib.request.Request(base + "/internal/prewarm", method="POST",
                                 data=json.dumps(body, separators=(",", ":")).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return int(json.loads(resp.read() or b"{}").get("accepted", 0))

def prewarm_clustering(job_id, aed_job_id, ordered_aed_ids, *, project_id=0, user_id=0,
                       fetch_rows=None, log=print) -> int:
    """POST the run's aed_ids (page order) as `roi` items. Returns accepted count.

    `fetch_rows(aed_job_id) -> iterable of the _ROWS_SQL columns`; injected so the
    test needs no DB. Never raises.
    """
    try:
        base = api_url()
        if base is None or not ordered_aed_ids:
            return 0
        rows = {int(r[0]): r for r in fetch_rows(aed_job_id)}
        items, skipped = [], 0
        for aid in ordered_aed_ids:
            r = rows.get(int(aid))
            if r is None:
                skipped += 1
                continue
            try:
                it = roi_item(r[1], r[2], r[3], r[4], r[5], r[6], stream_id(r[7], r[8]))
            except Exception:
                it = None
            if it is None:
                skipped += 1
                continue
            items.append({"roi": it})
        if not items:
            log(f"clustering-prewarm: job {job_id}: 0 renderable of {len(ordered_aed_ids)}")
            return 0
        try:
            timeout = float(os.environ.get("MEDIA_PREWARM_API_TIMEOUT_S", "5"))
        except Exception:
            timeout = 5.0
        total = 0
        for i in range(0, len(items), MAX_ITEMS):
            body = {"origin": ORIGIN, "project": int(project_id or 0), "user": int(user_id or 0),
                    "ref": f"clustering:{int(job_id)}", "requests": items[i:i + MAX_ITEMS]}
            try:
                total += _post(base, body, timeout)
            except urllib.error.HTTPError as e:
                log(f"clustering-prewarm: job {job_id} slice {i // MAX_ITEMS}: refused {e.code}")
            except Exception as e:
                log(f"clustering-prewarm: job {job_id} slice {i // MAX_ITEMS}: {e.__class__.__name__}: {e}")
        log(f"clustering-prewarm: job {job_id} (aed {aed_job_id}): accepted {total} of "
            f"{len(items)} roi(s), skipped {skipped}")
        return total
    except Exception as e:
        try:
            log(f"clustering-prewarm: job {job_id} failed ({e.__class__.__name__}: {e})")
        except Exception:
            pass
        return 0

def run_after_cluster(job_id, aed_job_id, globs, log=print) -> int:
    """Called by cluster_run_job.py after cluster.py returned normally.

    Uses cluster.py's in-memory `ids` (the aed_ids written to <job>_lda.json, in
    that order = the review page's order). Only fires if the job row is
    state='completed' (the empty-result path exits 0 with no ids). Never raises.
    """
    try:
        if api_url() is None:
            return 0
        ids = globs.get("ids") if isinstance(globs, dict) else None
        if ids is None:
            return 0
        ordered = [int(i) for i in ids]
        import sqlalchemy as sqal
        from db import connect
        session, engine, _md = connect()
        try:
            st = session.execute(sqal.text(
                "SELECT j.state, j.project_id, j.user_id FROM jobs j WHERE j.job_id = :j"),
                {"j": int(job_id)}).fetchone()
            if not st or st[0] != "completed":
                log(f"clustering-prewarm: job {job_id} not completed ({st and st[0]}); skipping")
                return 0
            rows = session.execute(sqal.text(_ROWS_SQL), {"aed_job_id": int(aed_job_id)}).fetchall()
        finally:
            session.close()
            engine.dispose()
        return prewarm_clustering(job_id, aed_job_id, ordered, project_id=st[1], user_id=st[2],
                                  fetch_rows=lambda _a: rows, log=log)
    except Exception as e:
        try:
            log(f"clustering-prewarm: job {job_id} skipped ({e.__class__.__name__}: {e})")
        except Exception:
            pass
        return 0