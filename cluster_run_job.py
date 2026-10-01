#!/usr/bin/env python3
"""
rfcx-local AED Clustering worker wrapper (job_type_id 9).
Phase D-ANALYSIS, Session B (2026-06-09).

Upstream `rfcx/arbimon-clustering` `cluster.py` was an AWS-only container the
arbimon-legacy frontend posted DIRECTLY to EKS (k8sClient.jobs.post), bypassing
the jobqueue. It takes its inputs as CLI flags:

    python cluster.py -e <eps> -m <minPts> -s <maxClusterSize> -j <job_id> -a <aed_job_id>

To run it through the rfcx-local dispatcher (which invokes every worker as
`<cmd> <job_id>`), this thin wrapper:
  1. takes ONLY the job_id (uniform dispatcher contract),
  2. reads job_params_audio_event_clustering for this job_id to resolve the
     audio_event_detection_job_id + the DBSCAN params (Min. Points / Distance
     Threshold / Max. Cluster Size) the frontend stored,
  3. resets state='processing', progress=0 so a re-run is idempotent (cluster.py
     does progress = progress+1 increments 1..4; without a reset a re-run would
     overshoot progress_steps),
  4. rebuilds sys.argv into the flag form cluster.py expects and execs it
     verbatim (runpy with __name__='__main__'), so ALL of the ML logic
     (DBSCAN + PaCMAP + LDA/PCA projection + the two result JSONs) is the
     upstream code, unmodified except the S3_ENDPOINT patch.

The result-JSON S3 keys (audio_events/<env>/clustering/<job>/<job>_lda.json and
_aed_info.json) and the job_params_audio_event_clustering.aeds_clustered /
clusters_detected columns are exactly what arbimon-legacy clustering-jobs.js +
routes/.../clustering-jobs.js reconstruct, so no frontend-contract change is
needed beyond routing creation through the jobqueue.
"""
import traceback
import os
import sys
import json
import runpy
import datetime as dt

import sqlalchemy as sqal
from db import connect


def main(job_id):
    session, engine, metadata = connect()
    jobs = sqal.Table('jobs', metadata, autoload=True, autoload_with=engine)
    jpac = sqal.Table('job_params_audio_event_clustering', metadata,
                      autoload=True, autoload_with=engine)

    row = session.execute(
        sqal.select([jpac.c.audio_event_detection_job_id, jpac.c.parameters])
        .where(jpac.c.job_id == job_id)
    ).fetchall()
    if not row:
        session.execute(jobs.update().where(jobs.c.job_id == job_id).values(
            state='error', completed=-1,
            remarks='No job_params_audio_event_clustering row',
            last_update=dt.datetime.now()))
        session.commit()
        print(f"FAIL: no clustering params for job {job_id}")
        return 1

    aed_job_id = int(row[0][0])
    params = json.loads(row[0][1]) if row[0][1] else {}
    # frontend stores these exact keys (clustering-jobs.js requestNewClusteringJob)
    min_pts = int(params.get('Min. Points', 5))
    eps = float(params.get('Distance Threshold', 0.15))
    max_size = int(params.get('Max. Cluster Size', 100))

    # Idempotent (re-)start: reset progress so cluster.py's 4 increments land
    # exactly on progress_steps; clear any prior terminal flag. cluster.py
    # overwrites the two result JSONs by the same S3 keys, so a re-run is safe.
    session.execute(jobs.update().where(jobs.c.job_id == job_id).values(
        state='processing', progress=0, completed=0, last_update=dt.datetime.now()))
    session.commit()
    session.close()
    engine.dispose()

    print(f"clustering job_id={job_id} aed_job_id={aed_job_id} "
          f"eps={eps} min_pts={min_pts} max_size={max_size}")

    # Hand off to the upstream cluster.py exactly as its CLI expects.
    sys.argv = ['cluster.py',
                '-e', str(eps),
                '-m', str(min_pts),
                '-s', str(max_size),
                '-j', str(job_id),
                '-a', str(aed_job_id)]
    try:
        globs = runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            'cluster.py'),
                               run_name='__main__')
    except Exception as e:
        # 2026-09-23 (rfcx-local §383): an uncaught exception in cluster.py
        # used to leave the row in state='processing' forever -- the K8s Job
        # failed and was GC'd, and the dispatcher reaper never revives a
        # non-resumable type. Record a terminal error the user can read, then
        # still exit non-zero so the failed-Job alerts keep firing.
        # (SystemExit is not an Exception: cluster.py's empty-result path
        # exits 0 via sys.exit and is NOT caught here.)
        traceback.print_exc()
        mark_error(job_id, error_remarks(e))
        return 1
    # 2026-10-01 (rfcx-local FINDING-2026-10-01-aed-roi-review-surfaces-vs-prewarm):
    # warm this run's thumbnails -- the ONLY ROIs the AED review page can show.
    # Fire-and-forget, after the job is already 'completed'; never fails the job.
    # Default OFF (MEDIA_PREWARM_API_URL unset). cluster.py's empty-result path
    # exits via SystemExit before this point, so it is never reached there.
    try:
        import clustering_prewarm
        clustering_prewarm.run_after_cluster(job_id, aed_job_id, globs)
    except Exception as e:
        print(f"clustering-prewarm: not run ({e.__class__.__name__}: {e})")
    return 0


def error_remarks(exc):
    """Short, user-facing reason (shown on the Jobs page)."""
    msg = ' '.join(str(exc).split())[:300]
    return ('Clustering failed: %s%s. Please try again; if it fails again, '
            'try a different Distance Threshold or contact support.'
            % (type(exc).__name__, (': ' + msg) if msg else ''))


def mark_error(job_id, remarks):
    """Terminal error on a FRESH connection (cluster.py's own session may be
    mid-transaction). Only a row still 'processing' and not cancel-requested
    is touched, so a user cancel is never overwritten. Best effort: a DB
    failure here must not mask the original exception in the logs."""
    try:
        session, engine, metadata = connect()
        jobs = sqal.Table('jobs', metadata, autoload=True, autoload_with=engine)
        session.execute(jobs.update().where(sqal.and_(
            jobs.c.job_id == job_id,
            jobs.c.state == 'processing',
            jobs.c.cancel_requested == 0,
        )).values(state='error', completed=-1, remarks=remarks,
                  last_update=dt.datetime.now()))
        session.commit()
        session.close()
        engine.dispose()
        print(f"marked job {job_id} error: {remarks}")
    except Exception as db_e:
        print(f"WARNING: could not mark job {job_id} error: {db_e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: cluster_run_job.py <job_id>")
        sys.exit(2)
    sys.exit(main(int(sys.argv[1].strip("'"))))
