#!/usr/bin/env python3
"""
rfcx-local clustering QUEUE consumer (job_type_id 9) — 2026-10-01.
Design: rfcx-local runbooks/DESIGN-2026-10-01-heavy-job-queue-clustering.md §6.

The rfcx-local jobqueue-dispatcher (CLUSTERING_QUEUE_MODE=all) publishes ONE message
per claimed clustering job onto a lane:

    clustering.work.<0..N-1>        fair lanes (project_id -> lane)
    clustering.work.priority.<i>    (dark until an entitlement exists)
    clustering.work.background.<i>  (dispatcher BACKGROUND_REMARK_PREFIXES cohort)

body = {"v": 1, "job_id": <int>, "lane": "...", "published_at": "..."}

This process is a KEDA-scaled Deployment (min 0, max = heavy-node count) that runs
ONE job at a time:

  1. basic_get round-robin over the lanes (priority -> fair... -> background), so a
     burst on one fair lane cannot starve the others.
  2. CLAIM the row in PG:  initializing -> processing  (also requires cancel_requested=0).
     Lost claim (already processing/terminal, cancelled, or reverted) => ack + drop:
     this is what makes a dispatcher/reaper re-publish harmless.
  3. ACK, then CLOSE the AMQP connection BEFORE running. Ack-after-run is unsafe here:
     the broker consumer_timeout is 30 min and pika's BlockingConnection cannot
     heartbeat while we compute, while 17/189 clustering jobs (180 d) ran > 25 min.
     Either would close the channel and REDELIVER = a duplicate concurrent run.
  4. Run `cluster_run_job.py <job_id>` as a SUBPROCESS — byte-identical to the
     per-job k8s Job command — so every byte of job memory returns to the node
     when it exits, and all crash handling (terminal error rows) is unchanged.
  5. Reconnect and loop. SIGTERM (KEDA scale-down / rollout): stop taking new
     messages and wait for the running job (terminationGracePeriodSeconds covers
     the 4 h active deadline).

A crash between ack and completion leaves the row 'processing', exactly as a
crashed per-job k8s Job does today; the dispatcher reaper surfaces it for review.
"""
import datetime as dt
import json
import os
import signal
import subprocess
import sys
import time

import pika
import sqlalchemy as sqal

from db import connect

QUEUE = os.environ.get("CLUSTERING_QUEUE", "clustering.work")
LANE_COUNT = int(os.environ.get("CLUSTERING_LANE_COUNT", "3"))
PRIORITY_COUNT = int(os.environ.get("CLUSTERING_PRIORITY_COUNT", "1"))
BACKGROUND_COUNT = int(os.environ.get("CLUSTERING_BACKGROUND_COUNT", "1"))
DELIVERY_LIMIT = int(os.environ.get("CLUSTERING_DELIVERY_LIMIT", "3"))
IDLE_SLEEP = float(os.environ.get("CLUSTERING_IDLE_SLEEP", "5"))
RUN_CMD = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "cluster_run_job.py")]

_stop = False


def _on_term(signum, frame):
    global _stop
    _stop = True
    print(f"signal {signum}: finishing current job (if any), taking no new work",
          flush=True)


def lanes():
    """Service order for one sweep: priority, fair lanes, background. Every lane is
    visited once per sweep (round-robin), so fair lanes share the worker evenly."""
    return ([f"{QUEUE}.priority.{i}" for i in range(PRIORITY_COUNT)]
            + [f"{QUEUE}.{i}" for i in range(LANE_COUNT)]
            + [f"{QUEUE}.background.{i}" for i in range(BACKGROUND_COUNT)])


def declare(ch):
    """Idempotent; identical arguments to the dispatcher's declare."""
    for q in lanes():
        dlq = q + ".dlq"
        ch.queue_declare(queue=dlq, durable=True, arguments={"x-queue-type": "quorum"})
        ch.queue_declare(queue=q, durable=True, arguments={
            "x-queue-type": "quorum",
            "x-delivery-limit": DELIVERY_LIMIT,
            "x-dead-letter-exchange": "",
            "x-dead-letter-routing-key": dlq})


def amqp():
    params = pika.URLParameters(os.environ["AMQP_URL"])
    params.heartbeat = 60
    params.blocked_connection_timeout = 60
    for attempt in range(8):
        try:
            return pika.BlockingConnection(params)
        except Exception as e:
            wait = min(60, 2 ** attempt)
            print(f"amqp connect failed ({e}); retry in {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError("amqp unreachable after retries")


def claim(job_id):
    """initializing -> processing, iff not cancelled. True = this worker owns the run."""
    session, engine, metadata = connect()
    try:
        jobs = sqal.Table("jobs", metadata, autoload=True, autoload_with=engine)
        res = session.execute(jobs.update().where(sqal.and_(
            jobs.c.job_id == job_id,
            jobs.c.job_type_id == 9,
            jobs.c.state == "initializing",
            sqal.func.coalesce(jobs.c.cancel_requested, 0) == 0,
        )).values(state="processing", last_update=dt.datetime.now()))
        session.commit()
        return res.rowcount == 1
    finally:
        session.close()
        engine.dispose()


def take_one():
    """One sweep over the lanes. Returns (job_id, lane) for a CLAIMED job, or None.
    Messages that cannot be claimed are acked + dropped here."""
    conn = amqp()
    try:
        ch = conn.channel()
        declare(ch)
        for lane in lanes():
            method, props, body = ch.basic_get(lane, auto_ack=False)
            if method is None:
                continue
            try:
                msg = json.loads(body)
                job_id = int(msg["job_id"])
            except Exception as e:
                # Malformed: reject without requeue -> DLQ via delivery limit / DLX.
                print(f"{lane}: malformed message ({e}); rejecting to DLQ", flush=True)
                ch.basic_reject(method.delivery_tag, requeue=False)
                continue
            try:
                owned = claim(job_id)
            except Exception as e:
                # DB trouble: put it back (requeue) and back off; the delivery limit
                # bounds how often this can repeat before the message is DLQ'd.
                print(f"job {job_id}: claim failed ({e}); requeue", flush=True)
                ch.basic_nack(method.delivery_tag, requeue=True)
                return None
            ch.basic_ack(method.delivery_tag)
            if owned:
                print(f"job {job_id}: claimed from {lane}", flush=True)
                return job_id, lane
            print(f"job {job_id}: not claimable (already running/finished/cancelled); "
                  f"dropped duplicate from {lane}", flush=True)
        return None
    finally:
        try:
            conn.close()
        except Exception:
            pass


def run(job_id):
    t0 = time.time()
    p = subprocess.Popen(RUN_CMD + [str(job_id)])
    while True:
        try:
            rc = p.wait()
            break
        except KeyboardInterrupt:
            continue
    print(f"job {job_id}: cluster_run_job exited rc={rc} after {time.time() - t0:.0f}s",
          flush=True)
    return rc


def main():
    signal.signal(signal.SIGTERM, _on_term)
    signal.signal(signal.SIGINT, _on_term)
    print(f"clustering consumer up: lanes={lanes()}", flush=True)
    while not _stop:
        try:
            got = take_one()
        except Exception as e:
            print(f"sweep failed ({e.__class__.__name__}: {e}); retry in 10s", flush=True)
            time.sleep(10)
            continue
        if got is None:
            time.sleep(IDLE_SLEEP)
            continue
        run(got[0])
    print("clustering consumer: stopped", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())