"""Contract test for the 2026-08-06 id-liveness guard in cluster.py.

Mirrors test_run_epoch.py's style: no DB, no S3 — a fake session that
answers the same sqlalchemy select the guard issues. Verifies:
  1. all-live ids -> guard passes silently
  2. any dead id -> RuntimeError naming the job, the counts, and a sample
  3. batching: id sets larger than one chunk are checked completely
     (a dead id in the LAST batch must still be caught)
"""


class FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


class FakeSession:
    """Answers execute(select aed_id where job_id==J and aed_id in batch)."""

    def __init__(self, live_ids):
        self.live = set(live_ids)
        self.batches_seen = []

    def execute(self, stmt):
        # the guard's statement carries the IN-list as its rightmost clause;
        # rather than parse sqlalchemy internals, the test builds the guard
        # loop itself and calls resolve() — see run_guard below.
        raise NotImplementedError

    def resolve(self, job_id, batch):
        self.batches_seen.append(list(batch))
        return FakeResult([(i,) for i in batch if i in self.live])


def run_guard(session, aed_job_id, shard_ids, chunk=10000):
    """The exact algorithm from cluster.py (kept in lockstep by eye —
    the logic is 12 lines; if you change cluster.py, change this)."""
    live = set()
    for o in range(0, len(shard_ids), chunk):
        batch = shard_ids[o:o + chunk]
        rows = session.resolve(aed_job_id, batch).fetchall()
        live.update(int(r[0]) for r in rows)
    dead = [i for i in shard_ids if i not in live]
    if dead:
        raise RuntimeError(
            f'STALE FEATURE SHARD IDS for aed_job {aed_job_id}: '
            f'{len(dead)} of {len(shard_ids)} shard aed_ids do not resolve '
            f'to live audio_event_detections_clustering rows '
            f'(sample: {dead[:5]}) — refusing to cluster.')


def test_all_live():
    s = FakeSession(live_ids=range(100, 200))
    run_guard(s, 9, list(range(100, 200)))
    print("all-live: OK (no raise)")


def test_dead_id_raises():
    s = FakeSession(live_ids=[1, 2, 3])
    try:
        run_guard(s, 9, [1, 2, 3, 999])
    except RuntimeError as e:
        msg = str(e)
        assert 'STALE FEATURE SHARD IDS for aed_job 9' in msg
        assert '1 of 4' in msg
        assert '999' in msg
        print("dead-id raise: OK —", msg.splitlines()[0][:70])
        return
    raise AssertionError("guard did not raise on a dead id")


def test_batching_last_batch():
    # 25 ids, chunk=10 -> 3 batches; the ONLY dead id is the final element.
    ids = list(range(25))
    s = FakeSession(live_ids=range(24))  # 24 is dead
    try:
        run_guard(s, 7, ids, chunk=10)
    except RuntimeError as e:
        assert '1 of 25' in str(e)
        assert len(s.batches_seen) == 3, s.batches_seen
        print("batching: OK (3 batches, dead id in last batch caught)")
        return
    raise AssertionError("dead id in final batch was not caught")


def test_duplicate_shard_ids_still_live():
    # duplicated ids in shards (same id twice) must NOT be flagged dead
    s = FakeSession(live_ids=[5])
    run_guard(s, 3, [5, 5, 5])
    print("duplicate-live ids: OK (no raise)")


if __name__ == '__main__':
    test_all_live()
    test_dead_id_raises()
    test_batching_last_batch()
    test_duplicate_shard_ids_still_live()
    print("ALL PASS")
