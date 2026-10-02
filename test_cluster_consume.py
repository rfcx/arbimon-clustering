"""Contract tests for cluster_consume.py (rfcx-local DESIGN-2026-10-01-heavy-job-queue-clustering.md).
No broker, no DB: pika and the claim are faked; the run step is captured.
  1. lane order = priority, fair 0..2, background (round-robin sweep).
  2. a claimable message is ACKED BEFORE the run starts, and the AMQP connection is
     CLOSED before the run (ack-then-run; no channel held across a long job).
  3. an unclaimable message (duplicate / cancelled / finished) is acked and NOT run.
  4. a claim DB failure -> nack(requeue=True), not ack, not run.
  5. a malformed body -> reject(requeue=False) (DLQ), sweep continues to the next lane.
  6. SIGTERM flag stops the loop after the current job.
  RED CONTROL: --red runs the same checks against a variant whose claim always succeeds
  (no dedupe) and which acks AFTER run; checks 2 and 3 must FAIL.
Run: python3 test_cluster_consume.py [--red]      exit 0 = all pass (or, with --red, all reds red).
"""
import importlib.util, json, os, sys, types
HERE = os.path.dirname(os.path.abspath(__file__))
RED = "--red" in sys.argv

# ---- fakes installed BEFORE import ---------------------------------------------
EVENTS = []
class FakeCh:
    def __init__(self, conn): self.conn = conn
    def queue_declare(self, queue, durable, arguments=None): pass
    def basic_get(self, queue, auto_ack=False):
        msgs = QUEUES.get(queue) or []
        if not msgs:
            return None, None, None
        body = msgs.pop(0)
        tag = len(EVENTS) + 1
        EVENTS.append(("get", queue, tag))
        return types.SimpleNamespace(delivery_tag=tag), None, body
    def basic_ack(self, tag): EVENTS.append(("ack", tag))
    def basic_nack(self, tag, requeue): EVENTS.append(("nack", tag, requeue))
    def basic_reject(self, tag, requeue): EVENTS.append(("reject", tag, requeue))
class FakeConn:
    def __init__(self, params): EVENTS.append(("open",))
    def channel(self): return FakeCh(self)
    def close(self): EVENTS.append(("close",))
fake_pika = types.SimpleNamespace(URLParameters=lambda url: types.SimpleNamespace(),
                                  BlockingConnection=FakeConn)
sys.modules["pika"] = fake_pika
fake_db = types.ModuleType("db"); fake_db.connect = lambda: None
sys.modules["db"] = fake_db
sys.modules.setdefault("sqlalchemy", types.ModuleType("sqlalchemy"))
os.environ["AMQP_URL"] = "amqp://unit-test"

spec = importlib.util.spec_from_file_location("cc", os.path.join(HERE, "cluster_consume.py"))
cc = importlib.util.module_from_spec(spec); spec.loader.exec_module(cc)

CLAIMABLE = set()
CLAIM_RAISES = set()
def fake_claim(job_id):
    if job_id in CLAIM_RAISES:
        raise RuntimeError("db down (test)")
    return True if RED else (job_id in CLAIMABLE)
cc.claim = fake_claim
RUNS = []
def fake_run(job_id):
    EVENTS.append(("run", job_id)); RUNS.append(job_id); return 0
cc.run = fake_run

if RED:
    # variant: ack AFTER the run (the unsafe ordering), holding the connection open.
    orig_take = cc.take_one
    def take_one_red():
        conn = cc.amqp(); ch = conn.channel()
        for lane in cc.lanes():
            m, p, b = ch.basic_get(lane)
            if m is None: continue
            j = int(json.loads(b)["job_id"])
            cc.run(j); ch.basic_ack(m.delivery_tag); conn.close()
            return None
        conn.close(); return None
    cc.take_one = take_one_red

FAIL = []
def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else "  -- " + detail))
    if not cond: FAIL.append(name)
def msg(j): return json.dumps({"v": 1, "job_id": j}).encode()
def reset(q):
    global QUEUES
    QUEUES = q; EVENTS.clear(); RUNS.clear()

def sweep_and_run():
    got = cc.take_one()
    if got: cc.run(got[0])

# 1
check("1 lane order", cc.lanes() == ["clustering.work.priority.0", "clustering.work.0",
      "clustering.work.1", "clustering.work.2", "clustering.work.background.0"], str(cc.lanes()))
# 2
reset({"clustering.work.1": [msg(100)]}); CLAIMABLE.clear(); CLAIMABLE.add(100)
sweep_and_run()
ev = [e[0] for e in EVENTS]
ok2 = ("ack" in ev and "run" in ev and ev.index("ack") < ev.index("run")
       and "close" in ev and ev.index("close") < ev.index("run"))
check("2 ack + close BEFORE run", ok2, str(EVENTS))
# 3
reset({"clustering.work.0": [msg(200)]}); CLAIMABLE.clear()
sweep_and_run()
check("3 unclaimable -> acked, NOT run", not RUNS and any(e[0] == "ack" for e in EVENTS), str(EVENTS))
# 4
reset({"clustering.work.0": [msg(300)]}); CLAIM_RAISES.add(300)
got = cc.take_one() if not RED else None
CLAIM_RAISES.clear()
if not RED:
    check("4 claim DB failure -> nack(requeue), no ack, no run",
          got is None and any(e[0] == "nack" and e[2] is True for e in EVENTS)
          and not any(e[0] == "ack" for e in EVENTS) and not RUNS, str(EVENTS))
# 5
if not RED:
    reset({"clustering.work.0": [b"not json"], "clustering.work.1": [msg(400)]}); CLAIMABLE.clear(); CLAIMABLE.add(400)
    got = cc.take_one()
    check("5 malformed -> reject(requeue=False), next lane served",
          any(e[0] == "reject" and e[2] is False for e in EVENTS) and got and got[0] == 400, str(EVENTS))
# 6
if not RED:
    reset({}); cc._stop = False
    calls = []
    def take_then_stop():
        calls.append(1); cc._stop = True; return None
    cc.take_one = take_then_stop; cc.IDLE_SLEEP = 0
    rc = cc.main()
    check("6 stop flag ends the loop", rc == 0 and len(calls) == 1, str(calls))

print("\n%d failure(s)%s" % (len(FAIL), " [RED CONTROL]" if RED else ""))
if RED:
    # In red mode the unsafe variant MUST fail checks 2 and 3.
    sys.exit(0 if {"2 ack + close BEFORE run", "3 unclaimable -> acked, NOT run"} <= set(FAIL) else 1)
sys.exit(1 if FAIL else 0)