"""Contract tests for the 2026-09-23 fixes (rfcx-local OPEN-ITEMS §383).

Runs WITHOUT a DB or S3: the functions under test are extracted from the
real source files by AST (cluster.py connects to the DB at import time, so it
cannot simply be imported). Requires the worker's numpy/sklearn -- run it in
the aed-clustering image:  python3 test_projection_and_terminal_states.py

  1. project_2d returns 2 columns when LDA collapses to 1 (the job-170584
     shape: many clusters on rank-deficient standardized input) -- the old
     code raised IndexError at mp[:,1].
  2. CONTROL: well-separated full-rank input still goes through LDA (2 cols).
  3. project_2d with 1-2 clusters uses PCA (unchanged upstream behaviour).
  4. project_2d on a single point / single feature still returns 2 columns.
  5. empty_result_remarks names the point count and threshold.
  6. error_remarks is short, single-line, and names the exception type.
Any failure exits 1; a missing dependency exits 2 (never a silent skip).
"""
import ast, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
try:
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.preprocessing import StandardScaler
except ImportError as e:
    print(f"CANNOT RUN (missing dependency: {e}) -- run in the aed-clustering image")
    sys.exit(2)


def load_funcs(path, names, extra_globals):
    tree = ast.parse(open(path).read())
    found = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    missing = set(names) - {n.name for n in found}
    if missing:
        print(f"FAIL: {path} lacks {sorted(missing)}")
        sys.exit(1)
    g = dict(extra_globals)
    exec(compile(ast.Module(body=found, type_ignores=[]), path, "exec"), g)
    return g


g = load_funcs(os.path.join(HERE, "cluster.py"), ["project_2d", "empty_result_remarks"],
               {"np": np, "PCA": PCA, "LinearDiscriminantAnalysis": LinearDiscriminantAnalysis})
project_2d, empty_result_remarks = g["project_2d"], g["empty_result_remarks"]
h = load_funcs(os.path.join(HERE, "cluster_run_job.py"), ["error_remarks"], {})
error_remarks = h["error_remarks"]

fails = []
def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  ({detail})" if detail else ""))
    if not cond:
        fails.append(name)

rng = np.random.default_rng(0)
sc = StandardScaler()

# 1. the 170584 shape: 10 clusters, 5 columns but rank 1 after scaling
base = rng.normal(size=(120, 1))
X1 = sc.fit_transform(np.hstack([base * k for k in (1, 2, 3, 4, 5)]))
y1 = np.repeat(np.arange(10), 12)
lda_cols = LinearDiscriminantAnalysis(n_components=2).fit_transform(X1, y1).shape[1]
check("precondition: LDA really collapses on this input", lda_cols < 2, f"LDA cols={lda_cols}")
mp = project_2d(X1, y1)
check("1 collapsed-LDA input -> 2 columns", mp.shape == (120, 2), str(mp.shape))
check("1 values finite", bool(np.isfinite(mp).all()))

# 2. CONTROL: full-rank, well separated -> LDA path, 2 columns, separates classes
y2 = np.repeat(np.arange(4), 30)
X2 = sc.fit_transform(rng.normal(size=(120, 5)) + y2[:, None] * np.array([3, -2, 1, 0.5, 2]))
mp2 = project_2d(X2, y2)
ref = LinearDiscriminantAnalysis(n_components=2).fit_transform(X2, y2)
check("2 CONTROL full-rank -> 2 columns", mp2.shape == (120, 2), str(mp2.shape))
check("2 CONTROL full-rank -> identical to plain LDA", bool(np.allclose(mp2, ref)))

# 3. <3 clusters -> PCA (upstream behaviour)
y3 = np.repeat([0, 1], 20)
X3 = sc.fit_transform(rng.normal(size=(40, 5)))
check("3 two clusters -> PCA", bool(np.allclose(np.abs(project_2d(X3, y3)),
                                              np.abs(PCA(n_components=2).fit_transform(X3)))))

# 4. degenerate sizes
check("4a single point -> (1,2)", project_2d(np.zeros((1, 5)), np.array([0])).shape == (1, 2))
check("4b single feature -> (n,2)",
      project_2d(rng.normal(size=(9, 1)), np.repeat([0, 1, 2], 3)).shape == (9, 2))

# 5. remarks
r = empty_result_remarks(345, 0.1)
check("5a noise remark names count+eps", "345" in r and "0.1" in r, r)
check("5b no-detections remark", "no detections" in empty_result_remarks(0, 0.3))

# 6. error remarks
e = error_remarks(IndexError("index 1 is out of bounds\nfor axis 1 with size 1" + "x" * 1000))
check("6 error remark one line, bounded, names type",
      "\n" not in e and len(e) < 500 and "IndexError" in e, f"len={len(e)}")

print(f"\n{'FAILED ' + str(len(fails)) if fails else 'ALL PASSED'}")
sys.exit(1 if fails else 0)