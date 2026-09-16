"""Unit tests for the F2 class-conditional INLP scrub (synthetic data only).

Run: python tests/test_inlp_scrub.py   (exit 0 iff all pass)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inlp_scrub import (fit_class_conditional_directions, inlp_scrub,
                            orthonormalize_against, project_nullspace)

GREEN, RED, RESET = "\033[92m", "\033[91m", "\033[0m"
PASS, FAIL = f"{GREEN}PASS{RESET}", f"{RED}FAIL{RESET}"
n_pass = n_fail = 0


def check(label, ok, suffix=""):
    global n_pass, n_fail
    tag = PASS if ok else FAIL
    print(f"  T{n_pass + n_fail + 1:02d} {label:60s} {tag}{suffix}")
    if ok:
        n_pass += 1
    else:
        n_fail += 1


def probe_acc(X, y, seed=0):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(X))
    n_tr = int(0.7 * len(X))
    clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(X[idx[:n_tr]], y[idx[:n_tr]])
    return clf.score(X[idx[n_tr:]], y[idx[n_tr:]])


def synthetic(n_per=120, d=32, seed=0):
    """Class signal in dims 0-1, bearing signal in dims 2-4, noise elsewhere.
    2 classes x 3 bearings each."""
    rng = np.random.RandomState(seed)
    X, cls, brg = [], [], []
    bearing_means = rng.randn(6, 3) * 4.0
    class_means = np.array([[3.0, -3.0], [-3.0, 3.0]])
    for b in range(6):
        c = b // 3
        x = rng.randn(n_per, d)
        x[:, 0:2] += class_means[c]
        x[:, 2:5] += bearing_means[b]
        X.append(x)
        cls += [c] * n_per
        brg += [b] * n_per
    return np.vstack(X), np.array(cls), np.array(brg)


# --- orthonormalisation ----------------------------------------------------
q = orthonormalize_against(np.array([[2.0, 0, 0], [0, 3.0, 0]]), None)
check("orthonormalize returns orthonormal rows",
      q.shape == (2, 3) and np.allclose(q @ q.T, np.eye(2), atol=1e-10))

q = orthonormalize_against(np.array([[1.0, 1.0, 0.0]]),
                           np.array([[1.0, 0.0, 0.0]]))
check("orthonormalize removes overlap with existing directions",
      q.shape == (1, 3) and np.allclose(abs(q[0]), [0, 1, 0], atol=1e-8))

q = orthonormalize_against(np.array([[2.0, 0.0, 0.0]]),
                           np.array([[1.0, 0.0, 0.0]]))
check("orthonormalize drops fully-dependent directions", q.shape[0] == 0)

# --- projection ------------------------------------------------------------
rng = np.random.RandomState(0)
X8 = rng.randn(50, 8)
d8 = orthonormalize_against(rng.randn(2, 8), None)
Xp = project_nullspace(X8, d8)
check("projection is idempotent",
      np.allclose(Xp, project_nullspace(Xp, d8), atol=1e-10))
check("projection kills the removed span",
      np.allclose(Xp @ d8.T, 0.0, atol=1e-10))
check("projection with no directions is a copy",
      np.allclose(project_nullspace(X8, None), X8))

# --- the core behaviour ----------------------------------------------------
X, cls, brg = synthetic()
pre_b, pre_c = probe_acc(X, brg), probe_acc(X, cls)
check("synthetic sanity: bearing and class probes start high",
      pre_b > 0.9 and pre_c > 0.9, f"  (b={pre_b:.2f}, c={pre_c:.2f})")

dirs, hist = inlp_scrub(X, brg, cls, max_rank=16)
Xs = project_nullspace(X, dirs)
post_b, post_c = probe_acc(Xs, brg), probe_acc(Xs, cls)
check("scrub kills linear bearing signal", post_b < 0.5, f"  (b={post_b:.2f})")
check("scrub preserves class signal", post_c > 0.85, f"  (c={post_c:.2f})")

dirs3, _ = inlp_scrub(X, brg, cls, max_rank=3)
check("rank cap respected and directions orthonormal",
      len(dirs3) <= 3 and np.allclose(dirs3 @ dirs3.T, np.eye(len(dirs3)),
                                      atol=1e-8))

d1, _ = inlp_scrub(X, brg, cls, max_rank=8, seed=42)
d2, _ = inlp_scrub(X, brg, cls, max_rank=8, seed=42)
check("same-seed determinism", np.allclose(d1, d2))

calls = []
_, hist1 = inlp_scrub(X, brg, cls, max_rank=16,
                      stop_fn=lambda Xc: calls.append(1) or True)
check("stop_fn halts after first round", len(hist1) == 1 and len(calls) == 1)

# --- degenerate class handling --------------------------------------------
Xd = np.random.RandomState(0).randn(90, 8)
cls_d = np.array([0] * 60 + [1] * 30)
brg_d = np.array([0] * 30 + [1] * 30 + [2] * 30)
dd = fit_class_conditional_directions(Xd, brg_d, cls_d)
check("single-bearing class is skipped, multi-bearing class contributes",
      dd.shape[0] >= 1)

# --- summary ---------------------------------------------------------------
print()
sep = "=" * 60
if n_fail == 0:
    print(f"{sep}\n  {GREEN}ALL {n_pass} TESTS PASSED{RESET}\n{sep}")
else:
    print(f"{sep}\n  {RED}{n_fail} TEST(S) FAILED{RESET}  ({n_pass} passed)\n{sep}")
sys.exit(0 if n_fail == 0 else 1)
