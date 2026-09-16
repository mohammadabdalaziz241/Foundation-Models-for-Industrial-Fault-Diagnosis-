"""
test_compare_cv_runs.py — unit tests for the paired CV comparison utility.

Run:
  python tests/test_compare_cv_runs.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.compare_cv_runs import compare, load_fold_scores, sign_test_p

GREEN, RED, RESET = "\033[32m", "\033[31m", "\033[0m"
n_pass = n_fail = 0


def check(label: str, cond: bool) -> None:
    global n_pass, n_fail
    if cond:
        n_pass += 1
        print(f"  {label:60s} {GREEN}PASS{RESET}")
    else:
        n_fail += 1
        print(f"  {label:60s} {RED}FAIL{RESET}")


def _write_run(tmp: Path, name: str, scores: dict[tuple[int, int], float]) -> Path:
    d = tmp / name
    d.mkdir(parents=True)
    agg = {"fold_summaries": [
        {"repeat": r, "fold": f, "final_bb_f1": v}
        for (r, f), v in scores.items()
    ]}
    (d / "aggregate.json").write_text(json.dumps(agg))
    return d


print("\n" + "=" * 72)
print("  compare_cv_runs unit tests")
print("=" * 72)

check("C01 sign test: 10/10 wins → p = 2·0.5^10",
      abs(sign_test_p(np.ones(10)) - 2 * 0.5**10) < 1e-12)
check("C02 sign test: balanced wins → p = 1",
      sign_test_p(np.array([1.0, -1.0] * 5)) == 1.0)
check("C03 sign test: all ties → p = 1",
      sign_test_p(np.zeros(5)) == 1.0)
expect = min(1.0, 2 * sum(comb(9, i) for i in range(0, 2)) / 2**9)
check("C04 sign test: 8/9 wins matches exact binomial",
      abs(sign_test_p(np.array([1.0] * 8 + [-1.0, 0.0])) - expect) < 1e-12)

with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    keys = [(r, f) for r in range(3) for f in range(4)]
    rng = np.random.default_rng(0)
    a_scores = {k: float(v) for k, v in zip(keys, rng.uniform(0.3, 0.7, 12))}
    b_scores = {k: a_scores[k] + 0.05 for k in keys}   # B uniformly better

    da = _write_run(tmp, "run_a", a_scores)
    db = _write_run(tmp, "run_b", b_scores)

    check("C05 load_fold_scores round-trips 12 fold keys",
          len(load_fold_scores(da, "final_bb_f1")) == 12)

    res = compare(da, db, "final_bb_f1", seed=42)
    check("C06 paired diff mean = +0.05 exactly",
          abs(res["diff_mean"] - 0.05) < 1e-12)
    check("C07 12/0 wins detected", res["n_b_wins"] == 12 and res["n_a_wins"] == 0)
    check("C08 sign test significant for uniform improvement",
          res["sign_test_p"] < 0.001)
    check("C09 bootstrap CI excludes zero",
          res["bootstrap_ci95"][0] > 0)
    check("C10 bootstrap CI deterministic given seed",
          compare(da, db, "final_bb_f1", seed=42)["bootstrap_ci95"]
          == res["bootstrap_ci95"])

    # Mismatched folds must raise
    c_scores = {k: 0.5 for k in keys[:8]}
    dc = _write_run(tmp, "run_c", c_scores)
    try:
        compare(da, dc, "final_bb_f1", seed=0)
        check("C11 mismatched fold sets raise ValueError", False)
    except ValueError:
        check("C11 mismatched fold sets raise ValueError", True)

print()
sep = "=" * 72
if n_fail == 0:
    print(f"{sep}\n  {GREEN}ALL {n_pass} TESTS PASSED{RESET}\n{sep}")
else:
    print(f"{sep}\n  {RED}{n_fail} TEST(S) FAILED{RESET}  ({n_pass} passed)\n{sep}")
sys.exit(0 if n_fail == 0 else 1)
