"""
test_xjtu_generic_samplers.py — unit tests for the dataset-agnostic N-way
balanced sampler and bearing-uniform sampler prepared ahead of the XJTU-SY
third-dataset experiment (src.joint_ssl.balanced_batch_indices_n,
natural_proportion_n, bearing_uniform_batch_indices).

These test PURE ARRAY/INDEX LOGIC against synthetic pool sizes and synthetic
bearing-id -> window-count mappings. None of them require XJTU-SY (or any
other real dataset) to exist — see
docs/project_review/XJTU_SY_DATA_AUDIT.md for why the dataset itself is not
yet available and what remains blocked as a result.

Also verifies the EXISTING 2-pool balanced_batch_indices / natural_proportion
are completely unaffected by this addition (same behaviour, same tests
passing as before this file existed).

Run:
  python tests/test_xjtu_generic_samplers.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.joint_ssl import (
    balanced_batch_indices,
    balanced_batch_indices_n,
    bearing_uniform_batch_indices,
    bearing_uniform_cycle,
    flat_uniform_cycle,
    mixed_balanced_batch_indices,
    natural_proportion,
    natural_proportion_n,
)

GREEN, RED, RESET = "\033[32m", "\033[31m", "\033[0m"
n_pass = n_fail = 0


def check(label: str, cond: bool) -> None:
    global n_pass, n_fail
    if cond:
        n_pass += 1
        print(f"  {label:72s} {GREEN}PASS{RESET}")
    else:
        n_fail += 1
        print(f"  {label:72s} {RED}FAIL{RESET}")


print("\n" + "=" * 78)
print("  XJTU-prep generic sampler tests (N-way balancing, bearing-uniform)")
print("=" * 78)

# ---------------------------------------------------------------------------
# balanced_batch_indices_n
# ---------------------------------------------------------------------------

rng = np.random.default_rng(0)
batches = list(balanced_batch_indices_n([100, 100, 100], batch_size=60, rng=rng))
check("N01 three equal pools: every batch has exactly batch_size//3 from each",
      all(len(a) == 20 and len(b) == 20 and len(c) == 20 for a, b, c in batches))
check("N02 three equal pools: epoch length = 100 // 20 = 5", len(batches) == 5)

rng = np.random.default_rng(0)
batches_imb = list(balanced_batch_indices_n([20, 200, 60], batch_size=60, rng=rng))
check("N03 imbalanced 3 pools: epoch length set by the LARGEST pool (200//20=10)",
      len(batches_imb) == 10)
check("N04 imbalanced 3 pools: every batch still exactly equal per pool",
      all(len(a) == 20 and len(b) == 20 and len(c) == 20 for a, b, c in batches_imb))
all_a_idx = np.concatenate([a for a, b, c in batches_imb])
check("N05 the smallest pool (n=20) is cycled — indices repeat across the epoch",
      len(all_a_idx) > 20 and set(all_a_idx.tolist()) == set(range(20)))

# Dataset-selection-frequency check (per instruction: report frequencies and
# max deviation from intended balance over many draws).
rng = np.random.default_rng(1)
big_run = list(balanced_batch_indices_n([500, 5000, 1500], batch_size=90, rng=rng))
counts = [sum(len(b[i]) for b in big_run) for i in range(3)]
total = sum(counts)
freqs = [c / total for c in counts]
max_dev = max(abs(f - 1 / 3) for f in freqs)
print(f"    Dataset selection frequencies over {len(big_run)} batches: "
      f"{[round(f, 4) for f in freqs]}  (target 0.3333 each)")
print(f"    Maximum deviation from intended balance: {max_dev:.6f}")
check("N06 dataset selection frequencies are EXACTLY 1/3 each by construction",
      max_dev < 1e-9)

rng = np.random.default_rng(0)
b1 = list(balanced_batch_indices_n([50, 50, 50], 30, rng))
rng2 = np.random.default_rng(0)
b2 = list(balanced_batch_indices_n([50, 50, 50], 30, rng2))
check("N07 deterministic given the same seeded Generator",
      all(all(np.array_equal(x, y) for x, y in zip(t1, t2)) for t1, t2 in zip(b1, b2)))

try:
    list(balanced_batch_indices_n([0, 10, 10], 30, np.random.default_rng(0)))
    check("N08 empty pool raises ValueError", False)
except ValueError:
    check("N08 empty pool raises ValueError", True)

try:
    list(balanced_batch_indices_n([10], 30, np.random.default_rng(0)))
    check("N09 fewer than 2 pools raises ValueError", False)
except ValueError:
    check("N09 fewer than 2 pools raises ValueError", True)

# N=2 sanity: independent implementation, but same external contract as the
# original 2-pool function (does not modify or call it).
rng = np.random.default_rng(0)
n2_batches = list(balanced_batch_indices_n([80, 80], batch_size=20, rng=rng))
check("N10 N=2 case: every batch exactly half/half (independent of the "
     "original 2-pool function)",
     all(len(a) == 10 and len(b) == 10 for a, b in n2_batches))

# ---------------------------------------------------------------------------
# natural_proportion_n
# ---------------------------------------------------------------------------

check("P01 three equal pools -> proportions [1/3, 1/3, 1/3]",
      all(abs(p - 1 / 3) < 1e-12 for p in natural_proportion_n([10, 10, 10])))
props = natural_proportion_n([100, 200, 700])
check("P02 proportions sum to 1", abs(sum(props) - 1.0) < 1e-12)
check("P03 correctness of a non-trivial ratio",
      abs(props[0] - 0.1) < 1e-12 and abs(props[1] - 0.2) < 1e-12
      and abs(props[2] - 0.7) < 1e-12)
try:
    natural_proportion_n([0, 0])
    check("P04 all-zero pools raise ValueError", False)
except ValueError:
    check("P04 all-zero pools raise ValueError", True)

# ---------------------------------------------------------------------------
# bearing_uniform_batch_indices
# ---------------------------------------------------------------------------

# Mimics XJTU-SY's large lifetime imbalance: one bearing with 10,000 windows
# (long-lived), four bearings with 100 windows each (short-lived).
bearing_sizes = {"long_lived": 10_000, "short_1": 100, "short_2": 100,
                 "short_3": 100, "short_4": 100}
rng = np.random.default_rng(0)
draws = bearing_uniform_batch_indices(bearing_sizes, n_samples=500, rng=rng)
from collections import Counter
draw_counts = Counter(bid for bid, _ in draws)
check("B01 all 5 bearings appear in the draws", set(draw_counts) == set(bearing_sizes))
counts = list(draw_counts.values())
check("B02 long-lived bearing does NOT dominate: per-bearing draw counts "
     f"differ by at most 1 ({counts})",
     max(counts) - min(counts) <= 1)
check("B03 each bearing drawn ~n_samples/n_bearings times (500/5=100)",
      all(abs(c - 100) <= 1 for c in counts))

for bid, w in draws:
    if not (0 <= w < bearing_sizes[bid]):
        check("B04 every window index is within its bearing's valid range", False)
        break
else:
    check("B04 every window index is within its bearing's valid range", True)

rng1 = np.random.default_rng(7)
rng2 = np.random.default_rng(7)
d1 = bearing_uniform_batch_indices(bearing_sizes, 200, rng1)
d2 = bearing_uniform_batch_indices(bearing_sizes, 200, rng2)
check("B05 deterministic given the same seeded Generator", d1 == d2)

try:
    bearing_uniform_batch_indices({}, 10, np.random.default_rng(0))
    check("B06 empty bearing_sizes raises ValueError", False)
except ValueError:
    check("B06 empty bearing_sizes raises ValueError", True)

try:
    bearing_uniform_batch_indices({"b1": 0, "b2": 10}, 10, np.random.default_rng(0))
    check("B07 a zero-window bearing raises ValueError", False)
except ValueError:
    check("B07 a zero-window bearing raises ValueError", True)

# n_samples not a multiple of n_bearings: counts still differ by <= 1.
rng = np.random.default_rng(3)
odd_draws = bearing_uniform_batch_indices(bearing_sizes, n_samples=503, rng=rng)
odd_counts = list(Counter(bid for bid, _ in odd_draws).values())
check("B08 non-multiple n_samples still keeps per-bearing counts within 1 of each other",
      max(odd_counts) - min(odd_counts) <= 1)

# ---------------------------------------------------------------------------
# bearing_uniform_cycle + mixed_balanced_batch_indices (the E1-H2 mechanism)
# ---------------------------------------------------------------------------

# Synthetic "XJTU-like" pool: 5 bearings with wildly different window counts,
# analogous to the real ssl_train imbalance (Bearing3_1=73,602 vs
# Bearing3_5=3,306) verified in docs/project_review/XJTU_SY_DATA_AUDIT.md.
xjtu_bearing_sizes = {"long_A": 70_000, "long_B": 40_000, "mid": 10_000,
                      "short_A": 3_000, "short_B": 1_200}
# Flat pool laid out contiguously per bearing, in the order above.
_offsets = {}
_cum = 0
for _bid, _n in xjtu_bearing_sizes.items():
    _offsets[_bid] = _cum
    _cum += _n
xjtu_pool_size = _cum
xjtu_index_map = {
    bid: np.arange(_offsets[bid], _offsets[bid] + n) for bid, n in xjtu_bearing_sizes.items()
}

rng = np.random.default_rng(5)
cyc = bearing_uniform_cycle(xjtu_bearing_sizes, xjtu_index_map, per_pull=10, rng=rng)
pulls = [next(cyc) for _ in range(200)]
check("M01 bearing_uniform_cycle: every pull has exactly per_pull indices",
      all(len(p) == 10 for p in pulls))
all_pulled = np.concatenate(pulls)
check("M02 bearing_uniform_cycle: every pulled index falls in its bearing's flat range",
      all(any(lo <= idx < lo + n for lo, n in
              [(_offsets[b], xjtu_bearing_sizes[b]) for b in xjtu_bearing_sizes])
          for idx in all_pulled[:50]))  # spot check a subset for speed
# Reconstruct which bearing each pulled index belongs to and confirm balance.
def _bearing_of(idx):
    for bid, lo in _offsets.items():
        if lo <= idx < lo + xjtu_bearing_sizes[bid]:
            return bid
    return None
bearing_hits = Counter(_bearing_of(i) for i in all_pulled)
counts = list(bearing_hits.values())
check(f"M03 bearing_uniform_cycle: long_A (70k) does NOT dominate over 2000 pulls "
     f"({dict(bearing_hits)})",
     max(counts) - min(counts) <= 5)

rng1 = np.random.default_rng(9)
rng2 = np.random.default_rng(9)
c1 = bearing_uniform_cycle(xjtu_bearing_sizes, xjtu_index_map, 10, rng1)
c2 = bearing_uniform_cycle(xjtu_bearing_sizes, xjtu_index_map, 10, rng2)
check("M04 bearing_uniform_cycle: deterministic given the same seeded Generator",
      all(np.array_equal(next(c1), next(c2)) for _ in range(20)))

# mixed_balanced_batch_indices: 2 pools, pool 1 (index 1) is the XJTU-like
# bearing-imbalanced pool, pool 0 is a plain flat pool (e.g. Paderborn-like).
rng = np.random.default_rng(0)
special = {1: bearing_uniform_cycle(xjtu_bearing_sizes, xjtu_index_map, per_pull=15, rng=rng)}
batches = list(mixed_balanced_batch_indices([1000, xjtu_pool_size], batch_size=30, rng=rng,
                                            special_cycles=special))
check("M05 mixed_balanced_batch_indices: every batch has per_pool=15 from EACH pool",
      all(len(a) == 15 and len(b) == 15 for a, b in batches))
n_batches_expected = max(1000, xjtu_pool_size) // 15
check("M06 mixed_balanced_batch_indices: epoch length matches the largest pool",
      len(batches) == n_batches_expected)
xjtu_side_all = np.concatenate([b for _a, b in batches])
xjtu_bearing_hits = Counter(_bearing_of(i) for i in xjtu_side_all)
xjtu_counts = list(xjtu_bearing_hits.values())
check(f"M07 mixed_balanced_batch_indices: XJTU-side draws stay bearing-balanced "
     f"across the whole run ({dict(xjtu_bearing_hits)})",
     max(xjtu_counts) - min(xjtu_counts) < 0.15 * max(xjtu_counts))

# Non-special pools behave exactly like balanced_batch_indices_n's plain case.
rng = np.random.default_rng(0)
plain_via_mixed = list(mixed_balanced_batch_indices([50, 50, 50], 30, rng, special_cycles=None))
rng2 = np.random.default_rng(0)
plain_via_n = list(balanced_batch_indices_n([50, 50, 50], 30, rng2))
check("M08 mixed_balanced_batch_indices with no special_cycles matches "
     "balanced_batch_indices_n exactly",
     all(all(np.array_equal(x, y) for x, y in zip(t1, t2))
         for t1, t2 in zip(plain_via_mixed, plain_via_n)))

try:
    list(mixed_balanced_batch_indices([0, 10], 20, np.random.default_rng(0)))
    check("M09 mixed_balanced_batch_indices: empty pool raises ValueError", False)
except ValueError:
    check("M09 mixed_balanced_batch_indices: empty pool raises ValueError", True)

try:
    list(mixed_balanced_batch_indices([10], 20, np.random.default_rng(0)))
    check("M10 mixed_balanced_batch_indices: fewer than 2 pools raises ValueError", False)
except ValueError:
    check("M10 mixed_balanced_batch_indices: fewer than 2 pools raises ValueError", True)

# ---------------------------------------------------------------------------
# flat_uniform_cycle + mixed_balanced_batch_indices's n_batches override
# (fixed-compute-budget mechanism for the equal-compute E1-H2 protocol)
# ---------------------------------------------------------------------------

rng = np.random.default_rng(2)
fcyc = flat_uniform_cycle(50, per_pull=10, rng=rng)
fpulls = [next(fcyc) for _ in range(30)]
check("FC01 flat_uniform_cycle: every pull has exactly per_pull indices",
      all(len(p) == 10 for p in fpulls))
check("FC02 flat_uniform_cycle: every pulled index in range(n)",
      all(0 <= i < 50 for p in fpulls for i in p))
rng1 = np.random.default_rng(3)
rng2 = np.random.default_rng(3)
c1 = flat_uniform_cycle(50, 10, rng1)
c2 = flat_uniform_cycle(50, 10, rng2)
check("FC03 flat_uniform_cycle: deterministic given the same seeded Generator",
      all(np.array_equal(next(c1), next(c2)) for _ in range(15)))
try:
    next(flat_uniform_cycle(0, 10, np.random.default_rng(0)))
    check("FC04 flat_uniform_cycle: empty pool raises ValueError", False)
except ValueError:
    check("FC04 flat_uniform_cycle: empty pool raises ValueError", True)

# n_batches override: a FIXED epoch length regardless of pool sizes,
# including when it exceeds every pool's own single-pass length (forcing
# with-replacement cycling on pools that would otherwise be "the largest").
rng = np.random.default_rng(0)
FIXED_N_BATCHES = 20
fixed_batches = list(mixed_balanced_batch_indices(
    [100, 100_000], batch_size=20, rng=rng, n_batches=FIXED_N_BATCHES))
check("NB01 n_batches override: epoch length is EXACTLY the fixed value, "
     "independent of the 100,000-window pool",
     len(fixed_batches) == FIXED_N_BATCHES)
check("NB02 n_batches override: every batch still exactly per_pool from each pool",
      all(len(a) == 10 and len(b) == 10 for a, b in fixed_batches))

# Confirm default (no override) reproduces the ORIGINAL max(pool_sizes)//per_pool
# behaviour exactly -- i.e. the new parameter is purely additive.
rng = np.random.default_rng(0)
default_batches = list(mixed_balanced_batch_indices([100, 100_000], batch_size=20, rng=rng))
check("NB03 no n_batches given: epoch length still follows the largest pool "
     f"(100000//10={100_000//10}, got {len(default_batches)})",
     len(default_batches) == 100_000 // 10)

# A fixed budget SMALLER than a pool's natural single pass still draws exactly
# that many batches (under-uses the pool, by design -- this is the intended
# "equal compute" behaviour: large pools are capped, not automatically fully
# consumed).
rng = np.random.default_rng(0)
small_fixed = list(mixed_balanced_batch_indices([500, 500], batch_size=20, rng=rng, n_batches=3))
check("NB04 a fixed budget smaller than either pool's natural length is honoured exactly",
      len(small_fixed) == 3)

# ---------------------------------------------------------------------------
# Existing 2-pool functions are completely unaffected by this addition.
# ---------------------------------------------------------------------------

rng = np.random.default_rng(0)
existing_batches = list(balanced_batch_indices(100, 100, batch_size=20, rng=rng))
check("E01 existing balanced_batch_indices (2-pool) still works unmodified",
      len(existing_batches) == 10
      and all(len(a) == 10 and len(b) == 10 for a, b in existing_batches))
check("E02 existing natural_proportion still works unmodified",
      abs(natural_proportion(30, 70) - 0.3) < 1e-12)

print()
sep = "=" * 78
if n_fail == 0:
    print(f"{sep}\n  {GREEN}ALL {n_pass} TESTS PASSED{RESET}\n{sep}")
else:
    print(f"{sep}\n  {RED}{n_fail} TEST(S) FAILED{RESET}  ({n_pass} passed)\n{sep}")
sys.exit(0 if n_fail == 0 else 1)
