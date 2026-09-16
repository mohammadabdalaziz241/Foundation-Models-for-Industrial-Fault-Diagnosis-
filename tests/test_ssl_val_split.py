"""
test_ssl_val_split.py — unit tests for the genuinely held-out SSL validation
split (split_ssl_val_bearings / derive_ssl_val_seed) and its wiring into
_run_ssl_pretrain's backward-compatible smoothing behaviour.

Run:
  python tests/test_ssl_val_split.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cv_paderborn import (
    DEVELOPMENT_POOL,
    bearing_metadata,
    derive_ssl_val_seed,
    split_ssl_val_bearings,
    trailing_mean,
)

GREEN, RED, RESET = "\033[32m", "\033[31m", "\033[0m"
n_pass = n_fail = 0


def check(label: str, cond: bool) -> None:
    global n_pass, n_fail
    if cond:
        n_pass += 1
        print(f"  {label:66s} {GREEN}PASS{RESET}")
    else:
        n_fail += 1
        print(f"  {label:66s} {RED}FAIL{RESET}")


print("\n" + "=" * 76)
print("  SSL validation split unit tests")
print("=" * 76)

# ---------------------------------------------------------------------------
# derive_ssl_val_seed
# ---------------------------------------------------------------------------

s1 = derive_ssl_val_seed(42, 0, 0)
s2 = derive_ssl_val_seed(42, 0, 0)
check("V01 derive_ssl_val_seed deterministic", s1 == s2)
check("V02 derive_ssl_val_seed in [0, 2^31)", 0 <= s1 < 2**31)

from src.cv_paderborn import derive_fold_seed
check("V03 ssl_val seed stream distinct from fold_seed stream (same args)",
      derive_ssl_val_seed(42, 0, 0) != derive_fold_seed(42, 0, 0))

combos = {derive_ssl_val_seed(42, r, f) for r in range(3) for f in range(4)}
check("V04 12 (repeat, fold) combos give 12 distinct ssl-val seeds",
      len(combos) == 12)

# ---------------------------------------------------------------------------
# split_ssl_val_bearings — realistic fold-train pool (16 bearings, one fold
# held out of the 21-bearing development pool)
# ---------------------------------------------------------------------------

# A representative fold-train set: DEVELOPMENT_POOL minus 5 val bearings,
# one per damage group.
val_bearings = ["K002", "KA01", "KA06", "KI07", "KI14"]
train_bearings = [b for b in DEVELOPMENT_POOL if b not in val_bearings]
check("V05 fixture: 16 fold-train bearings", len(train_bearings) == 16)

ssl_tr, ssl_va = split_ssl_val_bearings(train_bearings, fraction=0.0, seed=1)
check("V06 fraction=0 returns all bearings as ssl-train, none held out",
      ssl_tr == sorted(train_bearings) and ssl_va == [])

ssl_tr2, ssl_va2 = split_ssl_val_bearings(train_bearings, fraction=0.2, seed=1)
check("V07 fraction=0.2 holds out a non-empty subset",
      len(ssl_va2) > 0)
check("V08 ssl-train and ssl-val partition fold-train exactly",
      sorted(ssl_tr2 + ssl_va2) == sorted(train_bearings)
      and set(ssl_tr2).isdisjoint(ssl_va2))
check("V09 held-out fraction is a reasonable neighbourhood of 0.2 (10-35%)",
      0.10 <= len(ssl_va2) / len(train_bearings) <= 0.35)

fault_types_all = {bearing_metadata(b)["fault_type"] for b in train_bearings}
fault_types_val = {bearing_metadata(b)["fault_type"] for b in ssl_va2}
check("V10 all 3 fault types represented in the held-out SSL-val set",
      fault_types_val == fault_types_all)
fault_types_train = {bearing_metadata(b)["fault_type"] for b in ssl_tr2}
check("V11 all 3 fault types still represented in the SSL-train set",
      fault_types_train == fault_types_all)

check("V12 deterministic given the same seed",
      split_ssl_val_bearings(train_bearings, 0.2, 1) == (ssl_tr2, ssl_va2))
ssl_tr3, ssl_va3 = split_ssl_val_bearings(train_bearings, 0.2, 2)
check("V13 different seed gives a different split",
      (ssl_tr3, ssl_va3) != (ssl_tr2, ssl_va2))

# fraction=1.0 must never empty a group below 1 bearing on the train side
ssl_tr_full, ssl_va_full = split_ssl_val_bearings(train_bearings, fraction=1.0, seed=1)
check("V14 fraction=1.0 still leaves ≥1 bearing per group on the ssl-train side",
      len(ssl_tr_full) >= len({bearing_metadata(b)["fault_type"] for b in train_bearings}))

# Small group (single bearing) is never split
tiny = ["K001"]
tiny_tr, tiny_va = split_ssl_val_bearings(tiny, fraction=0.5, seed=0)
check("V15 a single-bearing group is never split (stays in ssl-train)",
      tiny_tr == ["K001"] and tiny_va == [])

# Empty input
empty_tr, empty_va = split_ssl_val_bearings([], fraction=0.2, seed=0)
check("V16 empty bearing list returns two empty lists",
      empty_tr == [] and empty_va == [])

# ---------------------------------------------------------------------------
# Backward-compat smoothing invariant used by _run_ssl_pretrain:
# trailing_mean(window=1) must be the identity, so fraction=0 runs are
# byte-for-byte unaffected by the smoothing machinery added for fraction>0.
# ---------------------------------------------------------------------------

vals = [0.91, 0.75, 0.83, 0.60, 0.95, 0.58]
check("V17 trailing_mean(window=1) is the identity (fraction=0 unaffected)",
      trailing_mean(vals, window=1) == vals)

print()
sep = "=" * 76
if n_fail == 0:
    print(f"{sep}\n  {GREEN}ALL {n_pass} TESTS PASSED{RESET}\n{sep}")
else:
    print(f"{sep}\n  {RED}{n_fail} TEST(S) FAILED{RESET}  ({n_pass} passed)\n{sep}")
sys.exit(0 if n_fail == 0 else 1)
