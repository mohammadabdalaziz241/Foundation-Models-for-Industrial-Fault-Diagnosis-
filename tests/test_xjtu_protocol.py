"""
test_xjtu_protocol.py — protocol validation and unit tests for the XJTU-SY
third-dataset SSL source (Conditions E1-H2; see
docs/project_review/XJTU_THIRD_DATASET_EXPERIMENT_PLAN.md).

Requires the real archive at data/raw_xjtu_sy and the prebuilt
data/processed_xjtu_sy (run `python -m src.run_preprocessing_xjtu` first).

Checklist covered here:
  1.  Archive facts verified against real files (bearing count, condition
      count, header, row count, resample ratio) — see also
      docs/project_review/XJTU_SY_DATA_AUDIT.md for the full per-file scan.
  2.  Channel order is name-based, not positional; a mislabelled header
      raises rather than being silently accepted.
  3.  Resampling ratio is EXACTLY 15/32 and length-exact on a real recording.
  4.  Windowing matches the 1024/50%-overlap contract (via the real prebuilt
      arrays, which is what the training pipeline actually consumes).
  5.  Whole-bearing SSL split: disjoint, covers all bearings, all three
      conditions represented in ssl_val.
  6.  Normalisation leakage: the scaler was fit on ssl_train ONLY (ssl_train
      windows are ~exactly zero-mean/unit-std post-transform; ssl_val is
      measurably NOT, proving it did not contribute to the fit).
  7.  Dataset-balancing / bearing-balancing wiring: pretrain_ssl composes
      the correct pools for each of the four new ssl_source values (E/F/G/H),
      and no XJTU-only pool ever includes a Paderborn sealed-test bearing
      (structurally impossible: XJTU pools carry XJTU bearing IDs only, but
      checked explicitly for conditions that DO combine with Paderborn).
  8.  Manifest completeness (data/manifests/xjtu_sy_manifest.csv, if built).

Run:
  python tests/test_xjtu_protocol.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cv_paderborn import TEST_BEARINGS
from src.joint_ssl import bearing_uniform_cycle, mixed_balanced_batch_indices
from src.xjtu_loader import (
    ALL_BEARINGS,
    BEARING_TO_CONDITION,
    CONDITIONS,
    CWRU_FS,
    EXPECTED_HEADER,
    EXPECTED_ROWS,
    RESAMPLE_DOWN,
    RESAMPLE_UP,
    XJTU_FS,
    list_recordings,
    load_xjtu_horizontal,
    read_xjtu_csv,
    resample_to_cwru_rate,
)
from src.xjtu_preprocessing import (
    bearing_index_map,
    bearing_window_counts,
    load_xjtu_split,
    split_xjtu_ssl_val_bearings,
    verify_no_leakage,
)
from scripts.train_reverse_transfer_cwru import (
    XJTU_SOURCES,
    SSL_SOURCES,
    prepare_paderborn_pool,
    prepare_xjtu_pool,
    pretrain_ssl,
)
from src.reverse_transfer import derive_condition_seeds

GREEN, RED, RESET = "\033[32m", "\033[31m", "\033[0m"
n_pass = n_fail = 0


def check(label: str, cond: bool) -> None:
    global n_pass, n_fail
    if cond:
        n_pass += 1
        print(f"  {label:74s} {GREEN}PASS{RESET}")
    else:
        n_fail += 1
        print(f"  {label:74s} {RED}FAIL{RESET}")


print("\n" + "=" * 80)
print("  XJTU-SY third-dataset protocol validation + unit tests")
print("=" * 80)

# ---------------------------------------------------------------------------
# 1. Archive facts (real files)
# ---------------------------------------------------------------------------

recs = list_recordings()
check("A01 15 physical bearings discovered", len({r.bearing for r in recs}) == 15)
check("A02 3 operating conditions discovered", len({r.condition for r in recs}) == 3)
check("A03 9216 total recordings discovered", len(recs) == 9216)
check("A04 every bearing's recording indices are contiguous from 1",
      all(
          sorted(r.recording_index for r in recs if r.bearing == b)
          == list(range(1, 1 + sum(1 for r in recs if r.bearing == b)))
          for b in ALL_BEARINGS
      ))

sample = next(r for r in recs if r.bearing == "Bearing1_1" and r.recording_index == 1)
h, v = read_xjtu_csv(sample.path)
check("A05 real recording has exactly 32768 rows", len(h) == EXPECTED_ROWS == len(v))
check("A06 XJTU_FS matches the archive's documented nominal rate", XJTU_FS == 25_600)

# ---------------------------------------------------------------------------
# 2. Channel order is name-based
# ---------------------------------------------------------------------------

import pandas as pd
import tempfile

check("A07 EXPECTED_HEADER matches the real archive's column names exactly",
      list(pd.read_csv(sample.path, nrows=0).columns) == EXPECTED_HEADER)

with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
    f.write("Vertical_vibration_signals,Horizontal_vibration_signals\n1.0,2.0\n3.0,4.0\n")
    bad_path = Path(f.name)
try:
    read_xjtu_csv(bad_path)
    check("A08 swapped/mislabelled header raises ValueError (never silently accepted)", False)
except ValueError:
    check("A08 swapped/mislabelled header raises ValueError (never silently accepted)", True)
finally:
    bad_path.unlink()

# ---------------------------------------------------------------------------
# 3. Resampling ratio and length-exactness
# ---------------------------------------------------------------------------

check("A09 resample ratio is EXACTLY 15/32 (12000/25600, gcd=800)",
      RESAMPLE_UP == 15 and RESAMPLE_DOWN == 32
      and CWRU_FS * RESAMPLE_DOWN == XJTU_FS * RESAMPLE_UP)

res = resample_to_cwru_rate(h)
check("A10 a real 32768-sample recording resamples to EXACTLY 15360 samples "
     f"(32768*15/32, got {len(res)})",
     len(res) == 32768 * 15 // 32 == 15360)
check("A11 resampled signal has no NaN/Inf", np.isfinite(res).all())

h_direct = load_xjtu_horizontal(sample.path)
check("A12 load_xjtu_horizontal matches manual read+resample_to_cwru_rate",
      np.allclose(h_direct, res))
try:
    load_xjtu_horizontal(sample.path, target_fs=25_600)
    check("A13 load_xjtu_horizontal rejects a non-12kHz target_fs", False)
except ValueError:
    check("A13 load_xjtu_horizontal rejects a non-12kHz target_fs", True)

# ---------------------------------------------------------------------------
# 4. Windowing (via the real prebuilt arrays the pipeline actually consumes)
# ---------------------------------------------------------------------------

X_ssltr, b_ssltr = load_xjtu_split("ssl_train")
X_sslva, b_sslva = load_xjtu_split("ssl_val")
check("A14 ssl_train windows are (N, 1024) float32", X_ssltr.ndim == 2 and X_ssltr.shape[1] == 1024
      and X_ssltr.dtype == np.float32)
check("A15 ssl_val windows are (N, 1024) float32", X_sslva.ndim == 2 and X_sslva.shape[1] == 1024
      and X_sslva.dtype == np.float32)
check("A16 windows-per-recording matches the 1024/50%-overlap formula "
     "((15360-1024)//512+1 = 29)",
     29 == (15360 - 1024) // 512 + 1)
expected_total = 9216 * 29
check(f"A17 total window count across both splits matches 9216 recordings * 29 windows "
     f"(expected {expected_total}, got {len(X_ssltr) + len(X_sslva)})",
     len(X_ssltr) + len(X_sslva) == expected_total)

# ---------------------------------------------------------------------------
# 5. Whole-bearing SSL split
# ---------------------------------------------------------------------------

ssl_train_b, ssl_val_b = split_xjtu_ssl_val_bearings()
verify_no_leakage(ssl_train_b, ssl_val_b)  # raises on failure
check("A18 whole-bearing split covers all 15 bearings exactly once",
      sorted(ssl_train_b + ssl_val_b) == sorted(ALL_BEARINGS))
check("A19 ssl_val covers all 3 operating conditions",
      {BEARING_TO_CONDITION[b] for b in ssl_val_b} == set(CONDITIONS))
check("A20 ssl_val is a strict, non-trivial subset (3 of 15 bearings)",
      0 < len(ssl_val_b) < len(ALL_BEARINGS))
check("A21 real ssl_train/ssl_val window arrays only contain their declared bearings",
      set(np.unique(b_ssltr).tolist()) == set(ssl_train_b)
      and set(np.unique(b_sslva).tolist()) == set(ssl_val_b))
check("A22 split is deterministic (same fraction/seed -> same bearings)",
      split_xjtu_ssl_val_bearings() == (ssl_train_b, ssl_val_b))
different_seed_tr, different_seed_va = split_xjtu_ssl_val_bearings(seed=7)
check("A23 a different seed changes which bearings are held out (not hardcoded)",
      set(different_seed_va) != set(ssl_val_b))

# ---------------------------------------------------------------------------
# 6. Normalisation leakage
# ---------------------------------------------------------------------------

check("A24 ssl_train is ~exactly zero-mean after its own scaler's transform "
     f"(mean={X_ssltr.mean():.2e})", abs(float(X_ssltr.mean())) < 1e-4)
check("A25 ssl_train is ~exactly unit-std after its own scaler's transform "
     f"(std={X_ssltr.std():.4f})", abs(float(X_ssltr.std()) - 1.0) < 1e-3)
check("A26 ssl_val is MEASURABLY off zero-mean/unit-std under the SAME scaler "
     f"(mean={X_sslva.mean():.4f}, std={X_sslva.std():.4f}) — proves ssl_val "
     "did not contribute to the scaler fit",
     abs(float(X_sslva.mean())) > 1e-4 or abs(float(X_sslva.std()) - 1.0) > 1e-3)

# ---------------------------------------------------------------------------
# 7. Dataset-balancing / bearing-balancing wiring for E/F/G/H
# ---------------------------------------------------------------------------

check("A27 all four new ssl_source values registered in SSL_SOURCES",
      {"xjtu_only", "paderborn_xjtu_balanced", "cwru_xjtu_balanced",
       "cwru_paderborn_xjtu_balanced"} <= set(SSL_SOURCES))
check("A28 XJTU_SOURCES set matches exactly the four new xjtu-inclusive sources",
      XJTU_SOURCES == {"xjtu_only", "paderborn_xjtu_balanced", "cwru_xjtu_balanced",
                       "cwru_paderborn_xjtu_balanced"})

bwc = bearing_window_counts(b_ssltr)
bim = bearing_index_map(b_ssltr)
check("A29 bearing_window_counts sums to the full ssl_train pool size",
      sum(bwc.values()) == len(X_ssltr))
check("A30 bearing_index_map indices are all in range and bearing-consistent",
      all(np.all(b_ssltr[idx] == bid) for bid, idx in bim.items()))

rng = np.random.default_rng(0)
cyc = bearing_uniform_cycle(bwc, bim, per_pull=32, rng=rng)
pulled = np.concatenate([next(cyc) for _ in range(50)])
check("A31 bearing_uniform_cycle wired to REAL XJTU bearing data draws valid "
     "flat indices into the real pool",
     pulled.min() >= 0 and pulled.max() < len(X_ssltr))
from collections import Counter
hit_bearings = Counter(str(b_ssltr[i]) for i in pulled)
check("A32 bearing_uniform_cycle on REAL data visits multiple distinct bearings "
     f"(not dominated by the largest) — {dict(hit_bearings)}",
     len(hit_bearings) >= 5)

# End-to-end pool-composition contract: run pretrain_ssl for one real epoch
# (max_batches capped) for each of the four new sources and confirm each
# activates exactly the pools its name implies.
X_pad_ssltr, X_pad_sslva, _pad_meta = prepare_paderborn_pool(ssl_val_fraction=0.2, ssl_val_seed=42)
X_xjtu_ssltr, X_xjtu_sslva, b_xjtu_ssltr, b_xjtu_sslva, _xjtu_meta = prepare_xjtu_pool()


def _smoke_args(ssl_source: str) -> argparse.Namespace:
    return argparse.Namespace(
        model="cnn1d", batch_size=16, lr=1e-3, ssl_lr=None, ssl_epochs=1,
        optimizer="adam", weight_decay=0.0, grad_clip=0.0, max_batches=1,
        smoothing_window=3, ssl_val_seed=42,
        ssl_batches_per_epoch=None, ssl_val_cap_per_source=None,
    )


expected_pools = {
    "xjtu_only": {"cwru": False, "paderborn": False, "xjtu": True},
    "paderborn_xjtu_balanced": {"cwru": False, "paderborn": True, "xjtu": True},
    "cwru_xjtu_balanced": {"cwru": True, "paderborn": False, "xjtu": True},
    "cwru_paderborn_xjtu_balanced": {"cwru": True, "paderborn": True, "xjtu": True},
}

for src, expect in expected_pools.items():
    seeds = derive_condition_seeds(42, 0)
    _enc, meta = pretrain_ssl(
        src, seeds, X_pad_ssltr, X_pad_sslva, _smoke_args(src),
        ROOT / "tests" / "_tmp_xjtu_wiring" / src,
        X_xjtu_ssltr=X_xjtu_ssltr, X_xjtu_sslva=X_xjtu_sslva, b_xjtu_ssltr=b_xjtu_ssltr,
    )
    got = {
        "cwru": meta["n_cwru_train_windows"] > 0,
        "paderborn": meta["n_paderborn_ssl_train_windows"] > 0,
        "xjtu": meta["n_xjtu_ssl_train_windows"] > 0,
    }
    check(f"A33 [{src}] activates exactly the pools its name implies {expect}", got == expect)
    has_xjtu_val = "xjtu_val_recon" in meta["history"][0]
    check(f"A34 [{src}] XJTU validation reconstruction loss recorded", has_xjtu_val)

# No XJTU pool bearing ID ever collides with a Paderborn sealed-test bearing
# code (structurally guaranteed — XJTU bearing IDs are "Bearing<c>_<n>"
# strings, Paderborn codes are "K00x"/"KA.."/"KI.." — checked explicitly since
# conditions F/H combine both datasets in the same SSL batch).
check("A35 no XJTU bearing ID string collides with any sealed Paderborn test bearing code",
      not (set(ALL_BEARINGS) & TEST_BEARINGS))

import shutil
shutil.rmtree(ROOT / "tests" / "_tmp_xjtu_wiring", ignore_errors=True)

# ---------------------------------------------------------------------------
# 8. Manifest completeness (if built)
# ---------------------------------------------------------------------------

manifest_path = ROOT / "data" / "manifests" / "xjtu_sy_manifest.csv"
if manifest_path.exists():
    rows = list(csv.DictReader(open(manifest_path)))
    check(f"A36 manifest has one row per real CSV file (9216, got {len(rows)})",
          len(rows) == 9216)
    check("A37 manifest covers all 15 bearings",
          {r["bearing"] for r in rows} == set(ALL_BEARINGS))
    check("A38 manifest records a validation_status for every row",
          all(r.get("validation_status") for r in rows))
else:
    print(f"  (A36-A38 skipped: {manifest_path} not yet built)")

print()
sep = "=" * 80
if n_fail == 0:
    print(f"{sep}\n  {GREEN}ALL {n_pass} TESTS PASSED{RESET}\n{sep}")
else:
    print(f"{sep}\n  {RED}{n_fail} TEST(S) FAILED{RESET}  ({n_pass} passed)\n{sep}")
sys.exit(0 if n_fail == 0 else 1)
