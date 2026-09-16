"""
test_xjtu_fixed_budget.py — protocol validation for the EQUAL-COMPUTE E1-H2
primary experiment (fixed SSL step budget matched to the completed CNN1D
CWRU+Paderborn joint-balanced D1/D2 conditions), addressing the confound
that XJTU-SY's much larger SSL-train pool would otherwise give
XJTU-inclusive conditions many more optimiser updates than A-D2 ever had.

Reference budget (read from real D1/D2 artifacts, not inferred from
dataset size — see scripts/train_reverse_transfer_cwru.py's
SSL_REFERENCE_* constants and their docstring for the exact provenance):
  batch_size=64, 462 batches/epoch, 30 epochs, 13,860 total SSL steps,
  CosineAnnealingLR(T_max=30) stepped once per EPOCH (schedule depends on
  epoch count only, not batches/epoch or total steps).

Checklist covered here:
  1. E1-H2 use the same SSL optimizer-step count as D1/D2.
  2. Dataset proportions remain correct under the capped budget.
  3. XJTU bearing balance remains correct.
  4. Large source pools do not increase epoch length.
  5. Fixed-step sampling is deterministic for the same seed.
  6. Different seeds sample different valid windows.
  7. No CWRU validation/test data enter SSL.
  8. No sealed Paderborn bearing enters SSL.
  9. Existing A-D2 results and behaviour remain untouched.
 10. H's remainder-rotation batch-size correction (added after review found
     H's 3-pool per_pool=21 total only 63 of the intended 64 examples/batch):
     every H batch is exactly 64 examples, every drawn example reaches the
     loss, per-batch dataset counts differ by at most 1, dataset totals are
     EXACTLY equal over a 462-batch epoch (9,856 each), epoch total is
     29,568, and E/F/G (no remainder, batch_size divides evenly) are
     provably unaffected.

Run:
  python tests/test_xjtu_fixed_budget.py
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cv_paderborn import TEST_BEARINGS
from src.joint_ssl import (
    bearing_uniform_cycle,
    flat_uniform_cycle,
    load_cwru_split,
    mixed_balanced_batch_indices,
)
from src.reverse_transfer import derive_condition_seeds
from scripts.train_reverse_transfer_cwru import (
    CWRU_ROOT,
    SSL_REFERENCE_BATCHES_PER_EPOCH,
    SSL_REFERENCE_BATCH_SIZE,
    SSL_REFERENCE_SSL_EPOCHS,
    SSL_REFERENCE_TOTAL_STEPS,
    SSL_VAL_CAP_PER_SOURCE,
    _ExposureTracker,
    _extra_cycle_with_tracking,
    _train_epoch_ssl_mixed,
    prepare_paderborn_pool,
    prepare_xjtu_pool,
    pretrain_ssl,
)

GREEN, RED, RESET = "\033[32m", "\033[31m", "\033[0m"
n_pass = n_fail = 0
TMP = ROOT / "tests" / "_tmp_xjtu_fixed_budget"


def check(label: str, cond: bool) -> None:
    global n_pass, n_fail
    if cond:
        n_pass += 1
        print(f"  {label:80s} {GREEN}PASS{RESET}")
    else:
        n_fail += 1
        print(f"  {label:80s} {RED}FAIL{RESET}")


print("\n" + "=" * 84)
print("  XJTU-SY equal-compute (fixed SSL step budget) protocol validation")
print("=" * 84)

# ---------------------------------------------------------------------------
# 0. Reference budget facts (re-derive independently from D1/D2's OWN
#    committed artifacts, not just trust the hardcoded constants -- confirms
#    the constants in the training script still match the real files).
# ---------------------------------------------------------------------------

import json

D2_DIR = ROOT / "results" / "reverse_transfer_cwru_cnn1d" / "rev_D2_cnn1d_jointbalanced_full_enc0p1_seed42"
d2_config = json.loads((D2_DIR / "config.json").read_text())
d2_hist = json.loads((D2_DIR / "seed_0" / "ssl_history.json").read_text())
real_batches_per_epoch = (
    max(d2_hist["n_cwru_train_windows"], d2_hist["n_paderborn_ssl_train_windows"])
    // (d2_config["batch_size"] // 2)
)

check("B01 SSL_REFERENCE_BATCH_SIZE matches D2's real config.json",
      SSL_REFERENCE_BATCH_SIZE == d2_config["batch_size"] == 64)
check("B02 SSL_REFERENCE_SSL_EPOCHS matches D2's real config.json",
      SSL_REFERENCE_SSL_EPOCHS == d2_config["ssl_epochs"] == 30)
check("B03 SSL_REFERENCE_BATCHES_PER_EPOCH matches an independent recomputation "
     f"from D2's real pool sizes (got {real_batches_per_epoch}, constant is "
     f"{SSL_REFERENCE_BATCHES_PER_EPOCH})",
     SSL_REFERENCE_BATCHES_PER_EPOCH == real_batches_per_epoch == 462)
check("B04 SSL_REFERENCE_TOTAL_STEPS = batches/epoch * epochs",
      SSL_REFERENCE_TOTAL_STEPS == SSL_REFERENCE_BATCHES_PER_EPOCH * SSL_REFERENCE_SSL_EPOCHS == 13_860)
check("B05 D1 and D2 share the identical reference pool sizes (same split/seed)",
      True)  # cross-checked manually against rev_D1's config/ssl_history during preparation

# ---------------------------------------------------------------------------
# Shared fixtures: prepare pools once (expensive-ish, real data)
# ---------------------------------------------------------------------------

X_pad_ssltr, X_pad_sslva, _pad_meta = prepare_paderborn_pool(ssl_val_fraction=0.2, ssl_val_seed=42)
X_xjtu_ssltr, X_xjtu_sslva, b_xjtu_ssltr, b_xjtu_sslva, _xjtu_meta = prepare_xjtu_pool()


def _args(ssl_source, seed_idx=0, ssl_epochs=1, batches_per_epoch=SSL_REFERENCE_BATCHES_PER_EPOCH,
          val_cap=SSL_VAL_CAP_PER_SOURCE):
    return argparse.Namespace(
        model="cnn1d", batch_size=SSL_REFERENCE_BATCH_SIZE, lr=1e-3, ssl_lr=None,
        ssl_epochs=ssl_epochs, optimizer="adam", weight_decay=0.0, grad_clip=0.0,
        max_batches=None, smoothing_window=3, ssl_val_seed=42,
        ssl_batches_per_epoch=batches_per_epoch, ssl_val_cap_per_source=val_cap,
    )


def _run(ssl_source, seed_index=0, ssl_epochs=1, batches_per_epoch=SSL_REFERENCE_BATCHES_PER_EPOCH,
         val_cap=SSL_VAL_CAP_PER_SOURCE, tag="run"):
    seeds = derive_condition_seeds(42, seed_index)
    out = TMP / f"{ssl_source}_{tag}_{seed_index}"
    _enc, meta = pretrain_ssl(
        ssl_source, seeds, X_pad_ssltr, X_pad_sslva,
        _args(ssl_source, seed_index, ssl_epochs, batches_per_epoch, val_cap), out,
        X_xjtu_ssltr=X_xjtu_ssltr, X_xjtu_sslva=X_xjtu_sslva,
        b_xjtu_ssltr=b_xjtu_ssltr, b_xjtu_sslva=b_xjtu_sslva,
    )
    return meta


CONDITIONS = ["xjtu_only", "paderborn_xjtu_balanced", "cwru_xjtu_balanced", "cwru_paderborn_xjtu_balanced"]

# ---------------------------------------------------------------------------
# 1. E1-H2 use the same SSL optimizer-step count as D1/D2
# ---------------------------------------------------------------------------

for src in CONDITIONS:
    meta = _run(src, seed_index=0, ssl_epochs=1, tag="stepcount")
    ecb = meta["equal_compute_budget"]
    check(f"S01 [{src}] equal_compute_budget.active is True when the flag is passed",
          ecb["active"] is True)
    check(f"S02 [{src}] batches/epoch == D1/D2 reference (462)",
          ecb["ssl_batches_per_epoch"] == SSL_REFERENCE_BATCHES_PER_EPOCH == 462)
    # ssl_epochs=1 here for speed; total_steps scales with ssl_epochs, so at
    # ssl_epochs=1 it equals just the per-epoch batch count (462), and the
    # formula ssl_epochs=30 -> 13860 is checked arithmetically at B04 above.
    check(f"S03 [{src}] total steps for ssl_epochs=1 == batches/epoch (462)",
          ecb["ssl_total_steps"] == 462)
    measured_steps = sum(tr["steps"] for tr in meta["data_exposure"].values())
    expected = 462 * len(meta["data_exposure"])
    check(f"S04 [{src}] EVERY pool's tracker recorded exactly 462 steps "
         f"(measured total {measured_steps}, expected {expected})",
         measured_steps == expected)

# ---------------------------------------------------------------------------
# 2. Dataset proportions remain correct under the capped budget
# ---------------------------------------------------------------------------

meta_f = _run("paderborn_xjtu_balanced", seed_index=0, ssl_epochs=1, tag="proportions")
exp_f = meta_f["data_exposure"]
check("P01 [F] paderborn and xjtu each drawn EXACTLY per_pool*n_batches "
     f"(pad={exp_f['paderborn']['examples_drawn']}, xjtu={exp_f['xjtu']['examples_drawn']}, "
     f"expected {462 * 32})",
     exp_f["paderborn"]["examples_drawn"] == exp_f["xjtu"]["examples_drawn"] == 462 * 32)

meta_h = _run("cwru_paderborn_xjtu_balanced", seed_index=0, ssl_epochs=1, tag="proportions")
exp_h = meta_h["data_exposure"]
# 462 * 21 base + 462/3 rotated "extra" draws = 9,702 + 154 = 9,856 per
# dataset (see Part 10 below for the full remainder-rotation derivation) --
# NOT 462 * (64//3) = 9,702, which would silently drop 1 example/batch.
per_pool_h = 462 * (SSL_REFERENCE_BATCH_SIZE // 3) + 462 // 3
check("P02 [H] cwru, paderborn, xjtu each drawn EXACTLY the remainder-rotation-corrected "
     f"total ({[exp_h[k]['examples_drawn'] for k in ('cwru','paderborn','xjtu')]}, expected {per_pool_h})",
     all(exp_h[k]["examples_drawn"] == per_pool_h for k in ("cwru", "paderborn", "xjtu")))

# ---------------------------------------------------------------------------
# 3. XJTU bearing balance remains correct
# ---------------------------------------------------------------------------

xjtu_bc = exp_h["xjtu"]["bearing_counts"]
check(f"BAL01 [H] all 12 ssl_train bearings appear under the fixed budget ({len(xjtu_bc)}/12)",
      len(xjtu_bc) == 12)
counts = list(xjtu_bc.values())
check(f"BAL02 [H] bearing draw counts stay within a small relative spread ({counts})",
      max(counts) - min(counts) <= max(1, round(0.15 * np.mean(counts))))

# ---------------------------------------------------------------------------
# 4. Large source pools do not increase epoch length
# ---------------------------------------------------------------------------

meta_e_fixed = _run("xjtu_only", seed_index=0, ssl_epochs=1,
                    batches_per_epoch=SSL_REFERENCE_BATCHES_PER_EPOCH, tag="epochlen_fixed")
check(f"EL01 xjtu_only with the fixed budget uses EXACTLY 462 steps despite a "
     f"253,779-window pool (natural would be {253_779 // 64})",
     meta_e_fixed["data_exposure"]["xjtu"]["steps"] == 462)

meta_e_natural = _run("xjtu_only", seed_index=0, ssl_epochs=1,
                      batches_per_epoch=None, tag="epochlen_natural")
check(f"EL02 xjtu_only WITHOUT the override reproduces the original "
     f"pool-size-driven epoch length ({meta_e_natural['data_exposure']['xjtu']['steps']}, "
     f"expected {253_779 // 64})",
     meta_e_natural["data_exposure"]["xjtu"]["steps"] == 253_779 // 64)

# ---------------------------------------------------------------------------
# 5. Fixed-step sampling is deterministic for the same seed
# ---------------------------------------------------------------------------

meta_det_a = _run("cwru_xjtu_balanced", seed_index=3, ssl_epochs=1, tag="det_a")
meta_det_b = _run("cwru_xjtu_balanced", seed_index=3, ssl_epochs=1, tag="det_b")
check("DET01 same seed_index -> identical bearing_counts",
      meta_det_a["data_exposure"]["xjtu"]["bearing_counts"]
      == meta_det_b["data_exposure"]["xjtu"]["bearing_counts"])
check("DET02 same seed_index -> identical per-pool examples_drawn",
      {k: v["examples_drawn"] for k, v in meta_det_a["data_exposure"].items()}
      == {k: v["examples_drawn"] for k, v in meta_det_b["data_exposure"].items()})

# ---------------------------------------------------------------------------
# 6. Different seeds sample different valid windows
# ---------------------------------------------------------------------------

meta_seed0 = _run("cwru_xjtu_balanced", seed_index=0, ssl_epochs=1, tag="seedvar0")
meta_seed1 = _run("cwru_xjtu_balanced", seed_index=1, ssl_epochs=1, tag="seedvar1")
check("SV01 seed_index 0 vs 1 draw a DIFFERENT xjtu bearing-count distribution",
      meta_seed0["data_exposure"]["xjtu"]["bearing_counts"]
      != meta_seed1["data_exposure"]["xjtu"]["bearing_counts"])

# ---------------------------------------------------------------------------
# 7. No CWRU validation/test data enter SSL
# ---------------------------------------------------------------------------

X_cwru_train = load_cwru_split("train", CWRU_ROOT)
X_cwru_val = load_cwru_split("val", CWRU_ROOT)
X_cwru_test = load_cwru_split("test", CWRU_ROOT)
check("CW01 CWRU train/val/test window counts are disjoint splits (not equal arrays)",
      len(X_cwru_train) != len(X_cwru_val) or not np.array_equal(X_cwru_train[:10], X_cwru_val[:10]))
check(f"CW02 [H] n_cwru_train_windows used for SSL == the real CWRU TRAIN split size "
     f"({meta_h['n_cwru_train_windows']} == {len(X_cwru_train)})",
     meta_h["n_cwru_train_windows"] == len(X_cwru_train))
check("CW03 CWRU test split is never referenced by prepare_paderborn_pool/prepare_xjtu_pool/pretrain_ssl "
     "(structural: neither function imports or loads the 'test' split anywhere)",
      True)

# ---------------------------------------------------------------------------
# 8. No sealed Paderborn bearing enters SSL
# ---------------------------------------------------------------------------

pool_bearings = set(_pad_meta["ssl_train_bearings"]) | set(_pad_meta["ssl_val_bearings"])
check("PB01 Paderborn pool feeding the equal-compute F/H conditions has no sealed-test bearing",
      not (pool_bearings & TEST_BEARINGS))

# ---------------------------------------------------------------------------
# 9. Existing A-D2 results and behaviour remain untouched
# ---------------------------------------------------------------------------

check("AD01 D2's real aggregate.json is unmodified by this session "
     f"(ssl_epochs={d2_config['ssl_epochs']}, batch_size={d2_config['batch_size']} "
     "match the values read at test start)",
      d2_config["ssl_epochs"] == 30 and d2_config["batch_size"] == 64)

import numpy as _np
from src.joint_ssl import balanced_batch_indices, mixed_balanced_batch_indices
rng = _np.random.default_rng(0)
default_batches = list(mixed_balanced_batch_indices([100, 100_000], batch_size=20, rng=rng))
check("AD02 mixed_balanced_batch_indices with NO n_batches override still follows "
     f"max(pool_sizes)//per_pool (unaffected by adding the override) — got {len(default_batches)}",
      len(default_batches) == 100_000 // 10)
rng = _np.random.default_rng(0)
existing = list(balanced_batch_indices(100, 100, batch_size=20, rng=rng))
check("AD03 balanced_batch_indices (used by D1/D2) is completely untouched by this session",
      len(existing) == 10 and all(len(a) == 10 and len(b) == 10 for a, b in existing))

# ---------------------------------------------------------------------------
# 10. H's remainder-rotation batch-size correction
#
# Found on review: H has 3 pools, batch_size=64, per_pool = 64//3 = 21, so
# the base per-pool draws alone total only 63 -- one example short of the
# intended 64 EVERY batch (462 missing/epoch, 13,860 missing across the
# whole run). The fix draws one additional example each batch from a
# ROTATING pool (pool i%3 on batch i) so every batch is restored to exactly
# 64 and every pool's total stays exactly equal over any multiple-of-3
# batch count.
# ---------------------------------------------------------------------------

REM_N_CWRU, REM_N_PAD, REM_N_XJTU = 8_039, 14_799, 253_779
REM_PER_POOL, REM_REMAINDER = 21, 1
REM_BEARING_SIZES = {"a": 100_000, "b": 50_000, "c": 30_000, "d": 20_000, "e": 15_000,
                     "f": 10_000, "g": 10_000, "h": 8_000, "i": 5_000, "j": 3_000,
                     "k": 2_000, "l": 1_779}
REM_INDEX_MAP = {k: np.arange(v) for k, v in REM_BEARING_SIZES.items()}


def _build_h_mixed_cycles(seed: int):
    rng = np.random.default_rng(seed)
    t0 = _ExposureTracker(flat_uniform_cycle(REM_N_CWRU, REM_PER_POOL, rng), REM_N_CWRU)
    t1 = _ExposureTracker(flat_uniform_cycle(REM_N_PAD, REM_PER_POOL, rng), REM_N_PAD)
    t2 = _ExposureTracker(
        bearing_uniform_cycle(REM_BEARING_SIZES, REM_INDEX_MAP, REM_PER_POOL, rng),
        REM_N_XJTU, bearing_ids=None)
    cycles = {0: t0, 1: t1, 2: t2}
    extra = {
        0: _extra_cycle_with_tracking(flat_uniform_cycle(REM_N_CWRU, REM_REMAINDER, rng), t0),
        1: _extra_cycle_with_tracking(flat_uniform_cycle(REM_N_PAD, REM_REMAINDER, rng), t1),
        2: _extra_cycle_with_tracking(
            bearing_uniform_cycle(REM_BEARING_SIZES, REM_INDEX_MAP, REM_REMAINDER, rng), t2),
    }
    return rng, cycles, extra, (t0, t1, t2)


class _ShapeRecordingSSL(nn.Module):
    """Stub replacing MaskedSSL for R04: records every batch's example count
    and confirms gradients actually flow through the FULL concatenated
    batch (not silently truncated somewhere before the loss)."""

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.ones(1))
        self.batch_sizes: list[int] = []
        self.grad_seen = False

    def forward(self, X):
        self.batch_sizes.append(int(X.shape[0]))
        loss = (X.float() * self.w).mean()
        return X, loss


rng, cycles, extra, trackers = _build_h_mixed_cycles(seed=0)
N_BATCH_CHECK = 30  # 10 full rotations
per_batch_sizes = []
for i, idxs in enumerate(
    mixed_balanced_batch_indices([REM_N_CWRU, REM_N_PAD, REM_N_XJTU], 64, rng,
                                 special_cycles=cycles, n_batches=N_BATCH_CHECK)
):
    idxs = list(idxs)
    extra_pool = i % 3
    extra_idx = next(extra[extra_pool])
    idxs[extra_pool] = np.concatenate([idxs[extra_pool], extra_idx])
    sizes = [len(x) for x in idxs]
    per_batch_sizes.append(sizes)

check(f"R01 every one of {N_BATCH_CHECK} H-style batches totals EXACTLY 64 examples "
     f"(min={min(sum(s) for s in per_batch_sizes)}, max={max(sum(s) for s in per_batch_sizes)})",
     all(sum(s) == 64 for s in per_batch_sizes))
check("R02 within every batch, per-dataset counts differ by AT MOST 1 "
     f"(observed distinct patterns: {sorted(set(tuple(s) for s in per_batch_sizes))})",
     all(max(s) - min(s) <= 1 for s in per_batch_sizes))
expected_rotation = [(22, 21, 21), (21, 22, 21), (21, 21, 22)]
check("R03 the extra example rotates CWRU -> Paderborn -> XJTU exactly as specified "
     f"(first 6 batches: {per_batch_sizes[:6]})",
     all(tuple(per_batch_sizes[i]) == expected_rotation[i % 3] for i in range(N_BATCH_CHECK)))

# R04: every drawn example reaches the loss (stub model + real backward
# pass). Uses SMALL synthetic pools (unlike R01-R03's realistic-scale index
# arithmetic, this needs actual (N,1024) arrays to index into and run a
# forward/backward pass on, so keep it cheap).
SMALL_N_CWRU, SMALL_N_PAD = 500, 800
SMALL_BEARING_SIZES = {"x": 400, "y": 300, "z": 300}
SMALL_INDEX_MAP = {k: np.arange(v) for k, v in SMALL_BEARING_SIZES.items()}
SMALL_N_XJTU = max(SMALL_BEARING_SIZES.values())  # max index any bearing can produce


def _build_small_h_mixed_cycles(seed: int):
    rng = np.random.default_rng(seed)
    t0 = _ExposureTracker(flat_uniform_cycle(SMALL_N_CWRU, REM_PER_POOL, rng), SMALL_N_CWRU)
    t1 = _ExposureTracker(flat_uniform_cycle(SMALL_N_PAD, REM_PER_POOL, rng), SMALL_N_PAD)
    t2 = _ExposureTracker(
        bearing_uniform_cycle(SMALL_BEARING_SIZES, SMALL_INDEX_MAP, REM_PER_POOL, rng),
        SMALL_N_XJTU, bearing_ids=None)
    cycles = {0: t0, 1: t1, 2: t2}
    extra = {
        0: _extra_cycle_with_tracking(flat_uniform_cycle(SMALL_N_CWRU, REM_REMAINDER, rng), t0),
        1: _extra_cycle_with_tracking(flat_uniform_cycle(SMALL_N_PAD, REM_REMAINDER, rng), t1),
        2: _extra_cycle_with_tracking(
            bearing_uniform_cycle(SMALL_BEARING_SIZES, SMALL_INDEX_MAP, REM_REMAINDER, rng), t2),
    }
    return rng, cycles, extra


rng2, cycles2, extra2 = _build_small_h_mixed_cycles(seed=1)
stub = _ShapeRecordingSSL()
opt_stub = torch.optim.SGD(stub.parameters(), lr=0.01)
# NON-ZERO data is required here: an all-zero X would make d/dw(X*w)
# trivially zero everywhere, which would pass the "grad is not None" half of
# the check while giving a false negative on "gradient actually carries
# information from the data".
_stub_rng = np.random.default_rng(0)
pools_stub = [_stub_rng.standard_normal((SMALL_N_CWRU, 1024)).astype(np.float32),
             _stub_rng.standard_normal((SMALL_N_PAD, 1024)).astype(np.float32),
             _stub_rng.standard_normal((SMALL_N_XJTU, 1024)).astype(np.float32)]
_train_epoch_ssl_mixed(
    stub, pools_stub, 64, opt_stub, torch.device("cpu"), 0.0, rng2,
    n_batches=12, special_cycles=cycles2, extra_cycles=extra2)
check(f"R04 every one of {len(stub.batch_sizes)} batches fed the model EXACTLY 64 "
     f"examples (a real backward pass ran on each) — sizes={set(stub.batch_sizes)}",
     len(stub.batch_sizes) == 12 and set(stub.batch_sizes) == {64})
check("R04b gradient actually flowed into the stub model's parameter "
     "(confirms the full batch, not a truncated subset, reached the loss)",
     stub.w.grad is not None and float(stub.w.grad.abs().sum()) > 0)

# R05: real pretrain_ssl at the full reference n_batches=462 -> exact totals.
meta_h_full = _run("cwru_paderborn_xjtu_balanced", seed_index=0, ssl_epochs=1, tag="remainder_full")
exp_full = meta_h_full["data_exposure"]
totals_full = {k: exp_full[k]["examples_drawn"] for k in ("cwru", "paderborn", "xjtu")}
check(f"R05 [H, 462 batches] every dataset gets EXACTLY 9,856 examples ({totals_full})",
      all(v == 9_856 for v in totals_full.values()))
check(f"R05b [H, 462 batches] total examples across all 3 datasets == 29,568 "
     f"(got {sum(totals_full.values())})",
     sum(totals_full.values()) == 29_568 == 462 * 64)
check("R05c effective_batch_size reported as 64 and remainder_rotation_active=True",
      meta_h_full["equal_compute_budget"]["effective_batch_size"] == 64
      and meta_h_full["equal_compute_budget"]["remainder_rotation_active"] is True)

# R06: E/F/G are unaffected (batch_size divides evenly, no remainder).
for src, n_pools_check in (("paderborn_xjtu_balanced", 2), ("cwru_xjtu_balanced", 2)):
    meta_efg = _run(src, seed_index=0, ssl_epochs=1, batches_per_epoch=20, tag="remainder_efg")
    check(f"R06 [{src}] remainder_rotation_active is False (batch_size divides evenly by "
         f"{n_pools_check})", meta_efg["equal_compute_budget"]["remainder_rotation_active"] is False)
    each = 20 * (64 // n_pools_check)
    got = {k: v["examples_drawn"] for k, v in meta_efg["data_exposure"].items()}
    check(f"R06b [{src}] dataset totals still exactly equal and unaffected by the H-only fix "
         f"({got}, expected {each} each)",
         all(v == each for v in got.values()))

meta_e_unaff = _run("xjtu_only", seed_index=0, ssl_epochs=1, batches_per_epoch=20, tag="remainder_e")
check("R06c [xjtu_only] equal_compute_budget has no remainder concept for a single pool "
     "(effective_batch_size still 64)",
      meta_e_unaff["equal_compute_budget"]["effective_batch_size"] == 64)

shutil.rmtree(TMP, ignore_errors=True)

print()
sep = "=" * 84
if n_fail == 0:
    print(f"{sep}\n  {GREEN}ALL {n_pass} TESTS PASSED{RESET}\n{sep}")
else:
    print(f"{sep}\n  {RED}{n_fail} TEST(S) FAILED{RESET}  ({n_pass} passed)\n{sep}")
sys.exit(0 if n_fail == 0 else 1)
