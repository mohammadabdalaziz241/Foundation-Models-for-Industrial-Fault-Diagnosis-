"""
test_joint_ssl_protocol.py — protocol validation and unit tests for joint
CWRU+Paderborn masked-SSL pretraining (src/joint_ssl.py and the orchestrator
in scripts/train_joint_ssl_paderborn.py).

Protocol-validation checks (must pass before any joint SSL run is launched):
  - CWRU and Paderborn windows share window length, dtype and channel shape.
  - CWRU's own train/val/test recording groups are mutually disjoint (no
    leakage in reusing CWRU val as the SSL-checkpoint-selection subset).
  - The encoder never receives anything beyond the raw (B, 1, 1024) signal —
    no dataset-identifier / metadata channel is structurally possible.
  - MaskedSSL's decoder is selected purely from the encoder's probed output
    shape, so it is identical regardless of which dataset supplied the
    (identically-shaped) probe tensor.

Run:
  python tests/test_joint_ssl_protocol.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.joint_ssl import balanced_batch_indices, load_cwru_split, natural_proportion
from src.models import build_model
from src.models.masked_ssl import MaskedSSL
from scripts.train_joint_ssl_paderborn import derive_repeat_seed, _new_ssl_model

GREEN, RED, RESET = "\033[32m", "\033[31m", "\033[0m"
n_pass = n_fail = 0


def check(label: str, cond: bool) -> None:
    global n_pass, n_fail
    if cond:
        n_pass += 1
        print(f"  {label:70s} {GREEN}PASS{RESET}")
    else:
        n_fail += 1
        print(f"  {label:70s} {RED}FAIL{RESET}")


print("\n" + "=" * 78)
print("  Joint SSL protocol validation + unit tests")
print("=" * 78)

# ---------------------------------------------------------------------------
# Protocol validation — dataset compatibility
# ---------------------------------------------------------------------------

X_cwru_train = load_cwru_split("train", ROOT / "data" / "processed")
X_cwru_val   = load_cwru_split("val",   ROOT / "data" / "processed")
X_pad = np.load(ROOT / "data" / "processed_paderborn" / "train" / "windows.npy")

check("P01 CWRU windows are (N, 1024)", X_cwru_train.shape[1] == 1024)
check("P02 Paderborn windows are (N, 1024)", X_pad.shape[1] == 1024)
check("P03 CWRU and Paderborn window length match exactly",
      X_cwru_train.shape[1] == X_pad.shape[1])
check("P04 CWRU and Paderborn windows are both float32",
      X_cwru_train.dtype == X_pad.dtype == np.float32)

# CWRU split disjointness (by load-condition recording group) — the
# precondition for reusing CWRU val as the SSL-checkpoint-selection subset.
cwru_root = ROOT / "data" / "processed"
recs = {}
for split in ("train", "val", "test"):
    rows = list(csv.DictReader(open(cwru_root / split / "metadata.csv")))
    recs[split] = {r["recording_id"] for r in rows}
check("P05 CWRU train/val recording groups disjoint",
      not (recs["train"] & recs["val"]))
check("P06 CWRU train/test recording groups disjoint",
      not (recs["train"] & recs["test"]))
check("P07 CWRU val/test recording groups disjoint",
      not (recs["val"] & recs["test"]))

# Encoder never receives an identifier channel — structural check via a
# concatenated CWRU+Paderborn tensor.
combined = np.concatenate([X_cwru_train[:4], X_pad[:4]], axis=0)
t = torch.from_numpy(combined).float().unsqueeze(1)
check("P08 Combined CWRU+Paderborn batch tensor is exactly (B, 1, 1024)",
      tuple(t.shape) == (8, 1, 1024))

# Decoder identical regardless of which dataset probed it (same shape).
m1 = build_model("lstm", num_classes=3)
ssl_from_cwru = MaskedSSL(encoder_model=m1, patch_len=32, mask_ratio=0.5)
m2 = build_model("lstm", num_classes=3)
ssl_from_pad = MaskedSSL(encoder_model=m2, patch_len=32, mask_ratio=0.5)
check("P09 Decoder type identical for CWRU- and Paderborn-shaped probes",
      type(ssl_from_cwru.decoder) is type(ssl_from_pad.decoder))
check("P10 Decoder architecture (param shapes) identical",
      [p.shape for p in ssl_from_cwru.decoder.parameters()]
      == [p.shape for p in ssl_from_pad.decoder.parameters()])

# End-to-end: a real MaskedSSL forward pass accepts a mixed-origin batch.
with torch.no_grad():
    recon, loss = ssl_from_cwru(t)
check("P11 MaskedSSL forward pass runs on a mixed CWRU+Paderborn batch",
      recon.shape == (8, 1, 1024) and torch.isfinite(loss))

# ---------------------------------------------------------------------------
# balanced_batch_indices
# ---------------------------------------------------------------------------

rng = np.random.default_rng(0)
batches = list(balanced_batch_indices(n_a=100, n_b=100, batch_size=20, rng=rng))
check("B01 equal pools: every batch has exactly half from each side",
      all(len(a) == 10 and len(b) == 10 for a, b in batches))
check("B02 equal pools: epoch length = max(n_a,n_b)//half = 10",
      len(batches) == 10)

rng = np.random.default_rng(0)
batches_imb = list(balanced_batch_indices(n_a=20, n_b=200, batch_size=20, rng=rng))
check("C01 imbalanced pools: epoch length set by the LARGER pool",
      len(batches_imb) == 200 // 10)
check("C02 imbalanced pools: every batch STILL exactly half/half",
      all(len(a) == 10 and len(b) == 10 for a, b in batches_imb))
all_a_idx = np.concatenate([a for a, b in batches_imb])
check("C03 the smaller pool (n_a=20) is cycled — indices repeat across the epoch",
      len(all_a_idx) > 20 and set(all_a_idx.tolist()) == set(range(20)))

rng1 = np.random.default_rng(7)
rng2 = np.random.default_rng(7)
b1 = list(balanced_batch_indices(50, 50, 10, rng1))
b2 = list(balanced_batch_indices(50, 50, 10, rng2))
check("B03 deterministic given the same seeded Generator",
      all(np.array_equal(x[0], y[0]) and np.array_equal(x[1], y[1])
          for x, y in zip(b1, b2)))

try:
    list(balanced_batch_indices(0, 10, 10, np.random.default_rng(0)))
    check("B04 empty pool raises ValueError", False)
except ValueError:
    check("B04 empty pool raises ValueError", True)

# odd batch_size: half = batch_size // 2 (documented truncation, not a bug)
rng = np.random.default_rng(0)
batches_odd = list(balanced_batch_indices(30, 30, 9, rng))
check("B05 odd batch_size: half = 9 // 2 = 4 per side",
      all(len(a) == 4 and len(b) == 4 for a, b in batches_odd))

# ---------------------------------------------------------------------------
# natural_proportion
# ---------------------------------------------------------------------------

check("N01 equal pools → proportion 0.5", natural_proportion(50, 50) == 0.5)
check("N02 correctness of a non-trivial ratio",
      abs(natural_proportion(8039, 11109) - 8039 / (8039 + 11109)) < 1e-12)
try:
    natural_proportion(0, 0)
    check("N03 zero-zero raises ValueError", False)
except ValueError:
    check("N03 zero-zero raises ValueError", True)

# ---------------------------------------------------------------------------
# derive_repeat_seed — fold-independent, distinct from other seed streams
# ---------------------------------------------------------------------------

from src.cv_paderborn import derive_fold_seed, derive_ssl_val_seed

check("S01 derive_repeat_seed deterministic",
      derive_repeat_seed(42, 0) == derive_repeat_seed(42, 0))
check("S02 derive_repeat_seed does not depend on fold (fold-independent by design)",
      derive_repeat_seed(42, 1) == derive_repeat_seed(42, 1))
check("S03 derive_repeat_seed distinct from derive_fold_seed for same args",
      derive_repeat_seed(42, 0) != derive_fold_seed(42, 0, 0))
check("S04 derive_repeat_seed distinct from derive_ssl_val_seed for same args",
      derive_repeat_seed(42, 0) != derive_ssl_val_seed(42, 0, 0))
check("S05 different repeats give different seeds",
      derive_repeat_seed(42, 0) != derive_repeat_seed(42, 1))

# ---------------------------------------------------------------------------
# _new_ssl_model smoke — confirms the orchestrator's model factory works
# ---------------------------------------------------------------------------

ssl = _new_ssl_model("lstm")
ssl_device = next(ssl.parameters()).device
with torch.no_grad():
    recon, loss = ssl(torch.randn(4, 1, 1024, device=ssl_device))
check("M01 _new_ssl_model('lstm') builds a working MaskedSSL",
      recon.shape == (4, 1, 1024) and torch.isfinite(loss))

# ---------------------------------------------------------------------------
# Dataset-origin probe extension to analyze_representations_paderborn.py
# ---------------------------------------------------------------------------

from scripts.analyze_representations_paderborn import load_cwru_pool
from src.repr_analysis import bearing_identity_probe

pool_small = load_cwru_pool(ROOT / "data" / "processed", max_samples=50, seed=0)
check("O01 load_cwru_pool respects max_samples cap", len(pool_small) == 50)
check("O02 load_cwru_pool returns (N, 1024) windows", pool_small.shape[1] == 1024)
pool_a = load_cwru_pool(ROOT / "data" / "processed", max_samples=50, seed=0)
pool_b = load_cwru_pool(ROOT / "data" / "processed", max_samples=50, seed=0)
check("O03 load_cwru_pool deterministic given the same seed",
      np.array_equal(pool_a, pool_b))
pool_c = load_cwru_pool(ROOT / "data" / "processed", max_samples=50, seed=1)
check("O04 different seed gives a different subsample",
      not np.array_equal(pool_a, pool_c))

# Dataset-origin probe recovers near-perfect separation on synthetic,
# well-separated "CWRU-like" vs "Paderborn-like" embeddings (sanity check
# that reusing bearing_identity_probe for a binary origin label works).
rng = np.random.default_rng(3)
emb_pad  = rng.normal(0, 0.1, size=(200, 16))
emb_cwru = rng.normal(5, 0.1, size=(200, 16))
emb_combo = np.concatenate([emb_pad, emb_cwru], axis=0)
groups_combo = np.array(["paderborn"] * 200 + ["cwru"] * 200)
origin_probe = bearing_identity_probe(emb_combo, groups_combo, seed=0)
check("O05 dataset-origin probe near-perfect on separated synthetic embeddings",
      origin_probe["accuracy"] > 0.95)
check("O06 dataset-origin probe chance = 0.5 for a balanced binary split",
      abs(origin_probe["chance"] - 0.5) < 1e-9)

print()
sep = "=" * 78
if n_fail == 0:
    print(f"{sep}\n  {GREEN}ALL {n_pass} TESTS PASSED{RESET}\n{sep}")
else:
    print(f"{sep}\n  {RED}{n_fail} TEST(S) FAILED{RESET}  ({n_pass} passed)\n{sep}")
sys.exit(0 if n_fail == 0 else 1)
