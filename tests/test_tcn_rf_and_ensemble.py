"""
test_tcn_rf_and_ensemble.py — unit tests for the large-RF TCN variant and the
complementary-error strategy functions.

Run:
  python tests/test_tcn_rf_and_ensemble.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models import MODEL_REGISTRY, build_model
from src.models.tcn1d import TCN1D, TCN1DLargeRF, receptive_field
from src.trainer import setup_partial_finetune
from scripts.analyze_complementary_errors import (
    hard_majority,
    oracle_any_correct,
    oracle_per_bearing,
)

GREEN, RED, RESET = "\033[32m", "\033[31m", "\033[0m"
n_pass = n_fail = 0


def check(label: str, cond: bool) -> None:
    global n_pass, n_fail
    if cond:
        n_pass += 1
        print(f"  {label:62s} {GREEN}PASS{RESET}")
    else:
        n_fail += 1
        print(f"  {label:62s} {RED}FAIL{RESET}")


print("\n" + "=" * 76)
print("  Large-RF TCN + ensemble-strategy unit tests")
print("=" * 76)

# ---------------------------------------------------------------------------
# receptive_field
# ---------------------------------------------------------------------------

check("T01 RF of standard TCN (k=3, dil 1,2,4,8) = 61",
      receptive_field(3, (1, 2, 4, 8)) == 61)
check("T02 RF of large TCN (k=3, dil 1..128) = 1021",
      receptive_field(3, (1, 2, 4, 8, 16, 32, 64, 128)) == 1021)

# ---------------------------------------------------------------------------
# TCN1DLargeRF
# ---------------------------------------------------------------------------

torch.manual_seed(0)
std   = TCN1D(num_classes=3)
large = TCN1DLargeRF(num_classes=3)
n_std   = sum(p.numel() for p in std.parameters())
n_large = sum(p.numel() for p in large.parameters())
ratio = n_large / n_std
check(f"T03 param counts: std={n_std:,} large={n_large:,} within ±10%",
      0.90 <= ratio <= 1.10)

check("T04 tcn_rf1021 registered in MODEL_REGISTRY",
      "tcn_rf1021" in MODEL_REGISTRY
      and isinstance(build_model("tcn_rf1021", num_classes=3), TCN1DLargeRF))

x = torch.randn(2, 1, 1024)
check("T05 forward → (2, 3) logits", large(x).shape == (2, 3))
check("T06 encode → (2, 128) embedding", large.encode(x).shape == (2, 128))
check("T07 forward_sequence → (2, 128, 1024)",
      large.forward_sequence(x).shape == (2, 128, 1024))

n_train = setup_partial_finetune(build_model("tcn_rf1021", num_classes=3),
                                 "tcn_rf1021")
check("T08 setup_partial_finetune supports tcn_rf1021 (second half trainable)",
      0 < n_train < n_large)

sd = large.state_dict()
m2 = TCN1DLargeRF(num_classes=3)
m2.load_state_dict(sd, strict=True)
check("T09 state_dict round-trip with strict=True", True)

# ---------------------------------------------------------------------------
# Ensemble strategy functions
# ---------------------------------------------------------------------------

#            idx:   0  1  2  3  4  5  6
y      = np.array([ 0, 0, 1, 1, 2, 2, 0])
groups = np.array(["b0", "b0", "b1", "b1", "b2", "b2", "b2"])
preds = {
    "A": np.array([0, 0, 2, 2, 2, 2, 1]),   # solves b0 fully, b2 2/3
    "B": np.array([1, 1, 1, 1, 0, 0, 2]),   # solves b1 fully
    "C": np.array([2, 0, 1, 2, 1, 1, 1]),   # partial overlap
}
# votes per idx: 0:(0,1,2) tie  1:(0,1,0)→0  2:(2,1,1)→1  3:(2,1,2)→2
#                4:(2,0,1) tie  5:(2,0,1) tie  6:(1,2,1)→1

mv = hard_majority(preds, fallback="A")
check("T10 majority vote picks strict majorities",
      mv[1] == 0 and mv[2] == 1 and mv[3] == 2 and mv[6] == 1)
check("T11 three-way ties fall back to baseline regime",
      mv[0] == preds["A"][0] and mv[4] == preds["A"][4]
      and mv[5] == preds["A"][5])

ow = oracle_any_correct(preds, y, fallback="A")
check("T12 window oracle correct wherever any regime is correct",
      (ow[:6] == y[:6]).all())
check("T13 window oracle falls back where all regimes are wrong",
      ow[6] == preds["A"][6] and ow[6] != y[6])

ob, choice = oracle_per_bearing(preds, y, groups)
check("T14 bearing oracle picks best regime per bearing",
      choice["b0"] == "A" and choice["b1"] == "B" and choice["b2"] == "A")
check("T15 bearing oracle predictions come from the chosen regime",
      (ob[groups == "b1"] == preds["B"][groups == "b1"]).all())

acc_oracle = (ob == y).mean()
acc_single = max((preds[n] == y).mean() for n in preds)
check("T16 bearing oracle ≥ best single regime", acc_oracle >= acc_single)

print()
sep = "=" * 76
if n_fail == 0:
    print(f"{sep}\n  {GREEN}ALL {n_pass} TESTS PASSED{RESET}\n{sep}")
else:
    print(f"{sep}\n  {RED}{n_fail} TEST(S) FAILED{RESET}  ({n_pass} passed)\n{sep}")
sys.exit(0 if n_fail == 0 else 1)
