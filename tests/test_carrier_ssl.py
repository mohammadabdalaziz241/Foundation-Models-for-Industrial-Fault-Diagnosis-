"""Unit tests for F1 carrier-ablation SSL variants (CPU, synthetic + tiny real model).

Run: python tests/test_carrier_ssl.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.carrier_ssl import (VARIANTS, CarrierAblationSSL, epoch_batches,
                             per_bearing_stats)
from src.models import build_model
from src.models.masked_ssl import MaskedSSL

GREEN, RED, RESET = "\033[92m", "\033[91m", "\033[0m"
n_pass = n_fail = 0


def check(label, ok, suffix=""):
    global n_pass, n_fail
    tag = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
    print(f"  T{n_pass + n_fail + 1:02d} {label:62s} {tag}{suffix}")
    n_pass += ok
    n_fail += (not ok)


torch.manual_seed(0)
np.random.seed(0)

model = build_model("lstm", num_classes=3)  # CPU per MaskedSSL device rule
x = torch.randn(4, 1, 1024) * 2.0 + 0.7    # nonzero DC, nonunit scale
bnames = np.array(["A", "B", "C"])
bmean = np.array([0.5, -0.3, 0.1], dtype=np.float32)
bstd = np.array([1.5, 0.7, 1.0], dtype=np.float32)
bidx = torch.tensor([0, 1, 2, 0])

# --- view math per variant --------------------------------------------------
ssl_ctl = CarrierAblationSSL(model, "control")
ei, tg = ssl_ctl._views(x, None)
check("control: input and target are x unchanged",
      torch.equal(ei, x) and torch.equal(tg, x))

ssl_dc = CarrierAblationSSL(model, "dc_removed")
ei, tg = ssl_dc._views(x, None)
check("dc_removed: both views have ~zero per-window mean",
      abs(ei.mean(-1)).max().item() < 1e-5 and torch.equal(ei, tg))

ssl_pw = CarrierAblationSSL(model, "target_pw")
ei, tg = ssl_pw._views(x, None)
check("target_pw: input untouched, target ~zero-mean unit-std",
      torch.equal(ei, x) and abs(tg.mean(-1)).max().item() < 1e-5
      and abs(tg.std(-1) - 1).max().item() < 1e-3)

ssl_pb = CarrierAblationSSL(model, "target_pb", bearing_mean=bmean,
                            bearing_std=bstd)
ei, tg = ssl_pb._views(x, bidx)
expected = (x - torch.tensor(bmean)[bidx].view(-1, 1, 1)) / \
           (torch.tensor(bstd)[bidx].view(-1, 1, 1) + 1e-8)
check("target_pb: target uses per-bearing constants via bearing_idx",
      torch.equal(ei, x) and torch.allclose(tg, expected))

ssl_d = CarrierAblationSSL(model, "dc_plus_pw")
ei, tg = ssl_d._views(x, None)
check("dc_plus_pw: DC-removed input + per-window-standardised target",
      abs(ei.mean(-1)).max().item() < 1e-5
      and abs(tg.std(-1) - 1).max().item() < 1e-3)

# --- control arm is loss-equivalent to historical MaskedSSL -----------------
torch.manual_seed(123)
_, loss_ctl = ssl_ctl(x)
base = MaskedSSL(encoder_model=model)
base.decoder.load_state_dict(ssl_ctl.decoder.state_dict())
torch.manual_seed(123)
_, loss_base = base(x)
check("control forward == historical MaskedSSL forward (same seed/mask)",
      torch.allclose(loss_ctl, loss_base),
      f"  ({loss_ctl.item():.6f} vs {loss_base.item():.6f})")

# --- error handling ---------------------------------------------------------
try:
    CarrierAblationSSL(model, "nope")
    check("unknown variant rejected", False)
except ValueError:
    check("unknown variant rejected", True)
try:
    CarrierAblationSSL(model, "target_pb")
    check("target_pb without stats rejected", False)
except ValueError:
    check("target_pb without stats rejected", True)
try:
    ssl_pb(x)
    check("target_pb forward without bearing_idx rejected", False)
except ValueError:
    check("target_pb forward without bearing_idx rejected", True)

# --- gradients flow to encoder in every variant -----------------------------
ok = True
for v in VARIANTS:
    m = build_model("lstm", num_classes=3)
    s = (CarrierAblationSSL(m, v, bearing_mean=bmean, bearing_std=bstd)
         if v == "target_pb" else CarrierAblationSSL(m, v))
    _, loss = s(x, bidx if v == "target_pb" else None)
    loss.backward()
    g = next(p.grad for p in m.encoder.parameters() if p.grad is not None)
    ok &= bool(torch.isfinite(loss)) and g.abs().sum().item() > 0
check("all 5 variants: finite loss, nonzero encoder gradients", ok)

# --- per-bearing stats ------------------------------------------------------
Xp = np.concatenate([np.full((3, 8), 2.0), np.full((2, 8), -1.0)])
bp = np.array(["b1"] * 3 + ["b0"] * 2)
names, m_, s_ = per_bearing_stats(Xp, bp)
check("per_bearing_stats: sorted names, exact means, zero stds",
      names == ["b0", "b1"] and np.allclose(m_, [-1.0, 2.0])
      and np.allclose(s_, 0.0))

# --- epoch batch generator --------------------------------------------------
rng = np.random.default_rng(42)
batches = list(epoch_batches(100, 7, 16, rng))
allidx = np.concatenate(batches)
check("epoch_batches: exact count and size",
      len(batches) == 7 and all(len(b) == 16 for b in batches))
counts = np.bincount(allidx, minlength=100)
check("epoch_batches: shuffled-cycling coverage (counts differ by <=1)",
      counts.max() - counts.min() <= 1)
r1 = np.concatenate(list(epoch_batches(100, 7, 16, np.random.default_rng(7))))
r2 = np.concatenate(list(epoch_batches(100, 7, 16, np.random.default_rng(7))))
check("epoch_batches: deterministic for same rng seed", np.array_equal(r1, r2))

print()
sep = "=" * 60
if n_fail == 0:
    print(f"{sep}\n  {GREEN}ALL {n_pass} TESTS PASSED{RESET}\n{sep}")
else:
    print(f"{sep}\n  {RED}{n_fail} TEST(S) FAILED{RESET}  ({n_pass} passed)\n{sep}")
sys.exit(0 if n_fail == 0 else 1)
