"""
Smoke test for structured channel pruning.

Runs entirely on a small SYNTHETIC model + random data — no CWRU files, no full
training experiment.  It verifies the mechanics that the dissertation evidence
depends on:

  * channel-importance selection returns the right shapes and respects minimums
  * physical dimension reduction (compact model is genuinely smaller)
  * weight copying is correct (retained channels reproduce the original slice)
  * forward-pass output shape equals the class count
  * dense parameter count actually drops
  * checkpoint save -> reload reproduces predictions within tolerance
  * one miniature optimisation step runs and changes the loss
  * a JSON-serialisable result dict can be produced

Run:
    python3 tests/smoke_structured_pruning.py
Exit code 0 = all checks passed.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.cnn1d import CNN1D
from src.models.cnn1d_pruned import CompactCNN1D
from src.efficiency import count_total_parameters
from scripts.prune_structured import (
    channel_importance, select_channels, build_compact, _run_safety_checks,
    _conv_of, N_BLOCKS, INPUT_SHAPE,
)

PASS, FAIL = "PASS", "FAIL"


def _check(name, cond):
    print(f"  [{PASS if cond else FAIL}] {name}")
    if not cond:
        raise AssertionError(name)


def main() -> int:
    torch.manual_seed(0)
    device = torch.device("cpu")
    num_classes = 4
    ratio = 0.25
    min_ch = 4

    # A full-width model with known random weights stands in for a trained ckpt.
    original = CNN1D(num_classes=num_classes).to(device).eval()
    orig_channels = tuple(_conv_of(original, b).weight.shape[0] for b in range(N_BLOCKS))
    print(f"Structured-pruning smoke test  (channels {orig_channels}, ratio {ratio})")

    # 1 — importance + selection
    for crit in ("l1", "bn_gamma"):
        scores = channel_importance(original, crit)
        keep = select_channels(scores, ratio, min_ch)
        _check(f"selection returns {N_BLOCKS} index tensors ({crit})", len(keep) == N_BLOCKS)
        _check(f"every layer keeps >= min_channels ({crit})",
               all(k.numel() >= min_ch for k in keep))
        _check(f"every layer pruned below original ({crit})",
               all(k.numel() < oc for k, oc in zip(keep, orig_channels)))
        _check(f"indices sorted & unique ({crit})",
               all(torch.equal(k, torch.sort(torch.unique(k)).values) for k in keep))

    # 2 — physical compaction + safety checks
    scores = channel_importance(original, "l1")
    keep = select_channels(scores, ratio, min_ch)
    compact = build_compact(original, keep, num_classes).to(device)
    _run_safety_checks(compact, original, keep, num_classes, device)
    _check("safety checks pass", True)

    pruned_channels = tuple(int(k.numel()) for k in keep)
    _check("compact param count < original",
           count_total_parameters(compact) < count_total_parameters(original))

    # 3 — weight copying correctness: block-0 retained filters must equal the
    #     corresponding slice of the original conv weight exactly.
    w_src = _conv_of(original, 0).weight.detach().index_select(0, keep[0])
    w_dst = _conv_of(compact, 0).weight.detach()
    _check("block-0 retained filters copied exactly", torch.allclose(w_src, w_dst))
    _check("head input dim == last kept channels",
           compact.head.weight.shape == (num_classes, pruned_channels[-1]))
    _check("head output dim == num_classes (unpruned)",
           compact.head.weight.shape[0] == num_classes)

    # 4 — forward-pass shape
    x = torch.randn(5, *INPUT_SHAPE[1:], device=device)
    with torch.no_grad():
        out = compact(x)
    _check("forward output shape (B, num_classes)", out.shape == (5, num_classes))

    # 5 — save / reload reproduces predictions
    with tempfile.TemporaryDirectory() as td:
        ckpt = Path(td) / "compact.pt"
        torch.save({"state_dict": compact.state_dict(),
                    "channels": pruned_channels, "num_classes": num_classes}, ckpt)
        blob = torch.load(ckpt, map_location="cpu", weights_only=True)
        reloaded = CompactCNN1D(channels=blob["channels"],
                                num_classes=blob["num_classes"]).to(device).eval()
        reloaded.load_state_dict(blob["state_dict"])
        with torch.no_grad():
            _check("reloaded predictions match saved model",
                   torch.allclose(compact(x), reloaded(x), atol=1e-6))

    # 6 — one miniature optimisation step actually updates weights / loss
    compact.train()
    opt = torch.optim.Adam(compact.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    y = torch.randint(0, num_classes, (5,), device=device)
    l0 = crit(compact(x), y)
    opt.zero_grad(); l0.backward(); opt.step()
    with torch.no_grad():
        l1 = crit(compact(x), y)
    l0v, l1v = l0.detach().item(), l1.item()
    _check("miniature optimisation step runs & changes loss",
           torch.isfinite(l0) and torch.isfinite(l1) and l0v != l1v)

    # 7 — a JSON-serialisable result dict can be produced
    result = {
        "pruned_channels": list(pruned_channels),
        "param_before": count_total_parameters(original),
        "param_after": count_total_parameters(compact),
        "macro_f1_placeholder": 0.0,
    }
    json.dumps(result)
    _check("result dict is JSON-serialisable", True)

    print("\nSMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
