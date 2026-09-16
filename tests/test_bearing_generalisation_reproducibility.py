"""Reproducibility tests for `bearing_generalisation_v1` (Stage 3.3).

Stage 3 found the SSL stage non-reproducible on CUDA. These tests pin the diagnosis so
it cannot silently regress or be silently "fixed" without anyone noticing:

* adaptation is bit-deterministic given a fixed encoder — asserted;
* the deterministic upsample is numerically equivalent to `nn.Upsample` — asserted;
* the deterministic upsample makes SSL bit-reproducible — asserted on CUDA;
* `DETERMINISTIC_SSL_DECODER` is still OFF, because the SSL decoder is a frozen
  protocol element and turning it on requires an amendment — asserted, so the flag
  cannot be flipped without this test being updated deliberately.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.bearing_generalisation_training import (
    DETERMINISTIC_SSL_DECODER, DETERMINISTIC_UPSAMPLE_FORWARD_TOLERANCE, F3_CONFIG,
    SCRATCH_CONFIG, SSL_CONFIG, _DeterministicLinearUpsample2x, _f3_stage,
    _unfrozen_blocks, make_ssl_decoder_deterministic,
)
from src.cv_paderborn import seed_everything
from src.models.cnn1d import CNN1D
from src.models.masked_ssl import MaskedSSL

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _hash_state(sd) -> str:
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(sd[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()


# ===========================================================================
# the frozen flag
# ===========================================================================

def test_deterministic_decoder_is_enabled_by_the_stage_3_amendment():
    """Amendment 2 §9.2 revision (2026-08-01) enables the deterministic upsample for
    every new O0 pretraining run, classified as a reproducibility correction. Turning it
    back off would silently restore a protocol whose same-seed noise is 3.1x the
    registered effect threshold, so the state is pinned here."""
    assert DETERMINISTIC_SSL_DECODER is True


def test_frozen_ssl_budget_is_unchanged():
    assert SSL_CONFIG["total_sample_presentations"] == 887_040
    assert SSL_CONFIG["frequency_weight"] == 0.0, "primary objective is O0"
    assert SSL_CONFIG["consistency_weight"] == 0.0


def test_frozen_f3_and_scratch_recipes_are_matched():
    for key in ("optimizer", "scheduler", "batch_size", "max_epochs", "patience",
                "checkpoint_metric", "weight_decay"):
        assert F3_CONFIG[key] == SCRATCH_CONFIG[key], key
    assert F3_CONFIG["encoder_learning_rate"] == pytest.approx(
        F3_CONFIG["head_learning_rate"] * F3_CONFIG["encoder_lr_scale"])
    assert F3_CONFIG["warmup_epochs"] == 5
    assert F3_CONFIG["unfreeze_schedule"] == {"block3": 6, "block2": 11, "block1": 16}


# ===========================================================================
# the deterministic upsample
# ===========================================================================

@pytest.mark.parametrize("length", [17, 37, 129])
def test_deterministic_upsample_matches_nn_upsample(length):
    """The declared numerical-equivalence bound between the original and deterministic
    forward operations. This is what licenses calling the change a reproducibility
    correction rather than a model change."""
    x = torch.randn(3, 5, length)
    ref = nn.Upsample(scale_factor=2, mode="linear", align_corners=False)(x)
    got = _DeterministicLinearUpsample2x()(x)
    assert got.shape == ref.shape
    worst = float((got - ref).abs().max())
    assert worst <= DETERMINISTIC_UPSAMPLE_FORWARD_TOLERANCE, worst


def test_deterministic_upsample_gradients_match_the_original():
    """Equivalent forward AND equivalent gradient: only the backward *implementation*
    differs (atomics versus a deterministic reduction), not the mathematics."""
    x = torch.randn(2, 4, 33, dtype=torch.float64, requires_grad=True)
    ref = nn.Upsample(scale_factor=2, mode="linear", align_corners=False)(x)
    ref.pow(2).sum().backward()
    g_ref = x.grad.clone()
    x.grad = None
    got = _DeterministicLinearUpsample2x()(x)
    got.pow(2).sum().backward()
    assert torch.allclose(x.grad, g_ref, atol=1e-10), float((x.grad - g_ref).abs().max())


def test_deterministic_upsample_replaces_every_layer():
    encoder = CNN1D(num_classes=3)
    ssl = MaskedSSL(encoder_model=encoder, patch_len=32, mask_ratio=0.5)
    n_upsample = sum(isinstance(m, nn.Upsample) for m in ssl.decoder.net)
    assert n_upsample == 3, "the CNN decoder is expected to upsample three times"
    assert make_ssl_decoder_deterministic(ssl) == 3
    assert not any(isinstance(m, nn.Upsample) for m in ssl.decoder.net)


def test_masked_ssl_forward_is_unchanged_by_the_replacement():
    torch.manual_seed(0)
    x = torch.randn(4, 1, 1024)
    outs = []
    for deterministic in (False, True):
        seed_everything(7)
        enc = CNN1D(num_classes=3)
        ssl = MaskedSSL(encoder_model=enc, patch_len=32, mask_ratio=0.5)
        if deterministic:
            make_ssl_decoder_deterministic(ssl)
        seed_everything(7)                      # identical mask draw
        with torch.no_grad():
            recon, loss = ssl(x)
        outs.append((recon, float(loss)))
    assert torch.allclose(outs[0][0], outs[1][0], atol=1e-5)
    assert outs[0][1] == pytest.approx(outs[1][1], abs=1e-5)


# ===========================================================================
# where the non-determinism actually lives
# ===========================================================================

def _short_ssl(deterministic: bool, device: torch.device, steps: int = 25) -> str:
    seed_everything(42)
    enc = CNN1D(num_classes=3)
    ssl = MaskedSSL(encoder_model=enc, patch_len=32, mask_ratio=0.5,
                    frequency_weight=0.0, consistency_weight=0.0)
    if deterministic:
        make_ssl_decoder_deterministic(ssl)
    enc.to(device)
    ssl.to(device)
    params = list(dict.fromkeys(list(enc.parameters()) + list(ssl.decoder.parameters())))
    opt = torch.optim.Adam(params, lr=1e-3)
    rng = np.random.default_rng(42)
    x = torch.randn(512, 1, 1024, device=device)
    for _ in range(steps):
        idx = rng.integers(0, len(x), 32)
        _recon, loss = ssl(x[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return _hash_state(enc.encoder.state_dict())


@cuda_only
def test_stock_ssl_decoder_is_not_reproducible_on_cuda():
    """The Stage 3 finding itself. If this ever starts passing as MATCH, PyTorch has
    gained a deterministic `upsample_linear1d_backward` and the amendment can be
    revisited."""
    dev = torch.device("cuda")
    a, b = _short_ssl(False, dev), _short_ssl(False, dev)
    assert a != b, ("stock SSL now reproduces on CUDA — re-check "
                    "upsample_linear1d_backward_out_cuda and revisit the Stage 3 finding")


@cuda_only
def test_deterministic_decoder_makes_ssl_bit_reproducible_on_cuda():
    dev = torch.device("cuda")
    assert _short_ssl(True, dev) == _short_ssl(True, dev)


def test_ssl_is_reproducible_on_cpu_either_way():
    dev = torch.device("cpu")
    assert _short_ssl(False, dev, steps=6) == _short_ssl(False, dev, steps=6)


def test_model_initialisation_is_deterministic():
    seed_everything(42)
    a = _hash_state(CNN1D(num_classes=3).state_dict())
    seed_everything(42)
    b = _hash_state(CNN1D(num_classes=3).state_dict())
    assert a == b


# ===========================================================================
# F3 stage accounting (Stage 3 Decision 2) — the schedule itself is UNCHANGED
# ===========================================================================

def test_f3_schedule_is_unchanged_by_the_stage_3_amendment():
    """Decision 2: patience, epoch cap, warm-up and unfreeze points stay exactly as
    frozen in Amendment 2 §9.1. Only reporting was added."""
    assert F3_CONFIG["warmup_epochs"] == 5
    assert F3_CONFIG["unfreeze_schedule"] == {"block3": 6, "block2": 11, "block1": 16}
    assert F3_CONFIG["max_epochs"] == 40
    assert F3_CONFIG["patience"] == 10
    assert F3_CONFIG["checkpoint_metric"] == "validation bearing-balanced macro-F1 (highest)"


@pytest.mark.parametrize("epoch,stage,unfrozen", [
    (1, "phase1_head_warmup_encoder_frozen", []),
    (5, "phase1_head_warmup_encoder_frozen", []),
    (6, "phase2_block3_unfrozen", ["block3"]),
    (10, "phase2_block3_unfrozen", ["block3"]),
    (11, "phase3_block3_block2_unfrozen", ["block2", "block3"]),
    (15, "phase3_block3_block2_unfrozen", ["block2", "block3"]),
    (16, "phase4_all_blocks_unfrozen", ["block1", "block2", "block3"]),
    (40, "phase4_all_blocks_unfrozen", ["block1", "block2", "block3"]),
])
def test_f3_stage_and_unfrozen_blocks(epoch, stage, unfrozen):
    assert _f3_stage(epoch) == stage
    assert _unfrozen_blocks("F3", epoch) == unfrozen


def test_f1_never_unfreezes_and_scratch_always_is_trainable():
    assert _unfrozen_blocks("F1", 40) == []
    assert _unfrozen_blocks("scratch", 1) == ["block1", "block2", "block3"]
