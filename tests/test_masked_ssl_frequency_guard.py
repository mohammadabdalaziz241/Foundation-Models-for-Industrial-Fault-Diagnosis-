"""Equivalence tests for the zero-weight frequency-term guard (Stage 3 Decision 3).

The guard skips the rFFT and its backward when `frequency_weight == 0`. That is a
removal of mathematically inactive computation, **not** a change to O0 — and this file
is the proof. Every quantity the optimiser can see is compared between the guarded path
and the previously unguarded path, under a strict declared tolerance of **exact
equality** (`torch.equal`), because `0.0 * f` and `0.0 * df/dx` are exactly zero in IEEE
754 and adding exact zero to a gradient buffer is a no-op.

Behaviour at `frequency_weight > 0` must be untouched; that is asserted too.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.cv_paderborn import seed_everything
from src.models.cnn1d import CNN1D
from src.models.masked_ssl import MaskedSSL

# Declared tolerance for the guarded-vs-unguarded comparison: EXACT.
EXACT = 0.0


def _build(force: bool, frequency_weight: float = 0.0, seed: int = 11) -> MaskedSSL:
    seed_everything(seed)
    encoder = CNN1D(num_classes=3)
    return MaskedSSL(encoder_model=encoder, patch_len=32, mask_ratio=0.5,
                     frequency_weight=frequency_weight, consistency_weight=0.0,
                     _force_auxiliary_terms=force)


def _one_step(model: MaskedSSL, x: torch.Tensor, seed: int = 5):
    """Forward, backward and one Adam step. Returns every observable quantity."""
    params = list(dict.fromkeys(list(model.encoder.parameters())
                                + list(model.decoder.parameters())))
    opt = torch.optim.Adam(params, lr=1e-3)
    seed_everything(seed)                       # identical mask draw
    recon, loss = model(x)
    components = {k: v.clone() for k, v in model.last_loss_components.items()}
    opt.zero_grad(set_to_none=True)
    loss.backward()
    enc_grads = {n: p.grad.detach().clone() for n, p in model.encoder.named_parameters()
                 if p.grad is not None}
    dec_grads = {n: p.grad.detach().clone() for n, p in model.decoder.named_parameters()
                 if p.grad is not None}
    opt.step()
    weights = {n: p.detach().clone() for n, p in model.named_parameters()}
    return {"recon": recon.detach().clone(), "loss": loss.detach().clone(),
            "components": components, "encoder_grads": enc_grads,
            "decoder_grads": dec_grads, "weights": weights}


def _assert_identical(a: dict, b: dict, what: str) -> None:
    assert set(a) == set(b), f"{what}: key sets differ"
    for k in sorted(a):
        assert torch.equal(a[k], b[k]), (
            f"{what}[{k}] differs; max abs {float((a[k] - b[k]).abs().max()):.3e}")


@pytest.fixture(scope="module")
def batch() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(8, 1, 1024)


# ===========================================================================
# the guard is equivalent at zero weight
# ===========================================================================

def test_masked_mse_loss_is_identical(batch):
    guarded = _one_step(_build(force=False), batch)
    unguarded = _one_step(_build(force=True), batch)
    assert torch.equal(guarded["components"]["time_loss"],
                       unguarded["components"]["time_loss"])


def test_total_loss_is_identical(batch):
    guarded = _one_step(_build(force=False), batch)
    unguarded = _one_step(_build(force=True), batch)
    assert torch.equal(guarded["loss"], unguarded["loss"])
    assert torch.equal(guarded["components"]["total_loss"],
                       unguarded["components"]["total_loss"])


def test_reconstruction_is_identical(batch):
    guarded = _one_step(_build(force=False), batch)
    unguarded = _one_step(_build(force=True), batch)
    assert torch.equal(guarded["recon"], unguarded["recon"])


def test_encoder_gradients_are_identical(batch):
    guarded = _one_step(_build(force=False), batch)
    unguarded = _one_step(_build(force=True), batch)
    assert guarded["encoder_grads"], "no encoder gradients were produced"
    _assert_identical(guarded["encoder_grads"], unguarded["encoder_grads"], "encoder grad")


def test_decoder_gradients_are_identical(batch):
    guarded = _one_step(_build(force=False), batch)
    unguarded = _one_step(_build(force=True), batch)
    assert guarded["decoder_grads"], "no decoder gradients were produced"
    _assert_identical(guarded["decoder_grads"], unguarded["decoder_grads"], "decoder grad")


def test_one_step_optimiser_updates_and_weights_are_identical(batch):
    guarded = _one_step(_build(force=False), batch)
    unguarded = _one_step(_build(force=True), batch)
    _assert_identical(guarded["weights"], unguarded["weights"], "post-step weight")


def test_multi_step_training_stays_identical(batch):
    """Ten steps, so any drift has room to accumulate."""
    outs = []
    for force in (False, True):
        model = _build(force=force)
        params = list(dict.fromkeys(list(model.encoder.parameters())
                                    + list(model.decoder.parameters())))
        opt = torch.optim.Adam(params, lr=1e-3)
        losses = []
        for step in range(10):
            seed_everything(100 + step)
            _recon, loss = model(batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        outs.append((losses, {n: p.detach().clone() for n, p in model.named_parameters()}))
    assert outs[0][0] == outs[1][0], "per-step losses diverged"
    _assert_identical(outs[0][1], outs[1][1], "10-step weight")


def test_guard_reports_zero_frequency_loss_and_skips_the_fft(batch):
    guarded = _one_step(_build(force=False), batch)
    unguarded = _one_step(_build(force=True), batch)
    assert float(guarded["components"]["frequency_loss"]) == 0.0
    # the unguarded path really did compute a non-trivial term, so the comparison above
    # is meaningful rather than vacuous
    assert float(unguarded["components"]["frequency_loss"]) > 0.0


# ===========================================================================
# behaviour at non-zero weight is untouched
# ===========================================================================

@pytest.mark.parametrize("weight", [0.5, 1.0])
def test_non_zero_weight_still_computes_the_frequency_term(batch, weight):
    model = _build(force=False, frequency_weight=weight)
    out = _one_step(model, batch)
    freq = float(out["components"]["frequency_loss"])
    assert freq > 0.0, "the frequency term must still run when its weight is non-zero"
    expected = (float(out["components"]["time_loss"]) + weight * freq)
    assert float(out["loss"]) == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize("weight", [0.5, 1.0])
def test_forcing_the_terms_changes_nothing_at_non_zero_weight(batch, weight):
    a = _one_step(_build(force=False, frequency_weight=weight), batch)
    b = _one_step(_build(force=True, frequency_weight=weight), batch)
    assert torch.equal(a["loss"], b["loss"])
    _assert_identical(a["weights"], b["weights"], "post-step weight")


def test_o0_configuration_has_both_auxiliary_weights_at_zero():
    from src.bearing_generalisation_training import SSL_CONFIG
    assert SSL_CONFIG["frequency_weight"] == 0.0
    assert SSL_CONFIG["consistency_weight"] == 0.0
    assert "O0" in SSL_CONFIG["objective"]
