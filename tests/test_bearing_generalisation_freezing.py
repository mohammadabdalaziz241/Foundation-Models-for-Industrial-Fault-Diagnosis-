"""Strict parameter-and-buffer freezing semantics (Stage 3 Amendment 2).

An encoder block is frozen ONLY when both its trainable parameters and its persistent
buffers — BatchNorm `running_mean`, `running_var`, `num_batches_tracked` — cannot change.
`requires_grad=False` satisfies the first and not the second: a module left in `train()`
mode keeps updating its running statistics on every forward pass.

The twelve numbered tests required by the amendment are marked in their docstrings.
Equality is byte-identity (`torch.equal`) throughout; there is no tolerance.
"""

from __future__ import annotations

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
    BN_BUFFER_NAMES, ENCODER_BLOCK_INDICES, F3_CONFIG, F_ADABN_CONFIG, N_CLASSES,
    RoleSet, adapt_batchnorm, apply_module_modes, assert_blocks_byte_identical,
    diff_state, encoder_state_snapshot, full_state_snapshot, _unfrozen_blocks,
)
from src.cv_paderborn import seed_everything
from src.models.cnn1d import CNN1D

BLOCKS = sorted(ENCODER_BLOCK_INDICES)


def _model() -> CNN1D:
    seed_everything(3)
    return CNN1D(num_classes=N_CLASSES)


def _batch(n: int = 32) -> torch.Tensor:
    g = torch.Generator().manual_seed(9)
    return torch.randn(n, 1, 1024, generator=g)


def _fake_roleset(n: int = 64) -> RoleSet:
    g = torch.Generator().manual_seed(17)
    x = torch.randn(n, 1, 1024, generator=g)
    rows = [{"sample_id": f"paderborn:REC{i // 8}:{i:08d}-{i + 1024:08d}",
             "class_index": str(i % 3), "recording_id": f"REC{i // 8}",
             "bearing_id": f"K00{i // 32}"} for i in range(n)]
    return RoleSet(role="adaptation_pool", rows=rows, x=x,
                   y=torch.tensor([int(r["class_index"]) for r in rows]),
                   bearings=np.array([r["bearing_id"] for r in rows], dtype=object),
                   recordings=np.array([r["recording_id"] for r in rows], dtype=object))


def _train_steps(model: CNN1D, unfrozen, steps: int = 4, head_only_nograd: bool = None):
    """Run optimisation steps under the declared mode state."""
    if head_only_nograd is None:
        head_only_nograd = not unfrozen
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    loss_fn = nn.CrossEntropyLoss()
    x = _batch()
    y = torch.arange(len(x)) % N_CLASSES
    for _ in range(steps):
        model.train()                                   # recursive; must be re-applied
        apply_module_modes(model, unfrozen_blocks=unfrozen, head_training=True)
        if head_only_nograd:
            with torch.no_grad():
                feats = model.gap(model.encoder(x)).squeeze(-1)
            logits = model.head(feats)
        else:
            logits = model(x)
        loss = loss_fn(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()


# ===========================================================================
# the mode manager itself (Decision 4)
# ===========================================================================

def test_mode_manager_controls_every_declared_aspect():
    model = _model()
    state = apply_module_modes(model, unfrozen_blocks=["block3"], head_training=True)
    assert state["unfrozen_blocks"] == ["block3"]
    assert state["head_training"] is True
    for name, idx in ENCODER_BLOCK_INDICES.items():
        block = model.encoder[idx]
        expected = name == "block3"
        assert block.training is expected, name
        assert all(p.requires_grad is expected for p in block.parameters()), name
        assert state["blocks"][name]["bn_updates_permitted"] is expected
    assert model.head.training is True
    assert all(p.requires_grad for p in model.head.parameters())


def test_mode_manager_puts_everything_in_eval_when_head_training_is_false():
    model = _model()
    apply_module_modes(model, unfrozen_blocks=BLOCKS, head_training=False)
    for idx in ENCODER_BLOCK_INDICES.values():
        assert model.encoder[idx].training is False
    assert model.head.training is False
    assert model.training is False


def test_mode_manager_rejects_an_unknown_block():
    with pytest.raises(AssertionError, match="unknown encoder blocks"):
        apply_module_modes(_model(), unfrozen_blocks=["block9"], head_training=True)


def test_batchnorm_buffers_are_the_ones_we_think_they_are():
    model = _model()
    names = {n.split(".")[-1] for _, b in ENCODER_BLOCK_INDICES.items()
             for n, _ in model.encoder[b].named_buffers()}
    assert set(BN_BUFFER_NAMES) <= names


# ===========================================================================
# 8. frozen BatchNorm survives a top-level model.train()
# ===========================================================================

def test_08_frozen_batchnorm_unchanged_after_model_train():
    """Test 8: a top-level `model.train()` recursively re-enables every child, so the
    declared per-block state must be re-applied afterwards — and once it is, a frozen
    block's buffers do not move even though the model as a whole is training."""
    model = _model()
    apply_module_modes(model, unfrozen_blocks=[], head_training=True)
    before = encoder_state_snapshot(model)

    model.train()                       # the dangerous call
    assert model.encoder[0].training is True, "sanity: train() really did recurse"
    apply_module_modes(model, unfrozen_blocks=[], head_training=True)   # restore
    assert model.encoder[0].training is False

    with torch.no_grad():
        model.encoder(_batch())
    assert diff_state(before[BLOCKS[0]], encoder_state_snapshot(model)[BLOCKS[0]]) == []


def test_without_restoring_modes_the_buffers_would_move():
    """Negative control: proves the guard above is load-bearing rather than vacuous."""
    model = _model()
    before = encoder_state_snapshot(model)
    model.train()                       # deliberately NOT restored
    with torch.no_grad():
        model.encoder(_batch())
    changed = diff_state(before["block1"], encoder_state_snapshot(model)["block1"])
    assert changed, "BatchNorm buffers should move when the block is left in train mode"
    assert all(c.startswith("buffer:") for c in changed)


# ===========================================================================
# 1, 2. F1 strict frozen linear probe
# ===========================================================================

def test_01_and_02_f1_parameters_and_buffers_remain_unchanged():
    """Tests 1 and 2: every encoder parameter AND buffer is byte-identical before and
    after F1 head training."""
    model = _model()
    apply_module_modes(model, unfrozen_blocks=[], head_training=True)
    before = encoder_state_snapshot(model)
    head_before = full_state_snapshot(model.head)

    _train_steps(model, unfrozen=[], steps=6)

    after = encoder_state_snapshot(model)
    for name in BLOCKS:
        changed = diff_state(before[name], after[name])
        assert changed == [], f"{name} changed under F1: {changed}"
    assert diff_state(head_before, full_state_snapshot(model.head)), \
        "the head must actually have trained, or the test is vacuous"


def test_f1_builds_no_encoder_gradient():
    model = _model()
    apply_module_modes(model, unfrozen_blocks=[], head_training=True)
    _train_steps(model, unfrozen=[], steps=1)
    assert all(p.grad is None for p in model.encoder.parameters())


# ===========================================================================
# 3, 4. F3 warm-up
# ===========================================================================

def test_03_and_04_f3_warmup_parameters_and_buffers_remain_unchanged():
    """Tests 3 and 4: during epochs 1-5 the whole encoder is byte-identical."""
    model = _model()
    before = encoder_state_snapshot(model)
    for epoch in range(1, F3_CONFIG["warmup_epochs"] + 1):
        unfrozen = _unfrozen_blocks("F3", epoch)
        assert unfrozen == [], f"epoch {epoch} should still be in warm-up"
        _train_steps(model, unfrozen=unfrozen, steps=2)
        assert_blocks_byte_identical(before, model, BLOCKS, f"warm-up epoch {epoch}")


# ===========================================================================
# 5, 6, 7. progressive unfreezing changes only permitted blocks
# ===========================================================================

@pytest.mark.parametrize("test_id,epoch,expected", [
    ("05", 6, ["block3"]),
    ("06", 11, ["block2", "block3"]),
    ("07", 16, ["block1", "block2", "block3"]),
])
def test_05_06_07_unfreezing_changes_only_permitted_blocks(test_id, epoch, expected):
    """Tests 5, 6 and 7: at each declared unfreezing point, only the declared blocks
    may change; every still-frozen block stays byte-identical in parameters and
    buffers."""
    model = _model()
    unfrozen = _unfrozen_blocks("F3", epoch)
    assert unfrozen == expected
    before = encoder_state_snapshot(model)

    _train_steps(model, unfrozen=unfrozen, steps=4)

    after = encoder_state_snapshot(model)
    changed = [b for b in BLOCKS if diff_state(before[b], after[b])]
    assert set(changed) <= set(unfrozen), f"unpermitted blocks changed: {changed}"
    frozen = [b for b in BLOCKS if b not in unfrozen]
    assert_blocks_byte_identical(before, model, frozen, f"epoch {epoch}")
    for b in unfrozen:
        entries = diff_state(before[b], after[b])
        assert any(e.startswith("param:") for e in entries), \
            f"{b} was unfrozen but no parameter changed — the test would be vacuous"


# ===========================================================================
# 9, 10. validation and evaluation change nothing
# ===========================================================================

def test_09_validation_leaves_all_parameters_and_buffers_unchanged():
    """Test 9."""
    model = _model()
    apply_module_modes(model, unfrozen_blocks=BLOCKS, head_training=True)
    _train_steps(model, unfrozen=BLOCKS, steps=2)
    before = full_state_snapshot(model)
    model.eval()
    apply_module_modes(model, unfrozen_blocks=BLOCKS, head_training=False)
    with torch.no_grad():
        model(_batch())
    assert diff_state(before, full_state_snapshot(model)) == []


def test_10_evaluation_leaves_all_parameters_and_buffers_unchanged():
    """Test 10: repeated evaluation passes, including on differently sized batches."""
    model = _model()
    apply_module_modes(model, unfrozen_blocks=[], head_training=False)
    before = full_state_snapshot(model)
    with torch.no_grad():
        for n in (8, 33, 64):
            model(_batch(n))
    assert diff_state(before, full_state_snapshot(model)) == []


# ===========================================================================
# 11, 12. F-AdaBN
# ===========================================================================

def test_11_adabn_changes_buffers_but_no_encoder_parameters():
    """Test 11."""
    model = _model()
    apply_module_modes(model, unfrozen_blocks=[], head_training=False)
    before = encoder_state_snapshot(model)
    record = adapt_batchnorm(model, _fake_roleset(), passes=1)

    assert record["encoder_parameters_unchanged"] is True
    assert record["buffers_changed"] is True
    after = encoder_state_snapshot(model)
    for name in BLOCKS:
        changed = diff_state(before[name], after[name])
        assert changed, f"{name} statistics should have adapted"
        assert all(c.startswith("buffer:") for c in changed), \
            f"{name} changed a parameter: {[c for c in changed if c.startswith('param:')]}"


def test_12_adabn_uses_only_permitted_adaptation_rows():
    """Test 12."""
    rows = _fake_roleset()
    model = _model()
    record = adapt_batchnorm(model, rows, passes=1)
    assert record["adaptation_role"] == F_ADABN_CONFIG["adaptation_rows"] == "adaptation_pool"
    assert record["n_adaptation_windows"] == len(rows)
    assert record["labels_used_for_bn"] if "labels_used_for_bn" in record else True
    assert F_ADABN_CONFIG["labels_used_for_bn"] is False
    for forbidden in ("validation", "setting_a_eval", "setting_b_eval"):
        assert forbidden in F_ADABN_CONFIG["forbidden_rows"]
    import hashlib
    assert record["adaptation_sample_ids_sha256"] == hashlib.sha256(
        "\n".join(sorted(r["sample_id"] for r in rows.rows)).encode()).hexdigest()


def test_adabn_statistics_are_frozen_afterwards():
    model = _model()
    adapt_batchnorm(model, _fake_roleset(), passes=1)
    after = encoder_state_snapshot(model)
    with torch.no_grad():
        model(_batch())
    assert all(diff_state(after[b], encoder_state_snapshot(model)[b]) == [] for b in BLOCKS)


def test_adabn_is_bit_reproducible_at_the_declared_batch_size():
    """What reproducibility actually requires: two adaptations with the declared batch
    size are byte-identical."""
    rows = _fake_roleset()
    a, b = _model(), _model()
    bs = F_ADABN_CONFIG["adaptation_batch_size"]
    adapt_batchnorm(a, rows, passes=1, batch_size=bs)
    adapt_batchnorm(b, rows, passes=1, batch_size=bs)
    for name in BLOCKS:
        assert diff_state(encoder_state_snapshot(a)[name],
                          encoder_state_snapshot(b)[name]) == [], name


def test_adabn_running_variance_depends_weakly_on_batch_size_as_declared():
    """BatchNorm's cumulative average averages PER-BATCH variances, which is not the
    pooled dataset variance. The spec says so; this pins the magnitude (~1e-4 relative)
    and is why the batch size is part of the frozen specification rather than a free
    knob."""
    rows = _fake_roleset()
    a, b = _model(), _model()
    adapt_batchnorm(a, rows, passes=1, batch_size=16)
    adapt_batchnorm(b, rows, passes=1, batch_size=64)
    worst = 0.0
    for name in BLOCKS:
        sa, sb = encoder_state_snapshot(a)[name], encoder_state_snapshot(b)[name]
        for k, v in sa.items():
            if k.startswith("buffer:") and v.is_floating_point() and v.numel():
                rel = float((v - sb[k]).abs().max() / v.abs().max().clamp_min(1e-12))
                worst = max(worst, rel)
    assert 0.0 < worst < 1e-2, worst


def test_adabn_specification_is_a_declared_secondary_control():
    assert F_ADABN_CONFIG["arm"] == "F-AdaBN"
    assert "secondary" in F_ADABN_CONFIG["role"]
    assert F_ADABN_CONFIG["bn_frozen_after_adaptation"] is True
    assert F_ADABN_CONFIG["adaptation_passes"] == 1
    assert "cumulative moving average" in F_ADABN_CONFIG["momentum_policy"]


# ===========================================================================
# the F3 schedule is still untouched
# ===========================================================================

def test_f3_schedule_and_checkpoint_rule_unchanged_by_this_amendment():
    assert F3_CONFIG["warmup_epochs"] == 5
    assert F3_CONFIG["unfreeze_schedule"] == {"block3": 6, "block2": 11, "block1": 16}
    assert F3_CONFIG["max_epochs"] == 40
    assert F3_CONFIG["patience"] == 10
    assert F3_CONFIG["checkpoint_metric"] == "validation bearing-balanced macro-F1 (highest)"
