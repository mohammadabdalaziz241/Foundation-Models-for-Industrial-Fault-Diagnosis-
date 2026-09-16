"""Tests for the Amendment 3 (§19.2-§19.3) declared blocks: B1/B2, F-AdaBN, Stage 5.

Covers exactly what the disposition preparation added:
  * the F_AdaBN strategy freezes everything except BatchNorm statistics, which adapt
    once and then never move;
  * the runner's block registry enumerates precisely the declared cell counts;
  * the Stage 5 confirmation partitions (make_folds(4,3,42) repeats 1-2) are valid
    rosters, distinct from repeat 0, with repeat 0 byte-identical to the committed
    Stage 4 partition.

No test here touches raw data, manifests, or the GPU.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.bearing_generalisation import (  # noqa: E402
    BEARING_REGIMES, LEGACY_COMPROMISED_BEARINGS, RECORDING_REGIMES, build_folds,
)
from src.bearing_generalisation_training import (  # noqa: E402
    F_ADABN_CONFIG, N_CLASSES, RoleSet, _unfrozen_blocks, adapt_batchnorm, finetune,
)
from src.models.cnn1d import CNN1D  # noqa: E402


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "run_bearing_generalisation_stage4",
        REPO / "scripts" / "run_bearing_generalisation_stage4.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _synthetic_roleset(n: int = 96, seed: int = 0) -> RoleSet:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 1, 1024, generator=g)
    rows = [{"sample_id": f"s{i:04d}"} for i in range(n)]
    return RoleSet(role="adaptation_pool", rows=rows, x=x,
                   y=torch.zeros(n, dtype=torch.long),
                   bearings=np.array([f"K00{1 + i % 3}" for i in range(n)]),
                   recordings=np.array([f"rec{i % 6}" for i in range(n)]))


# --- F_AdaBN strategy semantics --------------------------------------------------

def test_f_adabn_unfreezes_nothing():
    for epoch in range(1, 41):
        assert _unfrozen_blocks("F_AdaBN", epoch) == []


def test_unknown_strategy_is_rejected_before_touching_data():
    with pytest.raises(AssertionError, match="unknown fine-tuning strategy"):
        finetune("F_AdaBN_typo", None, None, "R1", 42, torch.device("cpu"))


def test_f_adabn_requires_a_pretrained_encoder():
    with pytest.raises(AssertionError, match="requires a pretrained encoder"):
        finetune("F_AdaBN", None, None, "R1", 42, torch.device("cpu"))


def test_adapt_batchnorm_changes_buffers_only():
    model = CNN1D(num_classes=N_CLASSES)
    params_before = {k: v.detach().clone()
                     for k, v in model.encoder.state_dict().items()
                     if "running_" not in k and "num_batches_tracked" not in k}
    record = adapt_batchnorm(model, _synthetic_roleset(),
                             passes=F_ADABN_CONFIG["adaptation_passes"],
                             batch_size=F_ADABN_CONFIG["adaptation_batch_size"])
    assert record["encoder_parameters_unchanged"] is True
    assert record["buffers_changed"] is True
    for name, before in params_before.items():
        after = model.encoder.state_dict()[name]
        assert torch.equal(before, after), f"parameter {name} moved"
    for block, entry in record["per_block"].items():
        assert entry["parameters_changed"] == [], block
    assert record["n_adaptation_windows"] == 96
    assert record["adaptation_role"] == "adaptation_pool"
    # every module is back in eval mode: statistics are frozen after adaptation
    assert not any(m.training for m in model.modules())


def test_adapt_batchnorm_is_deterministic_and_batch_order_independent():
    rows = _synthetic_roleset()
    stats = []
    for perm_seed in (None, 1):
        torch.manual_seed(7)                   # identical conv weights in both runs
        model = CNN1D(num_classes=N_CLASSES)
        r = rows
        if perm_seed is not None:
            order = torch.randperm(len(rows.x), generator=torch.Generator().manual_seed(perm_seed))
            r = RoleSet(role=rows.role, rows=[rows.rows[i] for i in order],
                        x=rows.x[order], y=rows.y[order],
                        bearings=rows.bearings[order.numpy()],
                        recordings=rows.recordings[order.numpy()])
        adapt_batchnorm(model, r, passes=1,
                        batch_size=F_ADABN_CONFIG["adaptation_batch_size"])
        stats.append({k: v.detach().clone()
                      for k, v in model.encoder.state_dict().items()
                      if "running_mean" in k or "running_var" in k})
    for k in stats[0]:
        assert torch.allclose(stats[0][k], stats[1][k], atol=1e-5), k


# --- runner block registry ---------------------------------------------------------

def test_block_registry_matches_declared_counts():
    runner = _load_runner()
    seeds = len(runner.DECLARED_SEEDS)

    def count(cfg, fold_names=None):
        folds = fold_names or cfg["fold_names"]
        return len(cfg["conditions"]) * len(cfg["regimes"]) * len(folds) * seeds

    assert count(runner.BLOCKS["must_complete"]) == 672
    assert count(runner.BLOCKS["b_regimes"]) == 336               # §19.2
    assert count(runner.BLOCKS["f_adabn"]) == 48                  # §19.3
    stage5_folds = tuple(f"r1fold{i}" for i in range(4))
    assert count(runner.BLOCKS["stage5"], stage5_folds) == 24     # per repeat; 48 over both


def test_f_adabn_block_scope_is_exactly_the_declaration():
    runner = _load_runner()
    cfg = runner.BLOCKS["f_adabn"]
    assert cfg["conditions"] == (("P2", "F_AdaBN"),)
    assert tuple(cfg["regimes"]) == tuple(RECORDING_REGIMES)
    assert cfg["pretrained_arms"] == ()          # reuses committed stage4 encoders
    assert cfg["stages"] == ("4.3",)
    assert "stage4_f_adabn" in str(cfg["results_root"])
    assert str(cfg["checkpoint_root"]).endswith("stage4/checkpoints")


def test_b_regimes_block_reuses_stage4_encoders_and_writes_elsewhere():
    runner = _load_runner()
    cfg = runner.BLOCKS["b_regimes"]
    assert tuple(cfg["regimes"]) == tuple(BEARING_REGIMES)
    assert cfg["conditions"] == runner.CONDITIONS
    assert cfg["stages"] == ("4.3",)
    assert cfg["results_root"] != runner.STAGE4_RESULTS
    assert str(cfg["checkpoint_root"]).endswith("stage4/checkpoints")


# --- Stage 5 confirmation partitions ------------------------------------------------

def test_repeat0_of_the_stage5_generator_is_the_committed_partition():
    committed = build_folds()                                    # make_folds(4, 1, 42)
    all_repeats = build_folds(n_repeats=3)                       # make_folds(4, 3, 42)
    repeat0 = [f for f in all_repeats if f["repeat"] == 0]
    assert repeat0 == committed


@pytest.mark.parametrize("repeat", [1, 2])
def test_stage5_repeats_are_valid_rosters(repeat):
    folds = [f for f in build_folds(n_repeats=3) if f["repeat"] == repeat]
    assert len(folds) == 4
    assert [f["fold_id"] for f in folds] == [f"r{repeat}fold{i}" for i in range(4)]
    setting_b_union: set = set()
    for f in folds:
        b = set(f["setting_b_bearings"])
        p1 = set(f["population_1_bearings"])
        assert not b & p1
        assert not (b | p1) & LEGACY_COMPROMISED_BEARINGS
        assert len(b | p1) == 21
        assert not b & setting_b_union, "a bearing is in Setting B of two folds"
        setting_b_union |= b
    assert len(setting_b_union) == 21, "Setting B must cover all 21 bearings exactly once"


def test_stage5_repeats_differ_from_repeat0_and_from_each_other():
    reps = {r: [f for f in build_folds(n_repeats=3) if f["repeat"] == r] for r in (0, 1, 2)}
    def rosters(fs):
        return tuple(tuple(f["setting_b_bearings"]) for f in fs)
    assert rosters(reps[0]) != rosters(reps[1])
    assert rosters(reps[0]) != rosters(reps[2])
    assert rosters(reps[1]) != rosters(reps[2])


# --- Stage 4.2 inventory-check mapping (added after the 2026-08-02 stage5 -----------
# fail-fast on the inapplicable P1/P4 check; protocol §19.7) -------------------------

def test_inventory_check_mapping_partitions_exactly_for_every_block():
    runner = _load_runner()
    full = set(runner.INVENTORY_CHECKS_ALL)
    for name, cfg in runner.BLOCKS.items():
        applicable = set(cfg["inventory_checks"])
        inapplicable = set(cfg["inventory_checks_inapplicable"])
        if "4.2" in cfg["stages"]:
            assert applicable | inapplicable == full, name
            assert not applicable & inapplicable, name
        else:
            assert applicable == set() and inapplicable == set(), name


def test_stage5_inventory_omits_only_the_p1_p4_check_with_a_recorded_reason():
    runner = _load_runner()
    cfg = runner.BLOCKS["stage5"]
    missing = set(runner.INVENTORY_CHECKS_ALL) - set(cfg["inventory_checks"])
    assert missing == {"6_p1_and_p4_share_the_cwru_source_digest"}
    reason = cfg["inventory_checks_inapplicable"][
        "6_p1_and_p4_share_the_cwru_source_digest"]
    assert "P2-only" in reason and "P1" in reason and "P4" in reason


def test_must_complete_inventory_runs_every_check():
    runner = _load_runner()
    cfg = runner.BLOCKS["must_complete"]
    assert tuple(cfg["inventory_checks"]) == runner.INVENTORY_CHECKS_ALL
    assert cfg["inventory_checks_inapplicable"] == {}


def _inventory_ctx(runner, tmp_path, applicable, inapplicable):
    (tmp_path / "checkpoints").mkdir(exist_ok=True)
    return {
        "checkpoint_root": tmp_path / "checkpoints",
        "results_root": tmp_path / "results",
        "block": "unit_test",
        "inventory_checks": applicable,
        "inventory_checks_inapplicable": inapplicable,
        "pretrained_arms": (), "fold_names": ("fold0",),
        "manifest_digest": "x", "cache_identity": "x",
        "preprocessing_digest": "x", "git_commit": "x",
    }


def test_stage_42_reproduces_the_stage5_failure_shape_when_check6_is_applicable(tmp_path):
    """No P1/P4 records + check 6 declared applicable == the 2026-08-02 fail-fast."""
    runner = _load_runner()
    skip4 = {"4_identical_configuration_reproduces": "unit test: no GPU pretraining"}
    applicable = tuple(k for k in runner.INVENTORY_CHECKS_ALL if k not in skip4)
    ctx = _inventory_ctx(runner, tmp_path, applicable, skip4)
    with pytest.raises(runner.FailFast, match="inventory validation failed"):
        runner.stage_42(ctx, records=[])


def test_stage_42_passes_with_the_stage5_mapping_and_records_the_reason(tmp_path):
    runner = _load_runner()
    skipped = {
        "4_identical_configuration_reproduces": "unit test: no GPU pretraining",
        "6_p1_and_p4_share_the_cwru_source_digest": "P2-only block",
    }
    applicable = tuple(k for k in runner.INVENTORY_CHECKS_ALL if k not in skipped)
    ctx = _inventory_ctx(runner, tmp_path, applicable, skipped)
    report = runner.stage_42(ctx, records=[])
    assert report["all_passed"] is True
    assert set(report["checks"]) == set(applicable)
    assert report["checks_inapplicable"] == skipped
    on_disk = json.loads(
        (tmp_path / "results" / "checkpoint_inventory" / "checkpoint_inventory.json")
        .read_text())
    assert on_disk["checks_inapplicable"] == skipped


def test_stage_42_rejects_a_mapping_that_does_not_partition(tmp_path):
    runner = _load_runner()
    applicable = tuple(runner.INVENTORY_CHECKS_ALL[:-1])   # one check simply dropped
    ctx = _inventory_ctx(runner, tmp_path, applicable, {})
    with pytest.raises(runner.FailFast, match="does not partition"):
        runner.stage_42(ctx, records=[])


# --- checkpoint reuse re-verifies the recorded hash (§19.7) -------------------------

def _fake_checkpoint(root, arm, fold, seed, payload):
    import hashlib
    d = root / f"{arm}_{fold}_seed{seed}"
    d.mkdir(parents=True)
    (d / "encoder.pt").write_bytes(payload)
    sha = hashlib.sha256(payload).hexdigest()
    (d / "pretrain_record.json").write_text(json.dumps({
        "arm": arm, "fold": fold, "seed": seed, "checkpoint_sha256": sha}))
    return d


def test_stage_41_reuse_reverifies_hashes_and_never_retrains(tmp_path):
    runner = _load_runner()
    root = tmp_path / "checkpoints"
    for seed in runner.DECLARED_SEEDS:
        _fake_checkpoint(root, "P2", "fold0", seed, payload=bytes([seed]) * 64)
    ctx = {"checkpoint_root": root, "pretrained_arms": ("P2",),
           "fold_names": ("fold0",)}
    records = runner.stage_41(ctx)          # ctx has no device/folds/cwru: any
    assert len(records) == 3                # attempt to retrain would KeyError

    (root / "P2_fold0_seed43" / "encoder.pt").write_bytes(b"tampered")
    with pytest.raises(runner.FailFast, match="does not match its pretrain_record"):
        runner.stage_41(ctx)
