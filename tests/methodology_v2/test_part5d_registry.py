"""Automated tests for the Part-5D frozen experiment registry."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.methodology_v2.part2_builder import verify_frozen_hashes  # noqa: E402
from src.methodology_v2.part3b_windows import (PART3B_DIR,  # noqa: E402
                                               verify_part3b_hashes)
from src.methodology_v2.part4c_normalizers import (  # noqa: E402
    verify_part4c_hashes)
from src.methodology_v2.experiment.heads import (CLASS_ORDERS,  # noqa: E402
                                                 DatasetHeads, head_seed)
from src.methodology_v2.experiment.label_subsets import (  # noqa: E402
    FRACTIONS, SEEDS, build_subset_table)
from src.methodology_v2.experiment.metrics import (  # noqa: E402
    classification_report, macro_domain_f1, macro_domain_recon_mse)
from src.methodology_v2.experiment.registry import (  # noqa: E402
    PART5D_DIR, UPSTREAM, verify_part5d_hash)
from src.methodology_v2.experiment.samplers import (  # noqa: E402
    SSLSampler, SupervisedSampler)
from src.methodology_v2.experiment.trainers import (  # noqa: E402
    DOWNSTREAM_EPOCHS, OPTIMIZER_SPEC, SSL_EPOCHS, build_patch_mask,
    validation_mask_seed)

needs_registry = pytest.mark.skipif(
    not (PART5D_DIR / "part5d_hashes.csv").exists(),
    reason="run scripts/methodology_v2/run_part5d.py first")


# ---------------------------------------------------------------------------
# upstream + seal
# ---------------------------------------------------------------------------

def test_all_upstream_seals():
    verify_frozen_hashes(); verify_part3b_hashes(); verify_part4c_hashes()
    spec = json.load(open(REPO_ROOT / "methodology_v2" / "part5_encoder"
                          / "pcste_encoder_spec.yaml"))
    h = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    assert h == UPSTREAM["part5b_architecture"] or h == (
        REPO_ROOT / "methodology_v2" / "part5_encoder"
        / "part5b_architecture_hash.txt").read_text().strip()


@needs_registry
def test_part5d_master_hash_verifies_and_fails_closed(tmp_path):
    verify_part5d_hash()
    import shutil
    t = tmp_path / "t"
    shutil.copytree(PART5D_DIR, t)
    f = t / "main_run_registry.csv"
    f.write_text(f.read_text().replace("s0_f1_s42_l100",
                                       "s0_f1_s42_l101", 1))
    with pytest.raises(AssertionError, match="fail closed"):
        verify_part5d_hash(t)


# ---------------------------------------------------------------------------
# heads
# ---------------------------------------------------------------------------

def test_heads_linear_only_correct_dims_and_pairing():
    h = DatasetHeads(init_seed=head_seed(1, 42))
    assert isinstance(h.heads["CWRU"], torch.nn.Linear)
    dims = {ds: h.heads[ds].out_features for ds in CLASS_ORDERS}
    assert dims == {"CWRU": 3, "JNU": 4, "HIT": 3, "MAFAULDA": 10}
    for ds in CLASS_ORDERS:               # single linear, no hidden MLP
        assert list(h.heads[ds].children()) == []
    h2 = DatasetHeads(init_seed=head_seed(1, 42))
    for (k1, a), (k2, b) in zip(h.state_dict().items(),
                                h2.state_dict().items()):
        assert k1 == k2 and torch.equal(a, b)   # paired S0/S1 identity
    h3 = DatasetHeads(init_seed=head_seed(2, 42))
    assert not all(torch.equal(a, b) for (_, a), (_, b) in
                   zip(h.state_dict().items(), h3.state_dict().items()))


def test_class_orders_match_frozen_taxonomy():
    man = pd.read_csv(PART3B_DIR / "window_manifest_fold_1.csv")
    for ds, classes in CLASS_ORDERS.items():
        sub = man[man["dataset"] == ds]
        field = "fault_type" if ds == "CWRU" else "original_label"
        assert set(sub[field].astype(str)) == set(classes), ds


# ---------------------------------------------------------------------------
# label fractions
# ---------------------------------------------------------------------------

@needs_registry
def test_nested_subsets_train_only_all_classes_deterministic():
    path = PART5D_DIR / "label_subsets" / "label_subset_f1_s42.csv"
    df = pd.read_csv(path)
    man = pd.read_csv(PART3B_DIR / "window_manifest_fold_1.csv")
    train_ids = set(man.loc[man["split"] == "train", "window_id"])
    assert set(df["window_id"]) <= train_ids           # TRAIN only
    assert len(df) == len(train_ids)                   # complete cover
    # nesting 5 c 10 c 25 c 50 c 100
    for lo, hi in ((5, 10), (10, 25), (25, 50), (50, 100)):
        sel_lo = set(df.loc[df[f"frac_{lo}"], "window_id"])
        sel_hi = set(df.loc[df[f"frac_{hi}"], "window_id"])
        assert sel_lo <= sel_hi, f"{lo}% not nested in {hi}%"
    # every class represented at 5%
    for (ds, cls), grp in df.groupby(["dataset", "class"]):
        assert grp["frac_5"].sum() >= 1, (ds, cls)
        assert grp["frac_100"].all()
    # deterministic regeneration matches the frozen artifact
    redo = build_subset_table(1, 42)
    pd.testing.assert_frame_equal(
        redo.reset_index(drop=True),
        df.drop(columns=["parent_train_manifest_sha256"])
        .reset_index(drop=True), check_dtype=False)


@needs_registry
def test_subset_group_coverage_and_ceil_counts():
    df = pd.read_csv(PART5D_DIR / "label_subsets"
                     / "label_subset_f1_s42.csv")
    import math
    for (ds, cls), grp in df.groupby(["dataset", "class"]):
        n = len(grp)
        for f in (5, 10, 25, 50):
            k = int(grp[f"frac_{f}"].sum())
            assert k == math.ceil(f / 100 * n), (ds, cls, f)
        # round-robin group coverage: at 25%, selected groups >= min(
        # n_groups, n_selected)
        n_groups = grp["group_id"].nunique()
        sel = grp[grp["frac_25"]]
        assert sel["group_id"].nunique() == min(n_groups, len(sel))


# ---------------------------------------------------------------------------
# samplers
# ---------------------------------------------------------------------------

def test_ssl_sampler_label_free_and_balanced():
    man = pd.read_csv(PART3B_DIR / "window_manifest_fold_1.csv")
    tr = man[man["split"] == "train"]
    with pytest.raises(AssertionError, match="label"):
        SSLSampler(tr, seed=1)             # label columns present -> refuse
    view = tr[["dataset", "group_id", "window_id"]]
    s = SSLSampler(view, seed=1)
    batch = s.next_batch()
    assert len(batch) == 64
    from collections import Counter
    counts = Counter(ds for ds, _ in batch)
    assert counts == {"CWRU": 16, "JNU": 16, "HIT": 16, "MAFAULDA": 16}
    # deterministic
    s2 = SSLSampler(view, seed=1)
    assert s2.next_batch() == SSLSampler(view, seed=1).next_batch()


@needs_registry
def test_supervised_sampler_hierarchy_and_subset_only():
    df = pd.read_csv(PART5D_DIR / "label_subsets"
                     / "label_subset_f1_s42.csv")
    sam = SupervisedSampler(df, fraction=0.05, seed=42)
    allowed = set(df.loc[df["frac_5"], "window_id"])
    seen_classes = {ds: set() for ds in CLASS_ORDERS}
    for _ in range(20):
        for ds, cls, wid in sam.next_batch():
            assert wid in allowed          # only selected labelled windows
            seen_classes[ds].add(cls)
    for ds, classes in CLASS_ORDERS.items():
        assert seen_classes[ds] == set(classes), ds   # class cycling


# ---------------------------------------------------------------------------
# optimizer / schedules / masks
# ---------------------------------------------------------------------------

def test_frozen_optimizer_and_schedules():
    assert OPTIMIZER_SPEC == {"optimizer": "AdamW", "lr": 3e-4,
                              "betas": (0.9, 0.95), "eps": 1e-8,
                              "weight_decay": 0.05,
                              "grad_clip_global_norm": 1.0,
                              "min_lr": 1e-6, "warmup_epochs": 5}
    assert SSL_EPOCHS == 60 and DOWNSTREAM_EPOCHS == 50


def test_validation_masks_fixed_across_epochs():
    metas = [("CWRU", "w1"), ("HIT", "w2")]
    grids = {"CWRU": (33, 23), "JNU": (33, 24), "HIT": (17, 24),
             "MAFAULDA": (33, 24)}
    v0 = build_patch_mask(metas, grids, seed=42, epoch=0,
                          fixed_validation=True)
    v9 = build_patch_mask(metas, grids, seed=42, epoch=9,
                          fixed_validation=True)
    assert torch.equal(v0, v9)             # epoch-independent
    t0 = build_patch_mask(metas, grids, seed=42, epoch=0,
                          fixed_validation=False)
    t1 = build_patch_mask(metas, grids, seed=42, epoch=1,
                          fixed_validation=False)
    assert not torch.equal(t0, t1)         # train masks vary
    assert validation_mask_seed(42) != validation_mask_seed(1337)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def test_metrics_reference_agreement_and_absent_classes():
    y_true = ["n", "ib", "ob", "tb", "n", "ib"]
    y_pred = ["n", "ib", "ob", "n", "n", "ob"]
    rep = classification_report(y_true, y_pred, "JNU")
    try:
        from sklearn.metrics import f1_score, accuracy_score
        skl = f1_score(y_true, y_pred, labels=list(CLASS_ORDERS["JNU"]),
                       average="macro", zero_division=0)
        assert rep["macro_f1"] == pytest.approx(skl, abs=1e-9)
        assert rep["accuracy"] == pytest.approx(
            accuracy_score(y_true, y_pred), abs=1e-9)
    except ImportError:
        pass
    # absent class (tb never predicted, and absent from truth below)
    rep2 = classification_report(["n", "n"], ["n", "ib"], "JNU")
    assert len(rep2["per_class_f1"]) == 4      # never dropped
    assert rep2["per_class_f1"][3] == 0.0      # tb absent -> 0.0
    reports = {ds: {"macro_f1": 0.5} for ds in CLASS_ORDERS}
    assert macro_domain_f1(reports) == pytest.approx(0.5)
    with pytest.raises(AssertionError):
        macro_domain_f1({"CWRU": {"macro_f1": 1.0}})
    assert macro_domain_recon_mse(
        {ds: 2.0 for ds in CLASS_ORDERS}) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# experiment matrix
# ---------------------------------------------------------------------------

@needs_registry
def test_run_matrix_counts_and_pairing():
    ssl = pd.read_csv(PART5D_DIR / "ssl_run_registry.csv")
    main = pd.read_csv(PART5D_DIR / "main_run_registry.csv")
    assert len(ssl) == 9
    assert len(main) == 90
    assert (main["arm"] == "S0").sum() == 45
    assert (main["arm"] == "S1").sum() == 45
    assert main["run_id"].is_unique and ssl["run_id"].is_unique
    assert (main["status"] == "REGISTERED").all()
    # pairing: every S1 row has an S0 partner with same fold/seed/frac
    # and the SAME label-subset hash; S1 depends on its fold/seed SSL run
    for _, r in main[main["arm"] == "S1"].iterrows():
        mate = main[(main["arm"] == "S0") & (main["fold"] == r["fold"])
                    & (main["seed"] == r["seed"])
                    & (main["label_fraction"] == r["label_fraction"])]
        assert len(mate) == 1
        assert mate.iloc[0]["label_subset_hash"] == r["label_subset_hash"]
        assert r["ssl_checkpoint_dependency"] == \
            f"ssl_f{r['fold']}_s{r['seed']}"
    assert set(main["seed"]) == {42, 1337, 2026}
    assert set(main["fold"]) == {1, 2, 3}
    assert sorted(set(main["label_fraction"])) == [0.05, 0.1, 0.25,
                                                   0.5, 1.0]


@needs_registry
def test_smoke_report_bounded_and_labelled():
    smoke = json.load(open(PART5D_DIR / "smoke_test_report.json"))
    assert "NOT_AN_EXPERIMENT" in smoke["label"]
    assert smoke["ssl_smoke"]["steps"] <= 2
    assert smoke["ssl_smoke"]["mixer_grad_nonzero"]
    assert smoke["s0_smoke"]["all_heads_received_gradients"]
    assert smoke["s1_loading_smoke"]["encoder_state_loaded"]
    assert smoke["s1_loading_smoke"]["paired_heads_identical_init"]
    fz = smoke["batch64_feasibility"]
    assert fz["effective_batch"] == 64
    assert fz["micro_batch"] * fz["gradient_accumulation"] == 64
    # no experimental checkpoints saved anywhere
    for f in PART5D_DIR.rglob("*"):
        assert f.suffix.lower() not in {".pt", ".pth", ".ckpt"}


def test_no_real_training_loop_in_package():
    pkg = REPO_ROOT / "src" / "methodology_v2" / "experiment"
    for py in pkg.glob("*.py"):
        text = py.read_text()
        assert "for epoch in range" not in text, py.name
        assert "test_identity" not in text, py.name   # no TEST access
