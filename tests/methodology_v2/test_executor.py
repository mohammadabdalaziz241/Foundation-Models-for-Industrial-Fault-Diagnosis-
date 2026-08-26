"""Executor integrity tests (launch infrastructure, not methodology).

Proves the executor's mechanical decisions preserve frozen values:
  1. the in-RAM RepStore returns BIT-IDENTICAL arrays to direct sealed
     Part-4C reader calls (cached-vs-uncached equality, as required
     before any deviation from pure lazy loading);
  2. batch streams are deterministic in the seed (pairing proof basis);
  3. the exact sign-flip test matches hand-computed values;
  4. the Phase-B authorization guard refuses non-100% fractions.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "experiment_executor",
    REPO / "scripts" / "methodology_v2" / "experiment_executor.py")
ex = importlib.util.module_from_spec(SPEC)
sys.modules["experiment_executor"] = ex
SPEC.loader.exec_module(ex)

from src.methodology_v2.part3b_windows import PART3B_DIR  # noqa: E402
from src.methodology_v2.part4c_reader import get_representation  # noqa: E402


@pytest.fixture(scope="module")
def man():
    return pd.read_csv(PART3B_DIR / "window_manifest_fold_1.csv"
                       ).set_index("window_id")


def test_repstore_bit_identical_to_sealed_reader(man):
    rng = np.random.default_rng(7)
    ids = []
    for ds in ex.DATASETS:
        sub = man.index[(man["dataset"] == ds) & (man["split"] == "train")]
        ids += [sub[i] for i in rng.choice(len(sub), 2, replace=False)]
    store = ex.RepStore(1, man)
    store.preload(ids)
    for wid in ids:
        fresh, meta = get_representation(wid, 1)
        x, f, t = store.rep(wid)
        assert np.array_equal(fresh, x)
        assert x.dtype == np.float32
        assert np.array_equal(
            np.asarray(meta["frequency_hz"], np.float32), f)
        assert np.array_equal(
            np.asarray(meta["time_seconds"], np.float32), t)
    chk = store.equivalence_check(k=4, seed=0)
    assert chk["bit_equal"] and chk["checked"] == 4


def test_streams_deterministic_in_seed(man):
    view = (man[man["split"] == "train"].reset_index()
            [["dataset", "group_id", "window_id"]])
    a = ex.build_ssl_stream(view, 42, 5)
    b = ex.build_ssl_stream(view, 42, 5)
    c = ex.build_ssl_stream(view, 1337, 5)
    assert ex.stream_hash(a) == ex.stream_hash(b)
    assert ex.stream_hash(a) != ex.stream_hash(c)
    assert all(len(batch) == 64 for batch in a)
    sub = pd.read_csv(ex.SUBSET_DIR / "label_subset_f1_s42.csv")
    d1 = ex.build_sup_stream(sub, 1.0, 42, 3)
    d2 = ex.build_sup_stream(sub, 1.0, 42, 3)
    assert ex.stream_hash(d1) == ex.stream_hash(d2)
    assert all(len(t) == 3 for batch in d1 for t in batch)


def test_ssl_validation_single_dataset_chunks(man):
    """Regression: single-dataset validation chunks collate to their own
    grid (CWRU 33x23, HIT 17x24) while build_patch_mask emits the
    global-max grid; ssl_validation must slice padding-only cells and
    produce finite per-dataset MSEs for ALL four datasets."""
    import torch
    from src.methodology_v2.experiment.trainers import (GRIDS,
                                                        SSLTrainer,
                                                        build_patch_mask)
    val_by_ds, ids = {}, []
    for ds in ex.DATASETS:
        sub = man.index[(man["dataset"] == ds)
                        & (man["split"] == "validation")]
        val_by_ds[ds] = [sub[0]]
        ids.append(sub[0])
    store = ex.RepStore(1, man)
    store.preload(ids)
    trainer = SSLTrainer(seed=42, device="cpu")
    per_ds = ex.ssl_validation(trainer, store, val_by_ds)
    assert set(per_ds) == set(ex.DATASETS)
    assert all(np.isfinite(v) for v in per_ds.values())
    # slicing preserves every masked cell (padding is all-False) and the
    # sliced mask equals the frozen per-window definition exactly
    from src.methodology_v2.encoder.ssl_design import (generate_mask,
                                                       window_rng)
    from src.methodology_v2.experiment.trainers import (MASK_GEOMETRY,
                                                        MASK_RATIO,
                                                        validation_mask_seed)
    for ds in ("CWRU", "HIT"):
        wid = val_by_ds[ds][0]
        full = build_patch_mask([(ds, wid)], GRIDS, 42, 0,
                                fixed_validation=True)
        fb, tp = GRIDS[ds]
        assert int(full.sum()) == int(full[:, :fb, :tp].sum())
        frozen = generate_mask(np.ones((fb, tp), dtype=bool), MASK_RATIO,
                               MASK_GEOMETRY,
                               window_rng(validation_mask_seed(42), 0, wid))
        assert torch.equal(full[0, :fb, :tp], torch.from_numpy(frozen))


def test_exact_sign_flip_hand_computed():
    # deltas [1,2,3]: only ++ + and -- - reach |mean| >= 2 -> p = 2/8
    assert ex.exact_sign_flip_p([1.0, 2.0, 3.0]) == pytest.approx(2 / 8)
    # symmetric single delta: p = 1.0 (both patterns tie exactly)
    assert ex.exact_sign_flip_p([0.5]) == pytest.approx(1.0)
    # all-zero deltas: every pattern ties -> p = 1.0
    assert ex.exact_sign_flip_p([0.0, 0.0]) == pytest.approx(1.0)


def test_phase_b_guard_refuses_fewshot():
    with pytest.raises((AssertionError, SystemExit)):
        ex.run_downstream("s0_f1_s42_l050")


def test_config_hash_ignores_only_status():
    reg = pd.read_csv(ex.PART5D_DIR / "ssl_run_registry.csv"
                      ).set_index("run_id")
    row = reg.loc["ssl_f1_s42"]
    h1 = ex.config_hash(row)
    row2 = row.copy()
    row2["status"] = "SOMETHING_ELSE"
    assert ex.config_hash(row2) == h1
    row3 = row.copy()
    row3["seed"] = 43
    assert ex.config_hash(row3) != h1
