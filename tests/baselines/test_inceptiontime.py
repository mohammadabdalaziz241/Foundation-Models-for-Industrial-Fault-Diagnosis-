from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.nn import functional as F

from src.baselines.inceptiontime import (CLASS_ORDERS, DATASETS,
                                         FourDomainInceptionTime)
from src.baselines.three_domain import (GuardedWindowAccess,
                                        FourDomainSampler, macro4,
                                        microbatch_objective)


def synthetic_train() -> pd.DataFrame:
    rows = []
    for ds in DATASETS:
        for label in CLASS_ORDERS[ds]:
            for group in range(2):
                for win in range(2):
                    rows.append({"dataset": ds, "split": "train",
                                 "original_label": label, "fault_type": label,
                                 "group_id": f"{ds}-{label}-{group}",
                                 "window_id": f"{ds}-{label}-{group}-{win}"})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("length", [25_000, 48_000, 50_000])
def test_inceptiontime_output_shape(length):
    model = FourDomainInceptionTime().eval()
    with torch.no_grad():
        features = model.encoder(torch.zeros(1, 1, length))
    assert features.shape == (1, 128)


def test_four_head_shapes():
    model = FourDomainInceptionTime().eval()
    assert {ds: model.heads[ds].out_features for ds in DATASETS} == {
        "CWRU": 3, "JNU": 4, "HIT": 3, "MAFAULDA": 10}


def test_sampler_size_rotation_and_train_only():
    sampler = FourDomainSampler(synthetic_train(), 42)
    expected = [(16, 16, 16, 16)] * 3
    valid = set(synthetic_train().window_id)
    for counts in expected:
        batch = sampler.next_batch()
        assert len(batch) == 64
        assert tuple(sum(ds == d for ds, _, _ in batch) for d in DATASETS) == counts
        assert all(wid in valid for _, _, wid in batch)


def test_sampler_rejects_non_train_and_missing_domain():
    bad = synthetic_train(); bad.loc[0, "split"] = "test"
    with pytest.raises(AssertionError, match="TRAIN only"):
        FourDomainSampler(bad, 42)
    missing = synthetic_train()[synthetic_train().dataset != "CWRU"]
    with pytest.raises(AssertionError, match="exactly"):
        FourDomainSampler(missing, 42)


def test_microbatch_loss_exactly_matches_dataset_mean():
    logits = torch.tensor([[2., 0.], [0., 1.], [1., 3.], [4., 1.]],
                          requires_grad=True)
    y = torch.tensor([0, 1, 1, 0])
    got = microbatch_objective([logits[:1], logits[1:3], logits[3:]],
                               [y[:1], y[1:3], y[3:]], 4)
    expected = F.cross_entropy(logits, y) / 4
    torch.testing.assert_close(got, expected)


def test_macro4_exact_equal_domain_mean_and_rejects_extra_domain():
    per = {ds: {"macro_f1": value} for ds, value in
           zip(DATASETS, (0.2, 0.4, 0.6, 0.8))}
    assert macro4(per, "macro_f1") == pytest.approx(0.5)
    per["EXTRA"] = {"macro_f1": 1.0}
    with pytest.raises(AssertionError, match="exactly"):
        macro4(per, "macro_f1")


def access_frame(split="test", dataset="JNU"):
    return pd.DataFrame([{"window_id": "w", "dataset": dataset,
                          "split": split}])


def test_smoke_cannot_access_test_even_if_seal_exists(tmp_path):
    seal = tmp_path / "test_seal.json"; seal.write_text("{}")
    called = []
    access = GuardedWindowAccess(access_frame(), seal,
                                 lambda row: called.append(row) or np.zeros(50_000),
                                 smoke=True)
    with pytest.raises(RuntimeError, match="sealed"):
        access.read("w", "test")
    assert not called


def test_real_test_loader_inaccessible_before_seal(tmp_path):
    called = []
    access = GuardedWindowAccess(access_frame(), tmp_path / "test_seal.json",
                                 lambda row: called.append(row) or np.zeros(50_000))
    with pytest.raises(RuntimeError, match="sealed"):
        access.read("w", "test")
    assert not called


def test_baseline_source_is_four_domain_and_isolated():
    root = Path(__file__).resolve().parents[2]
    source = (root / "scripts/baselines/run_inceptiontime.py").read_text()
    assert "FourDomainSampler" in source
    assert "macro4_f1" in source
    assert "inceptiontime_four_domain" in source
