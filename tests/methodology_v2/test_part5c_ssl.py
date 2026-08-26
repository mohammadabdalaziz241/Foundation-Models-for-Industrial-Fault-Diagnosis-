"""Automated tests for the Part-5C SSL design study."""
from __future__ import annotations

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
from src.methodology_v2.part3b_windows import verify_part3b_hashes  # noqa: E402
from src.methodology_v2.part4c_normalizers import (  # noqa: E402
    verify_part4c_hashes)
from src.methodology_v2.encoder import (PCSTE,  # noqa: E402
                                        collate_representations)
from src.methodology_v2.encoder.ssl_design import (  # noqa: E402
    ReconstructionProbe, baseline_mses, generate_mask,
    redundancy_metrics, window_rng)

PART5B_DIR = REPO_ROOT / "methodology_v2" / "part5_encoder"
PART5C_DIR = REPO_ROOT / "methodology_v2" / "part5_ssl_design"
needs_study = pytest.mark.skipif(
    not (PART5C_DIR / "baseline_reconstruction_metrics.csv").exists(),
    reason="run scripts/methodology_v2/run_part5c.py first")


def test_upstream_seals_and_parity():
    verify_frozen_hashes()
    verify_part3b_hashes()
    verify_part4c_hashes()
    parity = json.load(open(PART5B_DIR / "mamba_reference_parity.json"))
    assert parity["all_pass"] and parity["worst_fwd_max_abs_err"] == 0.0


# ---------------------------------------------------------------------------
# deterministic masking
# ---------------------------------------------------------------------------

def test_mask_determinism_and_epoch_variation():
    tv = np.ones((33, 24), dtype=bool)
    m1 = generate_mask(tv, 0.6, "M1_random", window_rng(7, 3, "w:x"))
    m2 = generate_mask(tv, 0.6, "M1_random", window_rng(7, 3, "w:x"))
    assert np.array_equal(m1, m2)                 # same seed/epoch/window
    m3 = generate_mask(tv, 0.6, "M1_random", window_rng(7, 4, "w:x"))
    assert not np.array_equal(m1, m3)             # epochs vary masks
    m4 = generate_mask(tv, 0.6, "M1_random", window_rng(8, 3, "w:x"))
    assert not np.array_equal(m1, m4)             # seeds independent


def test_mask_exact_ratio_and_valid_only():
    for ds_bands, ds_tp, v in ((33, 23, 759), (17, 24, 408)):
        tv = np.ones((ds_bands, ds_tp), dtype=bool)
        for geom in ("M1_random", "M2_block", "M5_mixed"):
            m = generate_mask(tv, 0.6, geom, window_rng(1, 0, "a"))
            assert int(m.sum()) == round(0.6 * v)
    # invalid patches never masked, never counted
    tv = np.ones((33, 24), dtype=bool)
    tv[17:, :] = False                            # HIT-like validity
    m = generate_mask(tv, 0.6, "M2_block", window_rng(2, 0, "b"))
    assert not m[17:, :].any()
    assert int(m.sum()) == round(0.6 * tv.sum())


# ---------------------------------------------------------------------------
# THE NON-NEGOTIABLE REQUIREMENT: reconstruction gradients reach the mixer
# ---------------------------------------------------------------------------

def _probe_batch():
    items = []
    for bins, frames, fs, n_fft in [(513, 184, 48000, 1024),
                                    (257, 192, 25000, 512),
                                    (513, 192, 50000, 1024)]:
        rng = np.random.default_rng(bins)
        x = rng.normal(size=(bins, frames)).astype(np.float32)
        f = np.arange(bins) * fs / n_fft
        t = np.arange(frames) * (n_fft // 4) / fs
        items.append((x, f, t))
    batch = collate_representations(items)
    pm = torch.zeros(3, 33, 24, dtype=torch.bool)
    grids = [(33, 23), (17, 24), (33, 24)]
    for i, (fb, tp) in enumerate(grids):
        tv = np.ones((fb, tp), dtype=bool)
        m = generate_mask(tv, 0.6, "M1_random", window_rng(1, 0, f"w{i}"))
        pm[i, :fb, :tp] = torch.from_numpy(m)
    return batch, pm


def test_masked_reconstruction_loss_trains_the_mixer():
    torch.manual_seed(0)
    enc = PCSTE()
    probe = ReconstructionProbe(enc)
    batch, pm = _probe_batch()
    out = probe(**batch, patch_mask=pm)
    out["loss"].backward()
    for name, mod in (("stem", enc.stem), ("coords", enc.coords),
                      ("temporal", enc.temporal), ("mixer", enc.mixer)):
        g = max(p.grad.abs().max().item() for p in mod.parameters()
                if p.grad is not None)
        assert g > 0, f"reconstruction gradients missing in {name}"


def test_masked_content_cannot_leak_into_predictions():
    """Changing the VALUES inside masked patches must not change any
    prediction (the embedding is replaced by the mask token); only the
    loss target changes."""
    torch.manual_seed(0)
    enc = PCSTE()
    probe = ReconstructionProbe(enc)
    batch, pm = _probe_batch()
    with torch.no_grad():
        a = probe(**batch, patch_mask=pm)
        b2 = {k: v.clone() for k, v in batch.items()}
        # perturb spec ONLY inside masked patches of sample 2
        for f in range(33):
            for t in range(24):
                if pm[2, f, t]:
                    b2["spec"][2, f * 16:(f + 1) * 16,
                               t * 8:(t + 1) * 8] += 5.0
        b = probe(**b2, patch_mask=pm)
    assert torch.equal(a["pred"][2], b["pred"][2])
    assert not torch.equal(
        a["loss"].expand(1), b["loss"].expand(1))  # targets did change


def test_loss_normalization_per_window_equal_weight():
    """A window with many masked cells must not out-weight one with
    few: the loss is the mean of per-window means."""
    torch.manual_seed(0)
    enc = PCSTE()
    probe = ReconstructionProbe(enc)
    batch, pm = _probe_batch()
    out = probe(**batch, patch_mask=pm)
    assert out["per_window_mse"].shape == (3,)
    assert torch.allclose(out["loss"], out["per_window_mse"].mean())


# ---------------------------------------------------------------------------
# baselines + artifacts
# ---------------------------------------------------------------------------

def test_baselines_sane_on_synthetic():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(257, 192)).astype(np.float32)
    tv = np.ones((17, 24), dtype=bool)
    m = generate_mask(tv, 0.6, "M1_random", window_rng(1, 0, "s"))
    b = baseline_mses(x, m)
    assert b["P0_zero"] == pytest.approx(1.0, abs=0.05)  # unit-var noise
    for v in b.values():
        assert np.isfinite(v) and v > 0
    r = redundancy_metrics(x)
    assert abs(r["temporal_corr_lag1"]) < 0.1            # iid noise


@needs_study
def test_study_artifacts_and_decoder_budget():
    for name in ("mask_redundancy_study.csv", "mask_geometry_options.csv",
                 "mask_ratio_study.csv",
                 "baseline_reconstruction_metrics.csv",
                 "decoder_options.csv", "loss_options.csv",
                 "ssl_compute_estimate.csv", "proposed_ssl_spec.yaml",
                 "part5c_recommendations.yaml",
                 "part5c_reproducibility.json"):
        assert (PART5C_DIR / name).exists(), name
    dec = pd.read_csv(PART5C_DIR / "decoder_options.csv")
    d1 = dec[dec["option"].str.startswith("D1")].iloc[0]
    assert d1["pct_of_encoder"] < 30.0
    base = pd.read_csv(PART5C_DIR / "baseline_reconstruction_metrics.csv")
    assert set(base["dataset"]) == {"CWRU", "JNU", "HIT", "MAFAULDA"}
    assert base[["P0_zero", "P1_temporal_neighbour",
                 "P2_frequency_neighbour"]].gt(0).all().all()


def test_no_trainer_code_in_ssl_design():
    src = (REPO_ROOT / "src" / "methodology_v2" / "encoder"
           / "ssl_design.py").read_text()
    for banned in ("Optimizer", "optim.", "lr_scheduler", "epoch_loop",
                   "for epoch in", "backward()", "classifier",
                   "few_shot", "CrossEntropy"):
        assert banned not in src, f"ssl_design contains '{banned}'"
    banned_files = {".pt", ".pth", ".ckpt"}
    for f in PART5C_DIR.rglob("*"):
        assert f.suffix.lower() not in banned_files
