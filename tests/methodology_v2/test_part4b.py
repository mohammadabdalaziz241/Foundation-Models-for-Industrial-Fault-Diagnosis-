"""Focused tests for the Part-4B representation verification and
N1/N2 normalization study."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.methodology_v2 import FORBIDDEN_IMPORTS  # noqa: E402
from src.methodology_v2.part2_builder import verify_frozen_hashes  # noqa: E402
from src.methodology_v2.part3b_windows import verify_part3b_hashes  # noqa: E402
from src.methodology_v2.part4a_repdesign import (  # noqa: E402
    Part4ADataAccessError, load_fold1_train)
from src.methodology_v2.part4b_freeze import (FINAL_TRANSFORM,  # noqa: E402
                                              PART4B_DIR, RATES, STD_FLOOR,
                                              TF_CONFIG, N2Stats,
                                              build_dev_sets, frequency_hz,
                                              n1_normalize, n2_normalize,
                                              rep_of, verification_table)

needs_study = pytest.mark.skipif(
    not (PART4B_DIR / "normalization_comparison.csv").exists(),
    reason="run scripts/methodology_v2/run_part4b.py first")


# ---------------------------------------------------------------------------
# frozen upstream + configuration
# ---------------------------------------------------------------------------

def test_upstream_seals_intact():
    verify_frozen_hashes()
    verify_part3b_hashes()


def test_physically_matched_configuration_exact():
    assert TF_CONFIG == {"CWRU": (1024, 256), "JNU": (1024, 256),
                         "HIT": (512, 128), "MAFAULDA": (1024, 256)}
    t = verification_table().set_index("dataset")
    assert t.loc["HIT", "analysis_ms"] == pytest.approx(20.480)
    assert t.loc["JNU", "analysis_ms"] == pytest.approx(20.480)
    assert t.loc["CWRU", "analysis_ms"] == pytest.approx(21.333)
    assert t.loc["HIT", "freq_resolution_hz"] == pytest.approx(48.828)
    assert t.loc["CWRU", "freq_resolution_hz"] == pytest.approx(46.875)
    # matched goal: 20-21.4 ms frames, 5.1-5.4 ms hop, 46.8-48.9 Hz bins
    assert t["analysis_ms"].between(20.4, 21.4).all()
    assert t["hop_ms"].between(5.1, 5.4).all()
    assert t["freq_resolution_hz"].between(46.8, 48.9).all()
    # shapes: (513,184) CWRU, (513,192) JNU/MaF, (257,192) HIT
    assert t.loc["CWRU", "tensor_shape"] == "(513, 184)"
    assert t.loc["JNU", "tensor_shape"] == "(513, 192)"
    assert t.loc["HIT", "tensor_shape"] == "(257, 192)"
    assert t.loc["MAFAULDA", "tensor_shape"] == "(513, 192)"
    # time-frame mismatch now small (8 frames) vs universal-1024 (98)
    assert 192 - 184 == 8


def test_physical_frequency_metadata():
    f_hit = frequency_hz("HIT")
    assert f_hit.size == 257
    assert f_hit[1] == pytest.approx(25_000 / 512)
    assert f_hit[-1] == pytest.approx(12_500)
    f_jnu = frequency_hz("JNU")
    assert f_jnu[-1] == pytest.approx(25_000)
    # matched-grid property: HIT (512@25k) and JNU/MaF (1024@50k) share an
    # IDENTICAL bin->Hz mapping over HIT's range; only CWRU differs
    assert f_hit[100] == pytest.approx(f_jnu[100])
    f_cwru = frequency_hz("CWRU")
    assert f_cwru[100] != pytest.approx(f_jnu[100])


def test_final_transform_is_log1p_only():
    assert FINAL_TRANSFORM == "log1p"
    x = np.random.default_rng(0).normal(size=25_000)
    rep = rep_of(x, "HIT")
    assert rep.shape == (192, 257)
    assert np.isfinite(rep).all()
    assert (rep >= 0).all()          # log1p of magnitude is non-negative
    src = (REPO_ROOT / "src" / "methodology_v2"
           / "part4b_freeze.py").read_text()
    for banned in ("PIL", "cv2", "skimage", "Resize", "interpolate",
                   "rgb", "RGB", "jpeg"):
        assert banned not in src


# ---------------------------------------------------------------------------
# N1
# ---------------------------------------------------------------------------

def test_n1_zero_mean_unit_std_and_floor():
    rng = np.random.default_rng(1)
    rep = np.abs(rng.normal(size=(192, 257)))
    xn, floored = n1_normalize(rep)
    assert not floored
    assert abs(float(xn.mean())) < 1e-10
    assert float(xn.std()) == pytest.approx(1.0, abs=1e-9)
    # near-zero variance handling: documented floor, flagged, finite
    flat = np.full((10, 5), 3.14)
    xz, fz = n1_normalize(flat)
    assert fz and np.isfinite(xz).all()
    assert np.allclose(xz, 0.0, atol=1e-6)  # float rounding of mean / floor


# ---------------------------------------------------------------------------
# N2
# ---------------------------------------------------------------------------

def test_n2_dataset_and_bin_specific_no_sharing():
    rng = np.random.default_rng(2)
    st = N2Stats()
    a = np.abs(rng.normal(loc=5, size=(100, 8)))
    b = np.abs(rng.normal(loc=1, size=(100, 8)))
    st.add("DS_A", a)
    st.add("DS_B", b)
    out = st.finalize()
    assert set(out) == {"DS_A", "DS_B"}
    assert not np.allclose(out["DS_A"]["mu"], out["DS_B"]["mu"])
    # per-bin: constant offset on one bin moves only that bin's mu
    c = a.copy()
    c[:, 3] += 100
    st2 = N2Stats()
    st2.add("DS_C", c)
    mu = st2.finalize()["DS_C"]["mu"]
    assert mu[3] > mu[2] + 90
    # normalization uses max(std, floor) and stays finite
    z = n2_normalize(c, st2.finalize()["DS_C"])
    assert np.isfinite(z).all()
    assert abs(float(z.mean())) < 1e-8


@needs_study
def test_n2_stats_fitted_from_train_only():
    log = json.load(open(PART4B_DIR / "part4b_signal_access_log.json"))
    train_ids = set(load_fold1_train()["window_id"])
    assert set(log["window_ids_stats_sample"]) <= train_ids
    assert set(log["window_ids_qualitative"]) <= train_ids
    # and the dev-window csv agrees
    dev = pd.read_csv(PART4B_DIR / "part4b_development_windows.csv")
    assert set(dev["window_id"]) <= train_ids


@needs_study
def test_development_windows_deterministic():
    _, qual, sample = build_dev_sets()
    dev = pd.read_csv(PART4B_DIR / "part4b_development_windows.csv")
    assert dev[dev["role"] == "stats_sample"]["window_id"].tolist() \
        == sample["window_id"].tolist()
    assert dev[dev["role"] == "qualitative"]["window_id"].tolist() \
        == qual["window_id"].tolist()


def test_guard_still_refuses_test_windows():
    from src.methodology_v2.part4a_repdesign import assert_fold1_train
    df = pd.read_csv(REPO_ROOT / "methodology_v2" / "part3_windows"
                     / "window_manifest_fold_1.csv")
    with pytest.raises(Part4ADataAccessError):
        assert_fold1_train(df[df["split"] == "test"].iloc[0])


# ---------------------------------------------------------------------------
# study artefacts
# ---------------------------------------------------------------------------

@needs_study
def test_comparison_results_consistent():
    c = pd.read_csv(PART4B_DIR / "normalization_comparison.csv")
    n1 = c[c["strategy"] == "N1"]
    assert (n1["post_mean"].abs() < 1e-3).all()
    assert (n1["post_std"] - 1.0).abs().max() < 1e-3
    # N1 is affine per window -> structure exactly preserved
    assert (n1["freq_profile_corr_pre_post"] > 0.9999).all()
    assert (n1["temporal_profile_corr_pre_post"] > 0.9999).all()
    n2 = c[c["strategy"] == "N2"]
    assert (n2["temporal_profile_corr_pre_post"] > 0.55).all()
    assert n2["temporal_profile_corr_pre_post"].mean() > 0.85
    num = pd.read_csv(PART4B_DIR / "normalization_numerics.csv")
    assert num["finite"].all()
    amp = pd.read_csv(PART4B_DIR / "amplitude_retention_metrics.csv")
    assert (amp["rank_corr_pre_energy_vs_postN1"] == 0).all()
    assert (amp["rank_corr_pre_energy_vs_postN2"] > 0.9).all()


@needs_study
def test_no_full_dataset_generation_or_models():
    banned = {".npy", ".npz", ".pt", ".pth", ".h5"}
    for f in PART4B_DIR.rglob("*"):
        assert f.suffix.lower() not in banned
    log = json.load(open(PART4B_DIR / "part4b_signal_access_log.json"))
    n_read = log["n_stats_sample"] + log["n_qualitative"]
    assert n_read < 1500            # bounded study, not 54,254 windows
    from conftest import assert_imports_clean
    assert_imports_clean(["src.methodology_v2.part4b_freeze"])
