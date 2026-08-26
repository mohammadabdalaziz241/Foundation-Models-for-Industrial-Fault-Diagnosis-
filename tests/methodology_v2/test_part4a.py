"""Focused tests for the Part-4A training-only STFT design study."""
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
from src.methodology_v2.part3b_windows import (PART3B_DIR,  # noqa: E402
                                               verify_part3b_hashes)
from src.methodology_v2.part4a_repdesign import (PART4A_DIR,  # noqa: E402
                                                 Part4ADataAccessError,
                                                 apply_transform,
                                                 assert_fold1_train,
                                                 candidate_grid,
                                                 load_fold1_train,
                                                 select_dev_windows,
                                                 tf_map)

needs_study = pytest.mark.skipif(
    not (PART4A_DIR / "part4a_development_windows.csv").exists(),
    reason="run scripts/methodology_v2/run_part4a.py first")


# ---------------------------------------------------------------------------
# frozen upstream
# ---------------------------------------------------------------------------

def test_part2_and_part3b_seals_unchanged():
    verify_frozen_hashes()
    verify_part3b_hashes()


# ---------------------------------------------------------------------------
# data access
# ---------------------------------------------------------------------------

def test_guard_refuses_non_train_windows():
    df = pd.read_csv(PART3B_DIR / "window_manifest_fold_1.csv")
    t = df[df["split"] == "test"].iloc[0]
    with pytest.raises(Part4ADataAccessError):
        assert_fold1_train(t)
    v = df[df["split"] == "validation"].iloc[0]
    with pytest.raises(Part4ADataAccessError):
        assert_fold1_train(v)


@needs_study
def test_development_windows_all_fold1_train_and_deterministic():
    dev = pd.read_csv(PART4A_DIR / "part4a_development_windows.csv")
    train = load_fold1_train()
    train_ids = set(train["window_id"])
    assert set(dev["window_id"]) <= train_ids
    assert dev["window_id"].is_unique
    # per dataset x class at most 3, every class present
    per = dev.groupby(["dataset", "class"]).size()
    assert (per <= 3).all()
    assert set(dev["dataset"]) == {"CWRU", "JNU", "HIT", "MAFAULDA"}
    # deterministic re-selection reproduces the frozen csv exactly
    redo = select_dev_windows(train)
    assert redo["window_id"].tolist() == dev["window_id"].tolist()


@needs_study
def test_signal_access_log_confined_to_fold1_train():
    log = json.load(open(PART4A_DIR / "part4a_signal_access_log.json"))
    train_ids = set(load_fold1_train()["window_id"])
    assert set(log["dev_window_ids"]) <= train_ids
    assert set(log["hit_audit_window_ids"]) <= train_ids


# ---------------------------------------------------------------------------
# transform correctness (conventions: periodic Hann, center=False,
# no padding, one-sided rfft)
# ---------------------------------------------------------------------------

def test_tf_map_shape_proves_no_centering_or_padding():
    x = np.zeros(50_000)
    z = tf_map(x, 1024, 256)
    assert z.shape == ((50_000 - 1024) // 256 + 1, 1024 // 2 + 1)
    assert z.shape[0] == 192          # centred/padded impls would differ
    z2 = tf_map(np.zeros(48_000), 2048, 1024)
    assert z2.shape == ((48_000 - 2048) // 1024 + 1, 1025)


def test_tf_map_physical_frequency_correct():
    fs, f0, n_fft = 48_000, 1_000.0, 1024
    t = np.arange(fs) / fs
    x = np.sin(2 * np.pi * f0 * t)
    mag = np.abs(tf_map(x, n_fft, n_fft // 4)).mean(axis=0)
    k = int(np.argmax(mag))
    assert abs(k * fs / n_fft - f0) <= fs / n_fft  # within one bin
    assert k == round(f0 * n_fft / fs)


def test_tf_map_deterministic():
    rng = np.random.default_rng(0)
    x = rng.normal(size=50_000)
    a = tf_map(x, 512, 128)
    b = tf_map(x.copy(), 512, 128)
    assert np.array_equal(a, b)


def test_transforms_behave():
    z = tf_map(np.random.default_rng(1).normal(size=25_000), 1024, 256)
    mag = apply_transform(z, "magnitude")
    assert (mag >= 0).all()
    assert np.allclose(apply_transform(z, "power"), mag ** 2)
    assert np.allclose(apply_transform(z, "log1p"), np.log1p(mag))
    db = apply_transform(z, "db")
    assert np.isfinite(db).all()      # eps prevents -inf on exact zeros
    assert np.isfinite(apply_transform(np.zeros((3, 5), complex),
                                       "db")).all()


@needs_study
def test_candidate_grid_math():
    g = candidate_grid()
    r = g[(g["dataset"] == "HIT") & (g["n_fft"] == 1024)
          & (g["overlap_pct"] == 75)].iloc[0]
    assert r["freq_bins"] == 513
    assert r["time_frames_1s"] == (25_000 - 1024) // 256 + 1 == 94
    assert r["bin_spacing_hz"] == pytest.approx(25_000 / 1024, abs=1e-3)
    assert r["nyquist_hz"] == 12_500
    on = pd.read_csv(PART4A_DIR / "stft_candidate_grid.csv")
    assert len(on) == len(g)          # 4 datasets x 5 n_fft x 2 hops


# ---------------------------------------------------------------------------
# no resizing / no image processing in the model-domain module
# ---------------------------------------------------------------------------

def test_no_image_resize_or_rgb_in_design_module():
    src = (REPO_ROOT / "src" / "methodology_v2"
           / "part4a_repdesign.py").read_text()
    for banned in ("PIL", "cv2", "skimage", "imresize", "Resize",
                   "interpolate", "jpeg", "JPEG", "rgb", "RGB"):
        assert banned not in src, f"model-domain module contains {banned}"


# ---------------------------------------------------------------------------
# HIT boundary audit
# ---------------------------------------------------------------------------

@needs_study
def test_hit_boundary_positions_and_train_scope():
    audit = pd.read_csv(PART4A_DIR / "hit_boundary_audit.csv")
    man = pd.read_csv(PART3B_DIR / "window_manifest_fold_1.csv")
    man = man.set_index("window_id")
    for _, a in audit.head(20).iterrows():
        w = man.loc[a["window_id"]]
        assert w["split"] == "train" and w["dataset"] == "HIT"
        bounds = [int(b) - int(w["start_sample"]) for b in
                  str(w["fragment_boundaries_crossed"]).split(",")]
        assert a["boundary_sample_in_window"] in bounds
    # audit conclusion sanity: no systematic broadband joint artefacts
    assert audit["max_frame_energy_zscore"].max() < 4.0
    assert audit["jump_ratio"].max() < 3.0


# ---------------------------------------------------------------------------
# determinism of the numeric study tables
# ---------------------------------------------------------------------------

@needs_study
def test_numeric_tables_deterministic(tmp_path):
    from src.methodology_v2.part3b_reader import read_window
    from src.methodology_v2.part4a_repdesign import run_numeric_studies
    run_numeric_studies(tmp_path, read_window)
    for name in ("part4a_development_windows.csv",
                 "stft_candidate_grid.csv", "stft_resolution_table.csv",
                 "frequency_coordinate_study.csv",
                 "stft_memory_estimates.csv", "stft_sharpness_metrics.csv",
                 "stft_dynamic_range_metrics.csv",
                 "hit_boundary_audit.csv"):
        assert (PART4A_DIR / name).read_bytes() \
            == (tmp_path / name).read_bytes(), f"{name} not deterministic"


# ---------------------------------------------------------------------------
# bounded figures + forbidden work
# ---------------------------------------------------------------------------

@needs_study
def test_figures_bounded_and_no_full_dataset_generation():
    figs = list((PART4A_DIR / "figures").glob("*.png"))
    assert 10 <= len(figs) <= 30
    banned = {".npy", ".npz", ".pt", ".pth", ".h5"}
    for f in PART4A_DIR.rglob("*"):
        assert f.suffix.lower() not in banned, \
            f"tensor artefact written: {f}"


def test_part4a_imports_no_model_or_training_code():
    from conftest import assert_imports_clean
    assert_imports_clean(["src.methodology_v2.part4a_repdesign"])
