"""Automated tests for Part 4C — sealed N2 normalizers and the final
representation reader."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.methodology_v2 import FORBIDDEN_IMPORTS  # noqa: E402
from src.methodology_v2.integrity import sha256_file  # noqa: E402
from src.methodology_v2.part2_builder import verify_frozen_hashes  # noqa: E402
from src.methodology_v2.part3b_windows import (PART3B_DIR,  # noqa: E402
                                               verify_part3b_hashes)
from src.methodology_v2.part4b_freeze import (TF_CONFIG,  # noqa: E402
                                              rep_of)
from src.methodology_v2.part4c_normalizers import (  # noqa: E402
    DATASETS, FOLD_IDS, NORM_DIR, PART4C_DIR, Part4CError,
    StreamingBinStats, assert_train_row, deterministic_savez,
    load_normalizer, train_windows, verify_part4c_hashes)

needs_fit = pytest.mark.skipif(
    not (PART4C_DIR / "normalizer_hashes.csv").exists(),
    reason="run scripts/methodology_v2/run_part4c.py first")


# ---------------------------------------------------------------------------
# upstream + seal
# ---------------------------------------------------------------------------

def test_upstream_seals_intact():
    verify_frozen_hashes()
    verify_part3b_hashes()


@needs_fit
def test_part4c_seal_verifies_and_fails_closed(tmp_path):
    verify_part4c_hashes()
    tampered = tmp_path / "t"
    shutil.copytree(PART4C_DIR, tampered,
                    ignore=shutil.ignore_patterns("figures"))
    f = tampered / "normalizers" / "fold_1" / "cwru.npz"
    data = f.read_bytes()
    f.write_bytes(data[:-1] + bytes([data[-1] ^ 0xFF]))  # flip last byte
    with pytest.raises(Part4CError, match="fail closed"):
        verify_part4c_hashes(tampered)


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def test_streaming_agrees_with_direct_reference():
    rng = np.random.default_rng(0)
    reps = [np.abs(rng.normal(loc=2, scale=3, size=(50, 17)))
            for _ in range(7)]
    st = StreamingBinStats(17)
    for r in reps:
        st.add_window(r)
    out = st.finalize()
    ref = np.concatenate(reps, axis=0)
    assert np.allclose(out["mean"], ref.mean(axis=0), atol=1e-12)
    assert np.allclose(out["std_raw"], ref.std(axis=0), atol=1e-10)
    assert out["n_frames"] == 350 and out["n_windows"] == 7


def test_streaming_matches_reference_on_real_train_windows():
    from src.methodology_v2.part3b_reader import read_window
    sub = train_windows(1, "JNU").head(12)
    st = StreamingBinStats(TF_CONFIG["JNU"][0] // 2 + 1)
    reps = []
    for _, row in sub.iterrows():
        rep = rep_of(read_window(row), "JNU")
        reps.append(rep)
        st.add_window(rep)
    out = st.finalize()
    ref = np.concatenate(reps, axis=0)
    assert np.allclose(out["mean"], ref.mean(axis=0), atol=1e-10)
    assert np.allclose(out["std_raw"], ref.std(axis=0), atol=1e-8)


def test_fitting_guard_rejects_non_train_and_wrong_fold():
    man = pd.read_csv(PART3B_DIR / "window_manifest_fold_1.csv")
    test_row = man[man["split"] == "test"].iloc[0]
    with pytest.raises(Part4CError, match="TRAIN-only"):
        assert_train_row(test_row, 1)
    val_row = man[man["split"] == "validation"].iloc[0]
    with pytest.raises(Part4CError):
        assert_train_row(val_row, 1)
    train_row = man[man["split"] == "train"].iloc[0]
    with pytest.raises(Part4CError):
        assert_train_row(train_row, 2)   # fold isolation
    assert_train_row(train_row, 1)       # legal


def test_std_floor_rule():
    st = StreamingBinStats(4)
    rep = np.ones((100, 4))
    rep[:, 2] = np.linspace(0, 1, 100)   # only bin 2 varies
    st.add_window(rep)
    out = st.finalize()
    assert set(out["floored_bins"]) == {0, 1, 3}
    assert (out["std_denominator"][[0, 1, 3]] == 1.0).all()
    assert out["std_denominator"][2] == out["std_raw"][2] > 1e-3


# ---------------------------------------------------------------------------
# fitted artifacts
# ---------------------------------------------------------------------------

@needs_fit
def test_twelve_normalizers_fold_and_dataset_isolated():
    reg = pd.read_csv(PART4C_DIR / "normalizer_registry.csv")
    assert len(reg) == 12
    assert set(zip(reg["fold"], reg["dataset"])) == {
        (f, d) for f in FOLD_IDS for d in DATASETS}
    man_hashes = reg.groupby("fold")["train_manifest_sha256"].nunique()
    assert (man_hashes == 1).all()
    # fold isolation: same dataset, different folds -> different stats
    for ds in DATASETS:
        m1 = load_normalizer(1, ds)["mean"]
        m2 = load_normalizer(2, ds)["mean"]
        m3 = load_normalizer(3, ds)["mean"]
        assert not np.array_equal(m1, m2)
        assert not np.array_equal(m2, m3)
    # dataset isolation within a fold: grids sized per dataset
    for f in FOLD_IDS:
        assert load_normalizer(f, "HIT")["mean"].size == 257
        for ds in ("CWRU", "JNU", "MAFAULDA"):
            assert load_normalizer(f, ds)["mean"].size == 513


@needs_fit
def test_normalizer_counts_match_manifests():
    reg = pd.read_csv(PART4C_DIR / "normalizer_registry.csv")
    for _, r in reg.iterrows():
        man = pd.read_csv(PART3B_DIR
                          / f"window_manifest_fold_{r['fold']}.csv")
        tr = man[(man["dataset"] == r["dataset"])
                 & (man["split"] == "train")]
        assert r["n_train_windows"] == len(tr)
        frames = {"CWRU": 184, "JNU": None, "HIT": 192,
                  "MAFAULDA": 192}[r["dataset"]]
        if r["dataset"] == "JNU":
            continue  # JNU frame counts vary per block length
        assert r["n_frames"] == len(tr) * frames


@needs_fit
def test_frequency_grids_physical():
    for f in FOLD_IDS:
        hit = load_normalizer(f, "HIT")["frequency_hz"]
        jnu = load_normalizer(f, "JNU")["frequency_hz"]
        cwru = load_normalizer(f, "CWRU")["frequency_hz"]
        assert hit[1] == pytest.approx(48.828125)
        assert jnu[1] == pytest.approx(48.828125)
        assert cwru[1] == pytest.approx(46.875)
        assert hit[-1] == pytest.approx(12_500)
        assert jnu[-1] == pytest.approx(25_000)
        assert cwru[-1] == pytest.approx(24_000)


def test_deterministic_npz_bytes(tmp_path):
    arrays = {"a": np.arange(10.0), "b": np.ones(3)}
    p1, p2 = tmp_path / "x1.npz", tmp_path / "x2.npz"
    deterministic_savez(p1, arrays)
    deterministic_savez(p2, arrays)
    assert p1.read_bytes() == p2.read_bytes()
    with np.load(p1) as z:
        assert np.array_equal(z["a"], arrays["a"])


# ---------------------------------------------------------------------------
# representation reader
# ---------------------------------------------------------------------------

@needs_fit
def test_reader_exact_n2_math_and_metadata():
    from src.methodology_v2.part3b_reader import read_window
    from src.methodology_v2.part4c_reader import get_representation
    man = pd.read_csv(PART3B_DIR / "window_manifest_fold_1.csv")
    for ds, bins, frames in (("CWRU", 513, 184), ("JNU", 513, 192),
                             ("HIT", 257, 192), ("MAFAULDA", 513, 192)):
        row = man[(man["dataset"] == ds)
                  & (man["split"] == "train")].iloc[0]
        x, meta = get_representation(row["window_id"], 1)
        assert x.shape == (bins, frames)
        assert x.dtype == np.float32
        assert np.isfinite(x).all()
        # manual N2 reference
        norm = load_normalizer(1, ds)
        rep = rep_of(read_window(row), ds)
        ref = ((rep - norm["mean"]) / norm["std_denominator"]).T
        assert np.allclose(x, ref.astype(np.float32), atol=0)
        assert meta["frequency_hz"].size == bins
        assert meta["time_seconds"].size == frames
        hop, fs = meta["hop"], meta["sampling_rate_hz"]
        assert meta["time_seconds"][1] == pytest.approx(hop / fs)


@needs_fit
def test_reader_deterministic_and_fold_specific():
    from src.methodology_v2.part4c_reader import get_representation
    man = pd.read_csv(PART3B_DIR / "window_manifest_fold_1.csv")
    wid = man[(man["dataset"] == "HIT")
              & (man["split"] == "train")].iloc[0]["window_id"]
    a, _ = get_representation(wid, 1)
    b, _ = get_representation(wid, 1)
    assert np.array_equal(a, b)
    # same window under another fold's normalizer differs (fold-specific)
    man2 = pd.read_csv(PART3B_DIR / "window_manifest_fold_2.csv")
    if wid in set(man2["window_id"]):
        c, _ = get_representation(wid, 2)
        assert not np.array_equal(a, c)


@needs_fit
def test_valtest_access_never_updates_statistics():
    from src.methodology_v2.part4c_reader import get_representation
    before = {p.name: sha256_file(p) for p in NORM_DIR.rglob("*.npz")}
    man = pd.read_csv(PART3B_DIR / "window_manifest_fold_1.csv")
    for sp in ("validation", "test"):
        row = man[man["split"] == sp].iloc[0]
        x, meta = get_representation(row["window_id"], 1)
        assert np.isfinite(x).all()      # mechanical checks only
        assert meta["split"] == sp
    after = {p.name: sha256_file(p) for p in NORM_DIR.rglob("*.npz")}
    assert before == after


@needs_fit
def test_train_sanity_results_recorded():
    sanity = json.load(open(PART4C_DIR / "normalization_sanity.json"))
    assert len(sanity) == 12
    for s in sanity:
        assert s["scope"] == "TRAIN"
        # the exact 0-mean/1-std invariant holds over ALL train frames by
        # construction; sampled per-window means fluctuate around it, and
        # tiny samples (JNU: 1 window at step 200) fluctuate most
        assert abs(s["post_mean_avg"]) < 1.0
        assert 0.5 < s["post_std_avg"] < 1.6
        if s["n_sampled_windows"] >= 10:
            assert abs(s["post_mean_avg"]) < 0.35
        assert s["finite_pct"] == 100.0
    shapes = pd.read_csv(PART4C_DIR / "representation_shapes.csv")
    assert len(shapes) == 12
    assert (shapes["dtype"] == "float32").all()


# ---------------------------------------------------------------------------
# forbidden work
# ---------------------------------------------------------------------------

def test_no_model_or_image_work():
    from conftest import assert_imports_clean
    assert_imports_clean(["src.methodology_v2.part4c_normalizers",
                          "src.methodology_v2.part4c_reader"])
    for mod in ("part4c_normalizers.py", "part4c_reader.py"):
        src = (REPO_ROOT / "src" / "methodology_v2" / mod).read_text()
        for banned in ("PIL", "cv2", "skimage", "Resize", "interpolate",
                       "rgb", "RGB", "torch", "mask", "encoder",
                       "decoder", "optimizer"):
            assert banned not in src, f"{mod} contains {banned}"


@needs_fit
def test_no_precomputed_spectrogram_files():
    exts = {p.suffix for p in PART4C_DIR.rglob("*") if p.is_file()}
    # only normalizer npz artifacts + csv/yaml/json/md are allowed
    assert exts <= {".npz", ".csv", ".yaml", ".json", ".md"}
    n_npz = len(list(PART4C_DIR.rglob("*.npz")))
    assert n_npz == 12   # normalizers only — no cached spectrograms
