"""Focused tests for the Part-3A input design study (audit-only stage)."""
from __future__ import annotations

import json
import sys
from fractions import Fraction
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.methodology_v2 import FORBIDDEN_IMPORTS  # noqa: E402
from src.methodology_v2.part2_builder import (PART2_DIR,  # noqa: E402
                                              verify_frozen_hashes)
from src.methodology_v2.part3a_study import (PART3A_DIR,  # noqa: E402
                                             DURATIONS_S, NATIVE_RATE,
                                             jnu_usable_block_s, n_windows,
                                             rate_study_df, rotations,
                                             write_study_tables)

needs_study = pytest.mark.skipif(
    not (PART3A_DIR / "window_count_estimates.csv").exists(),
    reason="run scripts/methodology_v2/run_part3a.py first")


# ---------------------------------------------------------------------------
# frozen-artefact protection
# ---------------------------------------------------------------------------

def test_part2_master_hash_still_matches():
    verify_frozen_hashes()  # fails closed on any byte change


@needs_study
def test_no_windows_or_tensors_written():
    banned = {".npy", ".npz", ".pt", ".pth", ".h5", ".hdf5", ".zarr"}
    files = list(PART3A_DIR.rglob("*"))
    assert files, "study output directory is empty"
    for f in files:
        assert f.suffix.lower() not in banned, f"tensor artefact: {f}"


def test_part3a_imports_no_training_or_preprocessing_code():
    from conftest import assert_imports_clean
    assert_imports_clean(["src.methodology_v2.part3a_study",
                          "src.methodology_v2.part3a_diagnostics"])


# ---------------------------------------------------------------------------
# calculation correctness
# ---------------------------------------------------------------------------

def test_window_count_equation():
    assert n_windows(10.0, 1.0, 0.0) == 10
    assert n_windows(10.0, 1.0, 0.5) == 19
    assert n_windows(1.0, 1.0, 0.0) == 1
    assert n_windows(0.9, 1.0, 0.0) == 0
    assert n_windows(5.0, 2.0, 0.0) == 2
    assert n_windows(5.0, 2.0, 0.5) == 4
    assert n_windows(1.3289, 2.0, 0.0) == 0  # shortest CWRU load-0 vs 2 s


def test_rotation_equation():
    assert rotations(600, 1.0) == pytest.approx(10.0)
    assert rotations(600, 0.25) == pytest.approx(2.5)
    assert rotations(1797, 1.0) == pytest.approx(29.95)
    assert rotations(200, 1.0) == pytest.approx(3.333, abs=1e-3)  # HIT rel


def test_rate_conversion_ratios():
    df = rate_study_df()

    def ratio(ds, rate):
        r = df[(df["dataset"] == ds) & (df["target_rate_hz"] == rate)]
        return Fraction(int(r["ratio_up_L"].iloc[0]),
                        int(r["ratio_down_M"].iloc[0]))
    assert ratio("CWRU", 25_000) == Fraction(25, 48)
    assert ratio("JNU", 25_000) == Fraction(1, 2)
    assert ratio("MAFAULDA", 25_000) == Fraction(1, 2)
    assert ratio("HIT", 25_000) == Fraction(1, 1)
    assert ratio("CWRU", 24_000) == Fraction(1, 2)
    assert ratio("HIT", 24_000) == Fraction(24, 25)
    assert ratio("HIT", 32_000) == Fraction(32, 25)
    up = df[(df["dataset"] == "HIT") & (df["target_rate_hz"] == 32_000)]
    assert "UPSAMPLE" in up["operation"].iloc[0]
    # no other candidate upsamples any dataset
    others = df[df["target_rate_hz"] != 32_000]
    assert not others["operation"].str.contains("UPSAMPLE").any()


def test_jnu_guard_satisfies_window_span():
    for w in DURATIONS_S:
        g = w  # frozen minimum rule
        assert g >= w
        # internal fault block: nominal 2.002 s loses G total
        internal = jnu_usable_block_s(2.002, w, True, True)
        assert internal == pytest.approx(2.002 - w)
        edge = jnu_usable_block_s(2.002, w, False, True)
        assert edge == pytest.approx(2.002 - w / 2)
    # 2.0 s window leaves internal fault blocks unable to hold one window
    assert jnu_usable_block_s(2.002, 2.0, True, True) < 2.0
    # 1.0 s window leaves exactly one window per internal fault block
    assert n_windows(jnu_usable_block_s(2.002, 1.0, True, True),
                     1.0, 0.0) == 1


# ---------------------------------------------------------------------------
# sealed-data enforcement
# ---------------------------------------------------------------------------

def test_diagnostics_guard_refuses_non_train_recordings():
    from src.methodology_v2.part3a_diagnostics import (SealedDataError,
                                                       _assert_train_fold1)
    fold1 = pd.read_csv(PART2_DIR / "global_fold_1.csv")
    test_rec = fold1[(fold1["split"] == "test")
                     & (fold1["dataset"] == "HIT")].iloc[0]["recording_id"]
    with pytest.raises(SealedDataError):
        _assert_train_fold1(fold1, test_rec)
    with pytest.raises(SealedDataError):
        _assert_train_fold1(fold1, "not_a_recording")


@needs_study
def test_diagnostics_read_only_fold1_train_data():
    diag = json.load(open(PART3A_DIR / "fold1_train_diagnostics.json"))
    fold1 = pd.read_csv(PART2_DIR / "global_fold_1.csv")
    for ds in ("CWRU", "HIT", "MAFAULDA"):
        train_ids = set(fold1[(fold1["dataset"] == ds)
                              & (fold1["split"] == "train")]["recording_id"])
        read = set(diag["recordings_read"][ds])
        assert read <= train_ids, f"{ds}: read outside fold-1 train"
    # JNU is sealed at temporal-region level: recordings appear in all
    # splits, but only train blocks A-C ([0, 3N/5)) may be read
    jn = fold1[fold1["dataset"] == "JNU"]
    train_blocks = set(jn[jn["split"] == "train"]["temporal_block_id"])
    assert train_blocks == {"A", "B", "C"}
    assert set(diag["recordings_read"]["JNU"]) <= set(jn["recording_id"])
    assert "fold 1 train" in diag["scope"].lower() \
        or "FOLD 1 TRAIN" in diag["scope"]


# ---------------------------------------------------------------------------
# real study-table invariants + determinism
# ---------------------------------------------------------------------------

@needs_study
def test_study_tables_deterministic(tmp_path):
    write_study_tables(tmp_path)
    for name in ["channel_census.csv", "sampling_rate_study.csv",
                 "window_duration_study.csv", "window_count_estimates.csv",
                 "jnu_guard_study.csv", "cwru_load0_study.csv"]:
        assert (PART3A_DIR / name).read_bytes() == \
            (tmp_path / name).read_bytes(), f"{name} not deterministic"


@needs_study
def test_window_counts_match_hand_calculation():
    df = pd.read_csv(PART3A_DIR / "window_count_estimates.csv")
    std = df[df["counting_basis"] == "standard"]

    def val(fold, ds, sp, w, ov):
        r = std[(std["fold_id"] == fold) & (std["dataset"] == ds)
                & (std["split"] == sp) & (std["window_s"] == w)
                & (std["overlap_pct"] == ov)]
        return int(r["n_windows"].iloc[0])
    # MaFaulDa: all recordings 5.0 s -> 5 windows @1s/0%, 9 @1s/50%
    assert val(1, "MAFAULDA", "validation", 1.0, 0) == 412 * 5
    assert val(1, "MAFAULDA", "train", 1.0, 50) == 1092 * 9
    # HIT: 14.7456 s -> 14 @1s/0%, 28 @1s/50% (concatenated basis)
    assert val(1, "HIT", "test", 1.0, 0) == 20 * 14
    assert val(1, "HIT", "train", 1.0, 50) == 94 * 28
    # JNU fold-1 @1s: train(A,B,C)@50% = 9*(2+1+1) + 3*(10+9+9) = 120
    assert val(1, "JNU", "train", 1.0, 50) == 120
    assert val(1, "JNU", "validation", 1.0, 0) == 9 * 1 + 3 * 5
    assert val(1, "JNU", "test", 1.0, 0) == 9 * 1 + 3 * 5
    # HIT series-constrained basis: 1.0 s window cannot fit in 0.8192 s
    sc = df[(df["counting_basis"] == "hit_series_constrained")
            & (df["fold_id"] == 1) & (df["split"] == "test")
            & (df["window_s"] == 1.0) & (df["overlap_pct"] == 0)]
    assert int(sc["n_windows"].iloc[0]) == 0


@needs_study
def test_cwru_load0_analysis():
    df = pd.read_csv(PART3A_DIR / "cwru_load0_study.csv")
    df = df.set_index("window_s")
    # Part 1 counted 14 short load-0 recordings across the 56-recording
    # audited set; the retained 3-class benchmark excludes Healthy, whose
    # Normal_0 file was the 14th -> 13 fault load-0 recordings remain.
    assert (df["n_load0_recordings"] == 13).all()
    assert df.loc[2.0, "n_load0_zero_windows"] == 1   # official 174 only
    assert df.loc[1.0, "n_load0_zero_windows"] == 0
    assert df.loc[0.5, "n_load0_zero_windows"] == 0
    assert bool(df.loc[2.0, "eliminates_any_recording"])
    assert not bool(df.loc[1.0, "eliminates_any_recording"])
