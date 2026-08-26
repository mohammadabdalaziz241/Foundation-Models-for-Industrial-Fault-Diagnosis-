"""Focused tests for the CWRU grouping re-check (bounded Part-1 addendum)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.methodology_v2 import FORBIDDEN_IMPORTS  # noqa: E402
from src.methodology_v2.registry import OUTPUT_DIR  # noqa: E402

needs_manifest = pytest.mark.skipif(
    not (OUTPUT_DIR / "recording_manifest.csv").exists(),
    reason="run Part 1 audit first")


@pytest.fixture(scope="module")
def recheck():
    from src.methodology_v2.grouping_recheck import build_recheck
    return build_recheck()


@needs_manifest
def test_retained_subset_is_exactly_56_recordings(recheck):
    df, _ = recheck
    assert len(df) == 56
    assert (df.groupby("class").size().to_dict()
            == {"Ball": 12, "Healthy": 4, "InnerRace": 12, "OuterRace": 28})
    # no 0.028" and no 12 kHz family
    assert not (df["diameter_mil"] == 28).any()
    assert df["load_hp"].isin([0, 1, 2, 3]).all()


@needs_manifest
def test_option_group_counts(recheck):
    _, stats = recheck
    assert stats["A_recording"]["n_groups"] == 56
    assert stats["B_specimen_x_load"]["n_groups"] == 56  # == A in 48k family
    assert stats["C1_installation"]["n_groups"] == 14
    assert stats["C2_specimen"]["n_groups"] == 10
    assert stats["C2_specimen"]["groups_per_class"] == {
        "Ball": 3, "Healthy": 1, "InnerRace": 3, "OuterRace": 3}
    # only C2 closes the physical-specimen leakage route
    assert stats["A_recording"]["same_physical_specimen_can_cross_partitions"]
    assert stats["B_specimen_x_load"][
        "same_physical_specimen_can_cross_partitions"]
    assert stats["C1_installation"][
        "same_physical_specimen_can_cross_partitions"]
    assert not stats["C2_specimen"][
        "same_physical_specimen_can_cross_partitions"]


@needs_manifest
def test_official_numbering_quirks_verified(recheck):
    df, _ = recheck
    ir014_0 = df[(df["class"] == "InnerRace") & (df["diameter_mil"] == 14)
                 & (df["load_hp"] == 0)].iloc[0]
    # official download number 174, internal variables X173_* — the known
    # numbering quirk; both identities must be carried
    assert ir014_0["official_download_number"] == 174
    assert ir014_0["internal_variable_id"] == "X173"
    ir021_3 = df[(df["class"] == "InnerRace") & (df["diameter_mil"] == 21)
                 & (df["load_hp"] == 3)].iloc[0]
    assert ir021_3["official_download_number"] == 217  # 216 skipped

    healthy = df[df["class"] == "Healthy"]
    assert sorted(healthy["official_download_number"]) == [97, 98, 99, 100]
    assert healthy["group_C2_specimen"].nunique() == 1  # single specimen


@needs_manifest
def test_numeric_sanity(recheck):
    df, _ = recheck
    assert df["rpm_in_file"].between(1700, 1800).all()
    assert (df["duration_s"] > 0).all()
    assert (df["n_samples"] > 0).all()
    assert df["recording_id"].is_unique


def test_recheck_imports_no_training_code():
    from conftest import assert_imports_clean
    assert_imports_clean(["src.methodology_v2.grouping_recheck"])
