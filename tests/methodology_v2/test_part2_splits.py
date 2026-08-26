"""Automated tests for the methodology_v2 Part-2 frozen split protocol."""
from __future__ import annotations

import inspect
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.methodology_v2 import FORBIDDEN_IMPORTS  # noqa: E402
from src.methodology_v2 import part2_protocol as P  # noqa: E402
from src.methodology_v2.part2_builder import (MAFAULDA_CLASSES,  # noqa: E402
                                              PART2_DIR, Part2ProtocolError,
                                              _hit_acceptance,
                                              _mafaulda_acceptance,
                                              build_jnu_rows,
                                              verify_frozen_hashes,
                                              write_outputs)

needs_folds = pytest.mark.skipif(
    not (PART2_DIR / "global_fold_1.csv").exists(),
    reason="run scripts/methodology_v2/run_part2_splits.py first")

# Approved rotation tables, INDEPENDENTLY retyped from the frozen
# instruction (not imported from the protocol module) so a silent edit of
# part2_protocol.py cannot pass unnoticed.
EXPECTED_CWRU_ROTATION = {
    1: {"train": {"IR007", "B014", "OR021"},
        "validation": {"IR014", "B021", "OR007"},
        "test": {"IR021", "B007", "OR014"}},
    2: {"train": {"IR014", "B021", "OR007"},
        "validation": {"IR021", "B007", "OR014"},
        "test": {"IR007", "B014", "OR021"}},
    3: {"train": {"IR021", "B007", "OR014"},
        "validation": {"IR007", "B014", "OR021"},
        "test": {"IR014", "B021", "OR007"}},
}
EXPECTED_JNU_ROTATION = {
    1: {"train": {"A", "B", "C"}, "validation": {"D"}, "test": {"E"}},
    2: {"train": {"B", "C", "D"}, "validation": {"E"}, "test": {"A"}},
    3: {"train": {"C", "D", "E"}, "validation": {"A"}, "test": {"B"}},
}


@pytest.fixture(scope="module")
def folds() -> dict[int, pd.DataFrame]:
    return {k: pd.read_csv(PART2_DIR / f"global_fold_{k}.csv")
            for k in (1, 2, 3)}


# ---------------------------------------------------------------------------
# global
# ---------------------------------------------------------------------------

@needs_folds
def test_exactly_three_folds_with_full_assignment(folds):
    assert sorted(folds) == [1, 2, 3]
    assert not (PART2_DIR / "global_fold_4.csv").exists()
    for k, df in folds.items():
        assert df["fold_id"].eq(k).all()
        assert df.groupby("dataset").size().to_dict() == {
            "CWRU": 52, "JNU": 108, "HIT": 134, "MAFAULDA": 1951}
        assert (df["methodology_version"] == P.METHODOLOGY_VERSION).all()
        usable = df[df["is_usable"] == True]  # noqa: E712
        assert set(usable["split"]) == {"train", "validation", "test"}
        assert (df.loc[df["is_usable"] == False, "split"]  # noqa: E712
                == "guard").all()


@needs_folds
def test_labels_match_part1_registry(folds):
    valid = {
        "CWRU": {"IR007", "IR014", "IR021", "B007", "B014", "B021",
                 "OR007@3", "OR007@6", "OR007@12", "OR014@6",
                 "OR021@3", "OR021@6", "OR021@12"},
        "JNU": {"n", "ib", "ob", "tb"},
        "HIT": {"0", "1", "2"},
        "MAFAULDA": set(MAFAULDA_CLASSES),
    }
    for df in folds.values():
        for ds, lab in valid.items():
            got = set(df.loc[df["dataset"] == ds, "original_label"]
                      .astype(str))
            assert got <= lab, f"{ds}: unexpected labels {got - lab}"


@needs_folds
def test_rerun_reproduces_byte_identical_manifests(tmp_path):
    write_outputs(tmp_path)
    for name in ["global_fold_1.csv", "global_fold_2.csv",
                 "global_fold_3.csv", "test_identity_fold_1.csv",
                 "test_identity_fold_2.csv", "test_identity_fold_3.csv",
                 "split_protocol.json", "split_hashes.csv",
                 "rejected_split_seeds.json", "fold_statistics.json"]:
        a = (PART2_DIR / name).read_bytes()
        b = (tmp_path / name).read_bytes()
        assert a == b, f"{name} not byte-identical on rerun"


@needs_folds
def test_seal_verifies_and_fails_closed(tmp_path):
    verify_frozen_hashes(PART2_DIR)  # must pass on the frozen artefacts

    tampered = tmp_path / "tampered"
    shutil.copytree(PART2_DIR, tampered)
    f = tampered / "test_identity_fold_1.csv"
    f.write_bytes(f.read_bytes().replace(b"test", b"tset", 1))
    with pytest.raises(Part2ProtocolError, match="FROZEN MANIFEST CHANGED"):
        verify_frozen_hashes(tampered)


# ---------------------------------------------------------------------------
# CWRU
# ---------------------------------------------------------------------------

@needs_folds
def test_cwru_subset_and_exclusions(folds):
    for df in folds.values():
        cw = df[df["dataset"] == "CWRU"]
        assert len(cw) == 52
        assert not (cw["original_label"] == "Normal").any()
        assert not cw["original_label"].str.contains("028").any()
        assert cw["source_file"].str.startswith("data/raw_cwru_48k/").all()
        assert (cw["sampling_rate_hz"] == 48000).all()
        assert cw["group_id"].nunique() == 9


@needs_folds
def test_cwru_specimen_integrity_and_class_coverage(folds):
    for df in folds.values():
        cw = df[df["dataset"] == "CWRU"]
        per_group = cw.groupby("group_id")
        assert (per_group["split"].nunique() == 1).all()
        for g, grp in per_group:
            assert set(grp["load"]) == {"0hp", "1hp", "2hp", "3hp"}, g
        # OR clock positions never cross partitions
        for spec in ("OR007", "OR021"):
            pos = cw[cw["original_label"].str.startswith(spec + "@")]
            assert pos["split"].nunique() == 1, spec
        for sp in ("train", "validation", "test"):
            types = set(cw.loc[cw["split"] == sp, "fault_type"])
            assert types == {"inner_race", "ball", "outer_race"}, sp


@needs_folds
def test_cwru_latin_rotation_matches_approved_table(folds):
    roles: dict[str, dict[int, str]] = {}
    for k, df in folds.items():
        cw = df[df["dataset"] == "CWRU"]
        spec_split = (cw.assign(spec=cw["group_id"].str
                                .removeprefix("cwru48k_"))
                      .groupby("spec")["split"].first().to_dict())
        for sp in ("train", "validation", "test"):
            got = {s for s, v in spec_split.items() if v == sp}
            assert got == EXPECTED_CWRU_ROTATION[k][sp], (k, sp, got)
        for s, v in spec_split.items():
            roles.setdefault(s, {})[k] = v
    # every specimen appears exactly once in each role across the 3 folds
    for s, by_fold in roles.items():
        assert sorted(by_fold.values()) == ["test", "train", "validation"], s


# ---------------------------------------------------------------------------
# JNU
# ---------------------------------------------------------------------------

@needs_folds
def test_jnu_block_structure_and_rotation(folds):
    for k, df in folds.items():
        jn = df[df["dataset"] == "JNU"]
        assert jn["recording_id"].nunique() == 12
        for rec, grp in jn.groupby("recording_id"):
            blocks = grp[grp["is_usable"] == True]  # noqa: E712
            guards = grp[grp["is_usable"] == False]  # noqa: E712
            assert list(blocks["temporal_block_id"]) == list("ABCDE")
            assert len(guards) == 4
            assert (guards["split"] == "guard").all()
            b = blocks.sort_values("temporal_start_sample")
            starts = b["temporal_start_sample"].tolist()
            ends = b["temporal_end_sample"].tolist()
            n = ends[-1]
            assert starts == [i * n // 5 for i in range(5)]
            assert ends == [(i + 1) * n // 5 for i in range(5)]
            # contiguous, non-overlapping
            assert all(ends[i] == starts[i + 1] for i in range(4))
            # guards anchored exactly at the internal boundaries
            assert sorted(guards["temporal_start_sample"]) == starts[1:]
            for sp, blks in EXPECTED_JNU_ROTATION[k].items():
                got = set(blocks.loc[blocks["split"] == sp,
                                     "temporal_block_id"])
                assert got == blks, (k, rec, sp, got)


@needs_folds
def test_jnu_class_speed_coverage_every_partition(folds):
    for df in folds.values():
        jn = df[(df["dataset"] == "JNU")
                & (df["is_usable"] == True)]  # noqa: E712
        for sp in ("train", "validation", "test"):
            part = jn[jn["split"] == sp]
            combos = set(zip(part["original_label"],
                             part["rpm"].astype(int)))
            assert combos == {(c, s) for c in ("n", "ib", "ob", "tb")
                              for s in (600, 800, 1000)}, sp


def test_jnu_builder_is_deterministic_no_rng():
    src = inspect.getsource(build_jnu_rows)
    assert "default_rng" not in src and "random" not in src


# ---------------------------------------------------------------------------
# HIT
# ---------------------------------------------------------------------------

@needs_folds
def test_hit_full_release_only_and_group_integrity(folds):
    for df in folds.values():
        h = df[df["dataset"] == "HIT"]
        assert len(h) == 134
        assert h["source_file"].str.startswith(
            "data/raw_hit/gdrive_full/").all()
        # the GitHub windowed release must never be a source or authority
        assert not h["source_file"].str.contains(
            "raw_hit/HIT-dataset").any()
        assert (h.groupby("group_id")["split"].nunique() == 1).all()
        assert h["group_id"].nunique() == 134


@needs_folds
def test_hit_structural_acceptance_criteria_hold(folds):
    part1 = pd.read_csv(REPO_ROOT / "methodology_v2" / "part1_audit"
                        / "recording_manifest.csv")
    sub = (part1[part1["dataset"] == "HIT"]
           .sort_values("recording_id").reset_index(drop=True))
    for df in folds.values():
        h = df[df["dataset"] == "HIT"]
        assign = dict(zip(h["recording_id"], h["split"]))
        assert _hit_acceptance(assign, sub) == []
        # target proportions realised
        counts = h.groupby("split")["group_id"].nunique().to_dict()
        assert counts == {"train": 94, "validation": 20, "test": 20}


# ---------------------------------------------------------------------------
# MaFaulDa
# ---------------------------------------------------------------------------

@needs_folds
def test_mafaulda_group_integrity_and_units(folds):
    for df in folds.values():
        mf = df[df["dataset"] == "MAFAULDA"]
        assert len(mf) == 1951
        assert (mf.groupby("group_id")["split"].nunique() == 1).all()
        normal = mf[mf["original_label"] == "normal"]
        # Normal exception: unit == recording (weaker, documented)
        assert (normal["group_id"] == normal["recording_id"]).all()
        assert (normal["grouping_type"] == "recording_normal_exception").all()
        faults = mf[mf["original_label"] != "normal"]
        assert (faults["grouping_type"] == "fault_configuration").all()
        assert faults["group_id"].nunique() == 41


@needs_folds
def test_mafaulda_class_coverage_and_taxonomy_untouched(folds):
    for df in folds.values():
        mf = df[df["dataset"] == "MAFAULDA"]
        assert set(mf["original_label"]) == set(MAFAULDA_CLASSES)
        assert not mf["original_label"].str.contains("inner").any()
        for sp in ("train", "validation", "test"):
            part = mf[mf["split"] == sp]
            assert set(part["original_label"]) == set(MAFAULDA_CLASSES), sp
        # frozen per-stratum allocation over group units
        alloc = (mf.groupby(["original_label", "split"])["group_id"]
                 .nunique().unstack(fill_value=0))
        for cls in MAFAULDA_CLASSES:
            c = {"normal": 49, "imbalance": 7,
                 "vertical-misalignment": 6}.get(cls, 4)
            n_val = max(1, round(0.15 * c))
            n_test = max(1, round(0.15 * c))
            assert alloc.loc[cls, "validation"] == n_val, cls
            assert alloc.loc[cls, "test"] == n_test, cls
            assert alloc.loc[cls, "train"] == c - n_val - n_test, cls


@needs_folds
def test_mafaulda_structural_acceptance_criteria_hold(folds):
    part1 = pd.read_csv(REPO_ROOT / "methodology_v2" / "part1_audit"
                        / "recording_manifest.csv")
    import numpy as np
    sub = (part1[part1["dataset"] == "MAFAULDA"]
           .sort_values("recording_id").reset_index(drop=True)).copy()
    sub["unit_id"] = np.where(sub["original_label"] == "normal",
                              sub["recording_id"], sub["group_id_candidate"])
    for df in folds.values():
        mf = df[df["dataset"] == "MAFAULDA"]
        unit_split = dict(zip(mf["group_id"], mf["split"]))
        assert _mafaulda_acceptance(unit_split, sub) == []


# ---------------------------------------------------------------------------
# seed governance and training-code guard
# ---------------------------------------------------------------------------

@needs_folds
def test_predeclared_seeds_recorded_and_rejections_file_exists():
    proto = json.load(open(PART2_DIR / "split_protocol.json"))
    assert proto["hit"]["predeclared_seeds"] == {"1": 101, "2": 102,
                                                 "3": 103}
    assert proto["mafaulda"]["predeclared_seeds"] == {"1": 201, "2": 202,
                                                      "3": 203}
    rej = json.load(open(PART2_DIR / "rejected_split_seeds.json"))
    assert "rejections" in rej
    for r in rej["rejections"]:
        assert {"dataset", "fold_id", "seed", "reasons",
                "replacement_seed"} <= set(r)
    # seeds actually used are the predeclared ones unless rejections exist
    if not rej["rejections"]:
        assert proto["seeds_used"] == {
            "1": {"HIT": 101, "MAFAULDA": 201},
            "2": {"HIT": 102, "MAFAULDA": 202},
            "3": {"HIT": 103, "MAFAULDA": 203}}


def test_part2_imports_no_training_or_windowing_code():
    from conftest import assert_imports_clean
    assert_imports_clean(["src.methodology_v2.part2_protocol",
                          "src.methodology_v2.part2_builder"])
    # usage rules encode the S0/S1 fairness contract
    assert "identical frozen fold assignments" in \
        P.USAGE_RULES["shared_rule"]
