"""Automated tests for the Part-3B frozen window extraction."""
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
from src.methodology_v2 import part3b_protocol as P  # noqa: E402
from src.methodology_v2.part2_builder import (PART2_DIR,  # noqa: E402
                                              verify_frozen_hashes)
from src.methodology_v2.part3b_reader import (hit_reference_stream,  # noqa: E402
                                              read_window)
from src.methodology_v2.part3b_windows import (PART3B_DIR,  # noqa: E402
                                               build_all,
                                               verify_part3b_hashes)

needs_windows = pytest.mark.skipif(
    not (PART3B_DIR / "window_manifest_fold_1.csv").exists(),
    reason="run scripts/methodology_v2/run_part3b.py first")


@pytest.fixture(scope="module")
def manifests() -> dict[int, pd.DataFrame]:
    return {k: pd.read_csv(PART3B_DIR / f"window_manifest_fold_{k}.csv")
            for k in (1, 2, 3)}


# ---------------------------------------------------------------------------
# frozen protocol
# ---------------------------------------------------------------------------

def test_part2_seal_unchanged():
    verify_frozen_hashes()


@needs_windows
def test_part3b_seal_verifies():
    verify_part3b_hashes()


@needs_windows
def test_windows_are_exactly_one_second_native_rate(manifests):
    for df in manifests.values():
        assert (df["window_duration_seconds"] == 1.0).all()
        for ds, rate in P.EXPECTED_NATIVE_RATE.items():
            sub = df[df["dataset"] == ds]
            assert (sub["native_sampling_rate_hz"] == rate).all()
            assert ((sub["end_sample"] - sub["start_sample"])
                    == rate).all(), f"{ds}: wrong samples/window"
        # stride policy 50/0/0
        assert (df.loc[df["split"] == "train", "stride_seconds"]
                == 0.5).all()
        assert (df.loc[df["split"] != "train", "stride_seconds"]
                == 1.0).all()


@needs_windows
def test_window_ids_unique_and_deterministic(manifests, tmp_path):
    for k, df in manifests.items():
        assert df["window_id"].is_unique, f"fold {k}"
    res = build_all(tmp_path)
    for k in (1, 2, 3):
        a = (PART3B_DIR / f"window_manifest_fold_{k}.csv").read_bytes()
        b = (tmp_path / f"window_manifest_fold_{k}.csv").read_bytes()
        assert a == b, f"fold {k} manifest not byte-identical on rerun"
    for name in ("jnu_guards_1s.csv", "hit_logical_stream_manifest.csv",
                 "window_hashes.csv"):
        assert (PART3B_DIR / name).read_bytes() == \
            (tmp_path / name).read_bytes()


@needs_windows
def test_frozen_channels_only(manifests):
    for df in manifests.values():
        for ds, ch in P.CHANNELS.items():
            assert (df.loc[df["dataset"] == ds, "channel"] == ch).all()


@needs_windows
def test_no_partial_windows_and_valid_bounds(manifests):
    part1 = pd.read_csv(REPO_ROOT / "methodology_v2" / "part1_audit"
                        / "recording_manifest.csv")
    n = part1.set_index("recording_id")["n_samples"]
    for df in manifests.values():
        assert (df["start_sample"] >= 0).all()
        non_hit = df[df["dataset"] != "HIT"]
        limits = n.reindex(non_hit["recording_id"]).to_numpy()
        assert (non_hit["end_sample"].to_numpy() <= limits).all()
        hit = df[df["dataset"] == "HIT"]
        assert (hit["end_sample"]
                <= P.HIT_FRAGMENTS_PER_STREAM
                * P.HIT_FRAGMENT_SAMPLES).all()


# ---------------------------------------------------------------------------
# split integrity (Part-2 inheritance)
# ---------------------------------------------------------------------------

@needs_windows
def test_windows_inherit_part2_assignment_exactly(manifests):
    for k, df in manifests.items():
        p2 = pd.read_csv(PART2_DIR / f"global_fold_{k}.csv")
        # CWRU/HIT/MaFaulDa: group -> exactly one split, matching Part 2
        for ds in ("CWRU", "HIT", "MAFAULDA"):
            w = df[df["dataset"] == ds]
            assert (w.groupby("group_id")["split"].nunique() == 1).all()
            p2sub = p2[(p2["dataset"] == ds)]
            expect = p2sub.set_index("recording_id")["split"]
            got = w.groupby("recording_id")["split"].first()
            joined = expect.reindex(got.index)
            assert (joined == got).all(), f"fold {k} {ds} split mismatch"
        # JNU: window split must match its (recording, block) Part-2 row
        jn = df[df["dataset"] == "JNU"]
        p2j = p2[(p2["dataset"] == "JNU")
                 & (p2["is_usable"] == True)]  # noqa: E712
        expect = p2j.set_index(["recording_id",
                                "temporal_block_id"])["split"]
        got = jn.set_index(["recording_id",
                            "temporal_block_id"])["split"]
        assert (expect.reindex(got.index) == got.to_numpy()).all() or \
            (expect.reindex(got.index).to_numpy() == got.to_numpy()).all()


@needs_windows
def test_eval_windows_do_not_overlap(manifests):
    for df in manifests.values():
        ev = df[df["split"] != "train"]
        for (_, _), grp in ev.groupby(["recording_id",
                                       "temporal_block_id"], dropna=False):
            g = grp.sort_values("start_sample")
            assert (g["start_sample"].to_numpy()[1:]
                    >= g["end_sample"].to_numpy()[:-1]).all()
            # no duplicate source intervals
            assert not g.duplicated(["start_sample", "end_sample"]).any()


# ---------------------------------------------------------------------------
# CWRU
# ---------------------------------------------------------------------------

@needs_windows
def test_cwru_subset_and_load0(manifests):
    for df in manifests.values():
        cw = df[df["dataset"] == "CWRU"]
        assert not (cw["original_label"] == "Normal").any()
        assert not cw["original_label"].str.contains("028").any()
        assert cw["source_file"].str.startswith("data/raw_cwru_48k/").all()
        assert cw["group_id"].nunique() == 9
        l0 = cw[cw["load"] == "0hp"]
        assert l0["recording_id"].nunique() == 13  # every load-0 retained
        assert (l0.groupby("recording_id").size() >= 1).all()


# ---------------------------------------------------------------------------
# JNU
# ---------------------------------------------------------------------------

@needs_windows
def test_jnu_guards_instantiated_exactly():
    g = pd.read_csv(PART3B_DIR / "jnu_guards_1s.csv")
    assert len(g) == 48  # 12 recordings x 4 anchors
    assert (g["guard_duration_samples"] == 50_000).all()
    assert (g["guard_start_sample"]
            == g["anchor_sample"] - 25_000).all()
    assert (g["guard_end_sample"] == g["anchor_sample"] + 25_000).all()


@needs_windows
def test_jnu_windows_avoid_guards_and_blocks(manifests):
    guards = pd.read_csv(PART3B_DIR / "jnu_guards_1s.csv")
    p2 = pd.read_csv(PART2_DIR / "global_fold_1.csv")
    blocks = p2[(p2["dataset"] == "JNU")
                & (p2["is_usable"] == True)].set_index(  # noqa: E712
        ["recording_id", "temporal_block_id"])
    for df in manifests.values():
        jn = df[df["dataset"] == "JNU"]
        for _, w in jn.iterrows():
            b = blocks.loc[(w["recording_id"], w["temporal_block_id"])]
            assert w["start_sample"] >= b["temporal_start_sample"]
            assert w["end_sample"] <= b["temporal_end_sample"]
        # vectorised guard-overlap check
        for _, g in guards.iterrows():
            sub = jn[jn["recording_id"] == g["recording_id"]]
            overlap = ((sub["start_sample"] < g["guard_end_sample"])
                       & (sub["end_sample"] > g["guard_start_sample"]))
            assert not overlap.any(), \
                f"window overlaps guard at {g['anchor_sample']}"


@needs_windows
def test_jnu_coverage_and_rotation(manifests):
    expected_roles = {1: {"train": "ABC", "validation": "D", "test": "E"},
                      2: {"train": "BCD", "validation": "E", "test": "A"},
                      3: {"train": "CDE", "validation": "A", "test": "B"}}
    for k, df in manifests.items():
        jn = df[df["dataset"] == "JNU"]
        for sp, blks in expected_roles[k].items():
            got = set(jn.loc[jn["split"] == sp, "temporal_block_id"])
            assert got == set(blks), (k, sp, got)
            part = jn[jn["split"] == sp]
            combos = set(zip(part["original_label"],
                             part["rpm"].astype(int)))
            assert combos == {(c, s) for c in ("n", "ib", "ob", "tb")
                              for s in (600, 800, 1000)}


# ---------------------------------------------------------------------------
# HIT
# ---------------------------------------------------------------------------

@needs_windows
def test_hit_stream_manifest_complete():
    st = pd.read_csv(PART3B_DIR / "hit_logical_stream_manifest.csv")
    assert len(st) == 134
    assert (st["n_fragments"] == 18).all()
    assert (st["total_stream_samples"] == 18 * 20_480).all()
    assert (st["channel"] == "ch3").all()
    # ordered, gap-free fragment ids per stream; no fragment reuse
    all_frags = []
    for _, r in st.iterrows():
        ids = [int(x) for x in r["ordered_fragment_ids"].split(",")]
        assert ids == list(range(ids[0], ids[0] + 18))
        all_frags += [(r["source_file"], i) for i in ids]
    assert len(all_frags) == len(set(all_frags))


@needs_windows
def test_hit_boundary_metadata_exact(manifests):
    frag = P.HIT_FRAGMENT_SAMPLES
    for df in manifests.values():
        hit = df[df["dataset"] == "HIT"]
        # every 25,000-sample window crosses >=1 boundary (frag 20,480)
        assert hit["crosses_source_fragment_boundary"].all()
        for _, w in hit.sample(n=200, random_state=0).iterrows():
            s, e = w["start_sample"], w["end_sample"]
            lf, le = s // frag, (e - 1) // frag
            expect = [str((i + 1) * frag) for i in range(lf, le)]
            assert w["fragment_boundaries_crossed"] == ",".join(expect)
            n_frag = len(str(w["source_fragment_ids"]).split(","))
            assert n_frag == le - lf + 1


@needs_windows
def test_hit_lazy_reader_matches_reference_concatenation(manifests):
    df = manifests[1]
    hit = df[df["dataset"] == "HIT"]
    # pick one boundary-crossing window per session incl. a 2-boundary one
    picks = hit.groupby("session").head(1)
    two = hit[hit["fragment_boundaries_crossed"].str.contains(",", na=False)]
    picks = pd.concat([picks, two.head(2)])
    for _, w in picks.iterrows():
        ref = hit_reference_stream(w["source_file"], w["recording_id"])
        expect = ref[w["start_sample"]:w["end_sample"]]
        got = read_window(w)
        assert got.dtype == expect.dtype
        assert np.array_equal(got, expect)


# ---------------------------------------------------------------------------
# MaFaulDa
# ---------------------------------------------------------------------------

@needs_windows
def test_mafaulda_channel_grouping_taxonomy(manifests):
    for df in manifests.values():
        mf = df[df["dataset"] == "MAFAULDA"]
        assert (mf["channel"] == "col3_underhang_radial").all()
        normal = mf[mf["original_label"] == "normal"]
        assert (normal["group_id"] == normal["recording_id"]).all()
        assert mf[mf["original_label"] != "normal"]["group_id"] \
            .nunique() == 41
        assert not mf["original_label"].str.contains("inner").any()
        assert mf["original_label"].nunique() == 10


# ---------------------------------------------------------------------------
# counts vs Part-3A + forbidden work
# ---------------------------------------------------------------------------

@needs_windows
def test_actual_counts_match_part3a_estimates():
    stats = json.load(open(PART3B_DIR / "window_statistics.json"))
    diffs = [c for c in stats["estimate_comparison"]
             if c["difference"] != 0]
    assert diffs == [], f"unexplained count discrepancies: {diffs}"


@needs_windows
def test_sampled_signal_integrity_clean():
    stats = json.load(open(PART3B_DIR / "window_statistics.json"))
    integ = stats["sampled_signal_integrity"]
    assert integ["bad_length"] == 0
    assert integ["nonfinite"] == 0
    assert integ["constant_windows"] == []


def test_part3b_imports_no_forbidden_code():
    from conftest import assert_imports_clean
    assert_imports_clean(["src.methodology_v2.part3b_protocol",
                          "src.methodology_v2.part3b_windows",
                          "src.methodology_v2.part3b_reader"])
