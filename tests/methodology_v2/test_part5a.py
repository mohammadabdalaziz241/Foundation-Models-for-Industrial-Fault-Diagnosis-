"""Focused tests for the Part-5A architecture and novelty audit."""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.methodology_v2 import FORBIDDEN_IMPORTS  # noqa: E402
from src.methodology_v2.part2_builder import verify_frozen_hashes  # noqa: E402
from src.methodology_v2.part3b_windows import verify_part3b_hashes  # noqa: E402
from src.methodology_v2.part4c_normalizers import (  # noqa: E402
    verify_part4c_hashes)
from src.methodology_v2.part5a_analysis import (PART5A_DIR,  # noqa: E402
                                                estimate_params,
                                                literature_registry,
                                                novelty_stress_test,
                                                parameter_budget,
                                                patch_geometry, write_all)

needs_artifacts = pytest.mark.skipif(
    not (PART5A_DIR / "patch_geometry_study.csv").exists(),
    reason="run scripts/methodology_v2/run_part5a.py first")


def test_all_upstream_seals_intact():
    verify_frozen_hashes()
    verify_part3b_hashes()
    verify_part4c_hashes()


def test_patch_geometry_math():
    pg = patch_geometry().set_index(
        ["dataset", "patch_freq_bins", "patch_time_frames"])
    r = pg.loc[("CWRU", 16, 8)]
    assert r["n_freq_patches"] == 33 and r["n_time_patches"] == 23
    assert r["tokens"] == 759 and r["pad_freq_bins"] == 15
    assert r["pad_time_frames"] == 0            # 184/8 exact
    assert r["patch_freq_width_hz"] == pytest.approx(750.0)
    r = pg.loc[("HIT", 16, 8)]
    assert r["tokens"] == 17 * 24 == 408
    assert r["patch_freq_width_hz"] == pytest.approx(781.25)
    assert r["padded_value_pct"] == pytest.approx(5.51, abs=0.01)
    r = pg.loc[("JNU", 8, 8)]
    assert r["n_freq_patches"] == 65 and r["tokens"] == 65 * 24
    r = pg.loc[("MAFAULDA", 32, 8)]
    assert r["n_freq_patches"] == 17 and r["tokens"] == 17 * 24


def test_parameter_budget_within_tiers():
    pb = parameter_budget().set_index("tier")
    assert 0.5e6 <= pb.loc["Tiny", "estimated_params"] <= 1.0e6
    assert 1.0e6 <= pb.loc["Small", "estimated_params"] <= 3.0e6
    assert 3.0e6 <= pb.loc["Medium", "estimated_params"] <= 8.0e6
    assert pb["within_target_range"].all()
    # formula sanity: quadratic in d, linear in blocks
    assert estimate_params(192, 4, "bimamba") \
        > estimate_params(128, 4, "bimamba")
    assert estimate_params(192, 4, "bimamba") \
        > estimate_params(192, 3, "bimamba")


def test_novelty_first_claims_all_rejected():
    nv = novelty_stress_test()
    firsts = nv[nv["candidate_claim"].str.lower().str.startswith("first")]
    assert len(firsts) >= 4
    assert firsts["verdict"].isin(["REJECTED", "TOO STRONG"]).all(), \
        "a 'First ...' claim survived — not allowed without extraordinary evidence"
    assert set(nv["verdict"]) <= {"REJECTED", "TOO STRONG",
                                  "POSSIBLY DEFENSIBLE", "DEFENSIBLE"}
    # every claim names its closest prior work
    assert nv["closest_prior_work"].str.len().gt(4).all()


def test_literature_registry_complete():
    lit = literature_registry()
    assert len(lit) >= 10
    assert lit["url"].str.startswith("http").all()
    for key in ("VibFM", "ECHO", "SSAMBA", "AudioMamba", "SepTr",
                "SpecTNT", "FISHER"):
        assert key in set(lit["key"]), f"missing required entry {key}"
    assert (lit["search_date"] == "2026-08-12").all()


@needs_artifacts
def test_artifacts_deterministic(tmp_path):
    write_all(tmp_path)
    for name in ("patch_geometry_study.csv",
                 "parameter_budget_estimates.csv",
                 "literature_registry.csv",
                 "novelty_claim_stress_test.csv"):
        assert (PART5A_DIR / name).read_bytes() \
            == (tmp_path / name).read_bytes(), f"{name} not deterministic"


@needs_artifacts
def test_required_artifacts_exist_no_checkpoints():
    for name in ("variable_shape_options.csv",
                 "coordinate_encoding_options.csv",
                 "sequence_organization_options.csv",
                 "backbone_options.csv", "frequency_mixer_options.csv",
                 "architecture_candidates.yaml",
                 "part5a_recommendations.yaml", "search_terms_log.txt"):
        assert (PART5A_DIR / name).exists(), name
    banned = {".pt", ".pth", ".ckpt", ".npz", ".npy", ".onnx"}
    for f in PART5A_DIR.rglob("*"):
        assert f.suffix.lower() not in banned, f"model artefact: {f}"


def test_no_model_layers_imported_or_referenced():
    from conftest import assert_imports_clean
    assert_imports_clean(["src.methodology_v2.part5a_analysis"])
    src = (REPO_ROOT / "src" / "methodology_v2"
           / "part5a_analysis.py").read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        mods = ([a.name for a in node.names]
                if isinstance(node, ast.Import) else
                [node.module] if isinstance(node, ast.ImportFrom)
                and node.module else [])
        for m in mods:
            assert not any(m.startswith(b) for b in
                           ("torch", "mamba", "tensorflow", "jax",
                            "flax", "keras")), f"model import {m}"
