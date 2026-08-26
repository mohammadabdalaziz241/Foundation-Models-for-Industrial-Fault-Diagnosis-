"""Automated tests for the methodology_v2 Part 1 audit.

Two layers:
  1. Unit tests of the audit primitives on synthetic fixtures (no raw data
     needed, always run).
  2. Invariant tests on the REAL generated artefacts under
     methodology_v2/part1_audit/ (skipped with an explicit reason if the
     audit has not been run yet).

All failures are loud (assert / ManifestValidationError).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.methodology_v2 import FORBIDDEN_IMPORTS  # noqa: E402
from src.methodology_v2.registry import (CWRU, JNU, HIT,  # noqa: E402
                                         MAFAULDA, DATASETS, OUTPUT_DIR)
from src.methodology_v2.schema import (MANIFEST_COLUMNS,  # noqa: E402
                                       ManifestValidationError,
                                       validate_manifest)
from src.methodology_v2.integrity import (boundary_jump_probe,  # noqa: E402
                                          sha256_array, sha256_file,
                                          signal_checks)

ARTS = OUTPUT_DIR
_manifest_path = ARTS / "recording_manifest.csv"

VALID_LABELS = {
    "CWRU": CWRU["valid_labels"],
    "JNU": JNU["valid_labels"],
    "HIT": HIT["valid_labels"],
    "MAFAULDA": {"normal", "imbalance", "horizontal-misalignment",
                 "vertical-misalignment",
                 "underhang/ball_fault", "underhang/cage_fault",
                 "underhang/outer_race",
                 "overhang/ball_fault", "overhang/cage_fault",
                 "overhang/outer_race"},
}


# --------------------------------------------------------------------------
# layer 1 — unit tests on synthetic data
# --------------------------------------------------------------------------

def _dummy_manifest_row(**over):
    row = {c: None for c in MANIFEST_COLUMNS}
    row.update({
        "dataset": "JNU", "recording_id": "jnu_x", "group_id_candidate": "g",
        "original_file": "f.csv", "original_label": "n",
        "sampling_rate_hz": 50_000, "duration_seconds": 1.0,
        "n_samples": 50_000, "source_url": "u",
        "metadata_confidence": "documented",
    })
    row.update(over)
    return row


def test_validate_accepts_minimal_valid_manifest():
    df = pd.DataFrame([_dummy_manifest_row()], columns=MANIFEST_COLUMNS)
    validate_manifest(df, VALID_LABELS)


@pytest.mark.parametrize("over,fragment", [
    ({"dataset": None}, "dataset"),
    ({"sampling_rate_hz": -1}, "positive"),
    ({"sampling_rate_hz": "fast"}, "numeric"),
    ({"n_samples": 0}, "positive"),
    ({"n_samples": 10.5}, "integral"),
    ({"original_label": "bogus"}, "registry"),
    ({"metadata_confidence": "guessed"}, "confidence"),
    ({"duration_seconds": 99.0}, "inconsistent"),
])
def test_validate_rejects_bad_rows(over, fragment):
    df = pd.DataFrame([_dummy_manifest_row(**over)],
                      columns=MANIFEST_COLUMNS)
    with pytest.raises(ManifestValidationError, match=fragment):
        validate_manifest(df, VALID_LABELS)


def test_validate_rejects_duplicate_recording_id():
    df = pd.DataFrame([_dummy_manifest_row(), _dummy_manifest_row()],
                      columns=MANIFEST_COLUMNS)
    with pytest.raises(ManifestValidationError, match="duplicate"):
        validate_manifest(df, VALID_LABELS)


def test_signal_checks_flags_pathologies():
    ok = signal_checks(np.sin(np.linspace(0, 60, 5000)))
    assert ok["ok"] and ok["n_nan"] == 0
    assert signal_checks(np.full(100, 3.14))["is_constant"]
    assert signal_checks(np.array([1.0, np.nan]))["n_nan"] == 1
    assert signal_checks(np.array([1.0, np.inf]))["n_inf"] == 1
    assert signal_checks(np.ones(10), expected_min_len=100)["too_short"]


def test_boundary_probe_detects_synthetic_concatenation():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 0.01, 60_000)
    b = rng.normal(5.0, 0.01, 60_000)          # gross DC step at the join
    probe = boundary_jump_probe(np.concatenate([a, b]), [60_000],
                                window=20_000)
    assert probe[0]["jump_ratio"] > 30
    smooth = rng.normal(0, 0.01, 120_000)
    probe2 = boundary_jump_probe(smooth, [60_000], window=20_000)
    assert probe2[0]["jump_ratio"] < 3


def test_sha256_array_is_container_independent(tmp_path):
    a = np.arange(1000, dtype=np.float64)
    assert sha256_array(a) == sha256_array(a.copy())
    assert sha256_array(a) != sha256_array(a.astype(np.float32))
    p = tmp_path / "x.bin"
    p.write_bytes(b"hello")
    assert sha256_file(p) == hashlib.sha256(b"hello").hexdigest()


def test_no_training_or_windowing_modules_imported():
    """The audit package must never pull in torch or legacy pipelines
    (checked in an isolated subprocess; see conftest)."""
    from conftest import assert_imports_clean
    assert_imports_clean([
        "src.methodology_v2.audit_cwru", "src.methodology_v2.audit_jnu",
        "src.methodology_v2.audit_hit", "src.methodology_v2.audit_mafaulda",
        "src.methodology_v2.census"])


def test_audit_sources_contain_no_windowing_calls():
    """Static scan: no windowing/split/training imports or calls in the
    audit code. Imports are checked via the AST (docstrings mentioning the
    ban are fine); call vocabulary via raw text."""
    import ast
    import io
    import tokenize
    pkg = REPO_ROOT / "src" / "methodology_v2"
    forbidden_calls = ("stft", "spectrogram", "train_test_split",
                      "DataLoader", "resample")
    for py in pkg.glob("*.py"):
        text = py.read_text()
        code_tokens = " ".join(
            t.string for t in tokenize.generate_tokens(
                io.StringIO(text).readline)
            if t.type not in (tokenize.STRING, tokenize.COMMENT))
        for word in forbidden_calls:
            assert word not in code_tokens, (
                f"{py.name} code contains '{word}'")
        tree = ast.parse(text)
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods = [node.module]
            for mod in mods:
                for bad in FORBIDDEN_IMPORTS:
                    assert not (mod == bad or mod.startswith(bad + ".")), (
                        f"{py.name} imports forbidden module {mod}")


# --------------------------------------------------------------------------
# layer 2 — invariants of the real generated artefacts
# --------------------------------------------------------------------------

needs_artifacts = pytest.mark.skipif(
    not _manifest_path.exists(),
    reason="run scripts/methodology_v2/run_part1_audit.py first")


@pytest.fixture(scope="module")
def manifest() -> pd.DataFrame:
    return pd.read_csv(_manifest_path)


@needs_artifacts
def test_real_manifest_passes_full_validation(manifest):
    present = {k: v for k, v in VALID_LABELS.items()
               if k in set(manifest["dataset"])}
    validate_manifest(manifest, present)


@needs_artifacts
def test_real_manifest_row_counts(manifest):
    counts = manifest.groupby("dataset").size().to_dict()
    # CWRU: 60 (12k DE) + 52 fault recs (48k DE) minus nothing, normals once
    assert counts.get("CWRU") == 112, counts
    assert counts.get("JNU") == 12, counts
    if "MAFAULDA" in counts:
        assert counts["MAFAULDA"] == MAFAULDA["expected_sequences"], counts
    if "HIT" in counts:
        assert 100 <= counts["HIT"] <= 140, counts


@needs_artifacts
def test_no_raw_file_hash_collision_across_datasets():
    """The same raw bytes must not appear under two datasets, and known
    CWRU normal duplicates must be the ONLY within-dataset file-level
    duplicates."""
    hashes = pd.read_csv(ARTS / "raw_file_hashes.csv")
    dup = hashes[hashes.duplicated("sha256", keep=False)]
    for sha, grp in dup.groupby("sha256"):
        assert grp["dataset"].nunique() == 1, f"cross-dataset dup: {grp}"
        files = sorted(grp["file"])
        assert all("Normal" in f or "normal" in f for f in files), (
            f"unexpected duplicate raw files: {files}")


@needs_artifacts
def test_census_and_manifest_agree(manifest):
    census = pd.read_csv(ARTS / "dataset_census.csv")
    for _, row in census.iterrows():
        sub = manifest[manifest["dataset"] == row["dataset"]]
        assert len(sub) == row["n_recordings"]


@needs_artifacts
def test_original_data_unmodified_spot_check():
    """Re-hash a deterministic sample of raw files and compare with the
    frozen audit hashes — proves the audit (and anything since) has not
    modified raw data."""
    hashes = pd.read_csv(ARTS / "raw_file_hashes.csv")
    real = hashes[~hashes["file"].str.contains(r"\[|github:", regex=True)]
    sample = real.sort_values("file").iloc[::max(1, len(real) // 20)]
    for _, rec in sample.iterrows():
        base = {"CWRU": REPO_ROOT / "data",
                "JNU": JNU["paths"]["root"],
                "HIT": REPO_ROOT,
                "MAFAULDA": MAFAULDA["paths"]["root"]}[rec["dataset"]]
        p = base / rec["file"]
        if not p.exists():
            p = REPO_ROOT / rec["file"]
        assert p.exists(), f"raw file vanished: {rec['file']}"
        assert sha256_file(p) == rec["sha256"], f"raw file CHANGED: {p}"


@needs_artifacts
def test_hit_official_split_evidence_recorded():
    details = json.load(open(ARTS / "integrity_details.json"))
    gh = details["extra"].get("HIT_github_release")
    if gh is None:
        pytest.skip("HIT not audited yet")
    assert gh["label_counts"]["ytest"] == {"0": 954, "1": 1008, "2": 450} or \
           gh["label_counts"]["ytest"] == {0: 954, 1: 1008, 2: 450}
    assert "provenance" in gh
