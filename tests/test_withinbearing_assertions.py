"""Negative controls for scripts/assert_withinbearing_manifests.py
(Amendment 5 §9, leakage spec R5): every planted violation must FAIL the
corresponding check. Run file-by-file:

    .venv/bin/python -m pytest tests/test_withinbearing_assertions.py -q
"""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.assert_withinbearing_manifests import (  # noqa: E402
    ID_COLUMNS,
    KB_BEARINGS,
    N_SPLITS,
    QUINTILE_SIZE,
    Report,
    SEALED_BEARINGS,
    _boundary_regex,
    _gnu_grep_count,
    check_rotation,
    check_wv4,
    find_forbidden_ids,
    find_interval_overlaps,
)

WB_ROOT = REPO_ROOT / "metadata/withinbearing_v1"
PYTHON = str(REPO_ROOT / ".venv/bin/python")


def _row(**overrides) -> dict:
    row = {col: "clean" for col in ID_COLUMNS}
    row.update(overrides)
    return row


# ------------------------------------------------------- WV2 / WV3 controls

def test_planted_sealed_bearing_is_caught():
    for sealed in SEALED_BEARINGS:
        hits = find_forbidden_ids(
            {"f.csv": [_row(bearing_id=sealed)]}, SEALED_BEARINGS)
        assert hits, f"planted sealed bearing {sealed} was NOT caught"


def test_planted_sealed_recording_id_is_caught():
    hits = find_forbidden_ids(
        {"f.csv": [_row(recording_id="N15_M07_F10_KA08_3")]},
        SEALED_BEARINGS)
    assert hits


def test_planted_kb_bearing_is_caught():
    hits = find_forbidden_ids(
        {"f.csv": [_row(sample_id="paderborn:N15_M07_F10_KB23_1:0-1")]},
        KB_BEARINGS)
    assert hits


def test_boundary_semantics_do_not_false_positive():
    # KA081 must NOT match KA08 (leakage spec R3)
    hits = find_forbidden_ids(
        {"f.csv": [_row(recording_id="KA081")]}, ("KA08",))
    assert not hits


# ------------------------------------------------------------ WV5 controls

def test_raw_byte_scan_fires_on_planted_token(tmp_path):
    planted = tmp_path / "planted.csv"
    planted.write_text("sample_id\npaderborn:N15_M07_F10_K005_7:0-1\n")
    assert _gnu_grep_count(_boundary_regex("K005"), planted) == 1


def test_raw_byte_scan_boundary(tmp_path):
    inside = tmp_path / "inside.csv"
    inside.write_text("x,N15_M07_F10_KA08_1\n")
    outside = tmp_path / "outside.csv"
    outside.write_text("x,KA081\n")
    assert _gnu_grep_count(_boundary_regex("KA08"), inside) == 1
    assert _gnu_grep_count(_boundary_regex("KA08"), outside) == 0


# ------------------------------------------------------------ WV7 controls

def test_interval_overlap_is_caught():
    hits = find_interval_overlaps({
        "train": {"REC": [(0, 24000)]},
        "test": {"REC": [(12000, 36000)]},
    })
    assert any("interval overlap" in h for h in hits)


def test_shared_recording_without_interval_overlap_is_still_caught():
    hits = find_interval_overlaps({
        "train": {"REC": [(0, 24000)]},
        "validation": {"REC": [(48000, 72000)]},
    })
    assert any("present in both" in h for h in hits)


def test_disjoint_recordings_pass():
    assert find_interval_overlaps({
        "train": {"A": [(0, 24000)]},
        "test": {"B": [(0, 24000)]},
    }) == []


# ------------------------------------------------------------ WV8 controls

def _synthetic_splits() -> tuple[dict, dict]:
    """A minimal valid universe: one bearing, 20 recordings, correct
    rotation. Returns (splits, committed_qmap)."""
    bearing = "K001"
    recs = [f"N15_M07_F10_{bearing}_{i}" for i in range(1, 21)]
    quintile = {rec: i // QUINTILE_SIZE for i, rec in enumerate(recs)}
    splits: dict[int, dict[str, list[dict]]] = {}
    for k in range(N_SPLITS):
        by_role: dict[str, list[dict]] = {
            "train": [], "validation": [], "test": []}
        for i, rec in enumerate(recs):
            q = quintile[rec]
            role = ("test" if q == k
                    else "validation" if q == (k + 1) % N_SPLITS
                    else "train")
            by_role[role].append({
                "recording_id": rec, "bearing_id": bearing,
                "repeat_index": str(i + 1)})
        splits[k] = by_role
    committed = {"bearings": {bearing: {
        "ordered_recordings": recs,
        "quintile": quintile,
    }}}
    return splits, committed


def test_valid_rotation_passes():
    splits, committed = _synthetic_splits()
    assert check_rotation(splits, committed) == []


def test_recording_in_test_twice_fails():
    splits, committed = _synthetic_splits()
    # move a split-1 train recording (Q0, tested in split 0) into test
    moved = next(r for r in splits[1]["train"]
                 if int(r["repeat_index"]) <= QUINTILE_SIZE)
    splits[1]["train"].remove(moved)
    splits[1]["test"].append(moved)
    violations = check_rotation(splits, committed)
    assert any("in test 2 times" in v for v in violations)


def test_wrong_role_placement_fails():
    splits, committed = _synthetic_splits()
    # swap one test recording with one train recording inside split 2
    t = splits[2]["test"].pop()
    tr = splits[2]["train"].pop()
    splits[2]["test"].append(tr)
    splits[2]["train"].append(t)
    violations = check_rotation(splits, committed)
    assert any("expected" in v for v in violations)


def test_tampered_quintile_map_fails():
    splits, committed = _synthetic_splits()
    bearing = next(iter(committed["bearings"]))
    qmap = committed["bearings"][bearing]["quintile"]
    first = next(iter(qmap))
    qmap[first] = (qmap[first] + 1) % N_SPLITS
    violations = check_rotation(splits, committed)
    assert violations


# ------------------------------------------------------------ WV4 controls

def test_checksum_tamper_fails(tmp_path):
    tampered = tmp_path / "withinbearing_v1"
    shutil.copytree(WB_ROOT, tampered)
    target = tampered / "splits/split0/train.csv"
    rows = target.read_text().splitlines()
    rows[1] = rows[1].replace("paderborn", "tampered", 1)
    target.write_text("\n".join(rows) + "\n")
    rep = Report()
    check_wv4(rep, tampered)
    assert rep.checks[0]["status"] == "FAIL"
    assert any("checksum mismatch" in v
               for v in rep.checks[0]["violations"])


def test_untampered_tree_passes_wv4():
    rep = Report()
    check_wv4(rep, WB_ROOT)
    assert rep.checks[0]["status"] == "PASS", rep.checks[0]["violations"]


# -------------------------------------------------------------- end-to-end

@pytest.mark.skipif(not WB_ROOT.exists(), reason="manifests not built")
def test_full_assertion_script_passes(tmp_path):
    proc = subprocess.run(
        [PYTHON, str(REPO_ROOT / "scripts/assert_withinbearing_manifests.py"),
         "--out-dir", str(tmp_path)],
        capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    report = json.loads((tmp_path / "leakage_assertions.json").read_text())
    assert report["overall"] == "PASS"
    assert {c["code"] for c in report["checks"]} == {
        "WV2", "WV3", "WV4", "WV5", "WV7", "WV8", "WCOV"}


def test_builder_is_deterministic(tmp_path):
    proc = subprocess.run(
        [PYTHON, str(REPO_ROOT / "scripts/build_withinbearing_manifests.py"),
         "--out-root", str(tmp_path / "rebuild")],
        capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    for rel in ["recording_quintiles.json", "manifest_checksums.json"] + [
            f"splits/split{k}/{role}.csv"
            for k in range(N_SPLITS)
            for role in ("train", "validation", "test")]:
        rebuilt = (tmp_path / "rebuild" / rel).read_bytes()
        committed = (WB_ROOT / rel).read_bytes()
        assert rebuilt == committed, f"non-deterministic rebuild: {rel}"


def test_planted_sealed_row_fails_end_to_end(tmp_path):
    """Full-pipeline negative control: a sealed row planted into a copy of
    the manifest tree must FAIL the script (not merely a unit checker)."""
    tampered = tmp_path / "withinbearing_v1"
    shutil.copytree(WB_ROOT, tampered)
    target = tampered / "splits/split3/train.csv"
    with target.open(newline="") as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0].keys())
    planted = dict(rows[0])
    planted["bearing_id"] = "KA08"
    planted["recording_id"] = "N15_M07_F10_KA08_1"
    planted["sample_id"] = "paderborn:N15_M07_F10_KA08_1:00000000-00024000"
    rows.append(planted)
    with target.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    proc = subprocess.run(
        [PYTHON, str(REPO_ROOT / "scripts/assert_withinbearing_manifests.py"),
         "--root", str(tampered), "--out-dir", str(tmp_path / "out")],
        capture_output=True, text=True)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    report = json.loads(
        (tmp_path / "out" / "leakage_assertions.json").read_text())
    failed = {c["code"] for c in report["checks"] if c["status"] == "FAIL"}
    assert "WV2" in failed and "WV5" in failed and "WV4" in failed
