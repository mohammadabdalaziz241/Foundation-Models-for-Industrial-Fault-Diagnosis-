"""Negative-control tests (spec R5) for scripts/assert_vibrationclip_manifests.py.

Every quarantine assertion must be shown to FAIL when a forbidden ID is planted.
A scanner that cannot find a planted violation proves nothing when it reports
zero hits (docs/leakage_assertion_spec.md, motivating incident 2026-08-01).
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "assert_vibrationclip_manifests.py"
PYTHON = sys.executable

sys.path.insert(0, str(REPO_ROOT))
from scripts.assert_vibrationclip_manifests import (  # noqa: E402
    PADERBORN_ALL, PRETRAINING_MANIFESTS, SEALED_BEARINGS, _py_boundary,
)

CWRU_SOURCE = REPO_ROOT / PRETRAINING_MANIFESTS[0]


def _run(*args: str, out_dir: Path) -> tuple[int, dict]:
    proc = subprocess.run(
        [PYTHON, str(SCRIPT), "--out-dir", str(out_dir), *args],
        capture_output=True, text=True, cwd=REPO_ROOT)
    report_path = out_dir / "leakage_assertions.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    return proc.returncode, report


def _status(report: dict, code: str) -> str:
    return next(c["status"] for c in report["checks"] if c["code"] == code)


def _plant_row(dst: Path, **overrides: str) -> None:
    """Copy the CWRU source manifest and append one row with planted fields."""
    with CWRU_SOURCE.open(newline="") as f:
        rows = list(csv.DictReader(f))
        fields = rows[0].keys()
    planted = dict(rows[0])
    planted.update(overrides)
    with dst.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows[:5] + [planted])


def test_baseline_passes(tmp_path):
    code, report = _run(out_dir=tmp_path)
    assert code == 0
    assert report["overall"] == "PASS"


def test_planted_paderborn_bearing_fails_v1(tmp_path):
    bad = tmp_path / "planted_paderborn.csv"
    _plant_row(bad, bearing_id="KA04", dataset="paderborn")
    code, report = _run("--pretraining-manifest", str(bad), out_dir=tmp_path)
    assert code == 1
    assert _status(report, "V1") == "FAIL"


def test_planted_paderborn_path_token_fails_even_with_clean_columns(tmp_path):
    bad = tmp_path / "planted_path.csv"
    _plant_row(bad, source_path="data/raw_paderborn/KI04/N15_M07_F10_KI04_3.mat")
    code, report = _run("--pretraining-manifest", str(bad), out_dir=tmp_path)
    assert code == 1
    assert _status(report, "V1") == "FAIL"


@pytest.mark.parametrize("sealed", ["K005", "KA30", "KI21"])
def test_planted_sealed_bearing_fails_v2(tmp_path, sealed):
    bad = tmp_path / "planted_sealed.csv"
    _plant_row(bad, recording_id=f"N15_M07_F10_{sealed}_1")
    code, report = _run("--pretraining-manifest", str(bad), out_dir=tmp_path)
    assert code == 1
    assert _status(report, "V2") == "FAIL"


def test_planted_kb_bearing_fails_v3(tmp_path):
    bad = tmp_path / "planted_kb.csv"
    _plant_row(bad, bearing_id="KB27")
    code, report = _run("--pretraining-manifest", str(bad), out_dir=tmp_path)
    assert code == 1
    # KB bearings are not in V2's sealed set; V3 must be the check that fires.
    assert _status(report, "V3") == "FAIL"


def test_prompt_corpus_identity_token_fails_v6(tmp_path):
    prompts = tmp_path / "prompts.json"
    prompts.write_text(json.dumps({
        "healthy": ["A healthy rolling bearing."],
        "inner": ["A rolling bearing with an inner-race fault from KA08."],
    }))
    code, report = _run("--prompts", str(prompts), out_dir=tmp_path)
    assert code == 1
    assert _status(report, "V6") == "FAIL"


def test_prompt_corpus_dataset_token_fails_v6(tmp_path):
    prompts = tmp_path / "prompts.json"
    prompts.write_text(json.dumps(
        {"healthy": ["A healthy rolling bearing from the cwru rig."]}))
    code, report = _run("--prompts", str(prompts), out_dir=tmp_path)
    assert code == 1
    assert _status(report, "V6") == "FAIL"


def test_prompt_scan_without_positive_control_is_indeterminate(tmp_path):
    # A corpus with no violations but also no occurrence of the positive-control
    # token must be INDETERMINATE, never PASS (R2).
    prompts = tmp_path / "prompts.json"
    prompts.write_text(json.dumps({"healthy": ["A healthy machine element."]}))
    code, report = _run("--prompts", str(prompts), out_dir=tmp_path)
    assert code == 2
    assert _status(report, "V6") == "INDETERMINATE"
    assert report["overall"] == "INDETERMINATE"


def test_clean_prompt_corpus_passes(tmp_path):
    prompts = tmp_path / "prompts.json"
    prompts.write_text(json.dumps({
        "healthy": ["A healthy rolling bearing."],
        "inner": ["A rolling bearing with an inner-race fault."],
        "outer": ["A rolling bearing with an outer-race fault."],
    }))
    code, report = _run("--prompts", str(prompts), out_dir=tmp_path)
    assert code == 0
    assert _status(report, "V6") == "PASS"


def test_boundary_semantics_r3():
    pat = _py_boundary("KA08")
    assert pat.search("N15_M07_F10_KA08_1")
    assert pat.search("KA08")
    assert not pat.search("KA081")
    assert not pat.search("XKA08Y".replace("X", "1").replace("Y", "2"))


def test_forbidden_id_lists_are_complete():
    assert set(SEALED_BEARINGS) == {
        "K005", "K006", "KA08", "KA09", "KA30", "KI08", "KI18", "KI21"}
    assert set(SEALED_BEARINGS) <= set(PADERBORN_ALL)
    assert {"KB23", "KB24", "KB27"} <= set(PADERBORN_ALL)
    assert len(PADERBORN_ALL) == 32
