"""Tests for vibrationclip_v1 text generation (Amendment 4 §8)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.vibrationclip.manifests import read_manifest  # noqa: E402
from src.vibrationclip.text import (  # noqa: E402
    CLASSES, all_unique_texts, assert_no_identity_tokens, corpus_for_manifest,
    eval_candidates, training_prompt,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PRE = REPO_ROOT / "metadata" / "vibrationclip_v1" / "pretraining"


def _labelled_rows():
    return (read_manifest(PRE / "s2s3_train.csv")
            + read_manifest(PRE / "s2s3_val.csv"))


def test_eval_candidates_are_exactly_three_class_only_prompts():
    for family in ("basic", "physics"):
        c = eval_candidates(family)
        assert set(c) == set(CLASSES)
        for text in c.values():
            # deployment-available content only: no load, speed, rpm, watts
            for token in ("rpm", " hp", " W ", "load", "Hz shaft"):
                assert token not in text, (family, text)


def test_training_prompt_is_deterministic():
    a = training_prompt("inner_race", "IR007_0_DE48k",
                        "load_hp=0;rpm=1796.0", "cwru_48k")
    b = training_prompt("inner_race", "IR007_0_DE48k",
                        "load_hp=0;rpm=1796.0", "cwru_48k")
    assert a == b


def test_cwru_gets_numeric_fault_frequency_hust_does_not():
    texts = [training_prompt("outer_race", f"OR007@6_{i}_DE48k",
                             "load_hp=0;rpm=1796.0", "cwru_48k")
             for i in range(4)]
    assert any("defect frequency near" in t for t in texts)
    hust_texts = [training_prompt("outer_race", f"O60{i}", "load_w=0;shaft_hz=24.93",
                                  "hust") for i in range(9)]
    assert all("defect frequency near" not in t for t in hust_texts)
    # HUST may still carry condition metadata
    assert any("shaft speed" in t for t in hust_texts)


def test_identity_token_enforcement_fires():
    for bad in ("A bearing from the cwru rig.",
                "A KA08 bearing with a fault.",
                "Recorded at N15_M07 condition.",
                "A 6205 deep-groove bearing.",
                "Bearing1_1 outer race failure."):
        with pytest.raises(ValueError):
            assert_no_identity_tokens(bad)
    assert_no_identity_tokens("A rolling bearing with an inner-race fault "
                              "near 162 Hz at 1796 rpm.")


def test_full_labelled_corpus_is_identity_free_and_covers_all_recordings():
    rows = _labelled_rows()
    corpus = corpus_for_manifest(rows)  # raises on any identity token
    assert set(corpus) == {r["recording_id"] for r in rows}
    n_dupes = len(corpus) - len(set(corpus.values()))
    assert n_dupes > 0  # duplicate texts exist -> §9 multi-positive loss matters


def test_unique_corpus_contains_candidates():
    rows = _labelled_rows()
    texts = set(all_unique_texts(rows))
    for family in ("basic", "physics"):
        assert set(eval_candidates(family).values()) <= texts


def test_planted_token_fixture_is_caught_by_v6(tmp_path):
    """Leakage-spec §3: the prompt corpus scanner must be shown to fire on a
    deliberately planted token in a fixture (never in real prompts)."""
    import json
    import subprocess
    fixture = tmp_path / "planted_prompts.json"
    fixture.write_text(json.dumps(
        ["A rolling bearing with an inner-race fault.",
         "A healthy rolling bearing from KI21."]))  # planted sealed bearing
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts/assert_vibrationclip_manifests.py"),
         "--prompts", str(fixture), "--out-dir", str(tmp_path)],
        capture_output=True, text=True, cwd=REPO_ROOT)
    assert proc.returncode == 1
    report = json.loads((tmp_path / "leakage_assertions.json").read_text())
    v6 = next(c for c in report["checks"] if c["code"] == "V6")
    assert v6["status"] == "FAIL"
    assert any("KI21" in v for v in v6["violations"])
