"""Structural tests for the Gate D' full pipeline (Amendment 5), validated
on FIXTURES before any real result exists (house rule). Run file-by-file:

    .venv/bin/python -m pytest tests/test_withinbearing_full_pipeline.py -q

Covers: the frozen §3.6 decision rule (>= +0.02 AND 5/5 splits), the §3.7
saturation declaration path, aggregator refusal on missing inputs, §4.4
selection reading validation fields only (adversarial fixture where the
test ranking disagrees), selection tie-break order, and the runner's
refusal semantics (completed-run refusal, tampered-checkpoint refusal,
LODO go/no-go gate).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        name, REPO_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


agg = _load("aggregate_withinbearing")
runner = _load("run_withinbearing")

SEEDS = (42, 43, 44)


# ------------------------------------------------------------- fixtures
def _metrics(macro: float) -> dict:
    return {"macro_f1": macro, "bb_f1": macro,
            "outer_race_recall": macro,
            "confusion": [[10, 0, 0], [0, 10, 0], [0, 0, 10]]}


def write_eval_fixture(full_dir: Path, pre_by_split, rand_by_split) -> None:
    for arm, values in (("WB-PRE", pre_by_split), ("WB-RAND", rand_by_split)):
        for split in range(5):
            for seed in SEEDS:
                p = full_dir / "eval_a" / arm / f"split{split}_seed{seed}.json"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps({
                    "arm": arm, "split": split, "seed": seed,
                    "test": _metrics(values[split]),
                    "validation": _metrics(values[split])}) + "\n")


# ------------------------------------------------- §3.6 / §3.7 decision
def test_success_case(tmp_path):
    write_eval_fixture(tmp_path, [0.50, 0.52, 0.54, 0.51, 0.53],
                       [0.47, 0.48, 0.50, 0.48, 0.49])
    r = agg.aggregate(tmp_path)["primary"]
    assert r["success_criterion_met"] and not r["saturated"]
    assert r["n_positive_splits"] == 5
    assert abs(r["mean_difference"] - 0.036) < 1e-9
    assert r["verdict"].startswith("SUCCESS")


def test_sign_consistency_fails_despite_large_mean(tmp_path):
    write_eval_fixture(tmp_path, [0.70, 0.70, 0.70, 0.70, 0.40],
                       [0.50, 0.50, 0.50, 0.50, 0.45])
    r = agg.aggregate(tmp_path)["primary"]
    assert r["mean_difference"] > 0.02
    assert r["n_positive_splits"] == 4
    assert not r["success_criterion_met"]
    assert r["verdict"].startswith("NOT MET")


def test_threshold_fails_despite_all_positive(tmp_path):
    write_eval_fixture(tmp_path, [0.51, 0.51, 0.51, 0.51, 0.51],
                       [0.50, 0.50, 0.50, 0.50, 0.50])
    r = agg.aggregate(tmp_path)["primary"]
    assert r["n_positive_splits"] == 5
    assert not r["success_criterion_met"]


def test_saturation_declaration_path(tmp_path):
    write_eval_fixture(tmp_path, [0.995, 0.99, 0.992, 0.991, 0.993],
                       [0.990, 0.985, 0.988, 0.987, 0.989])
    r = agg.aggregate(tmp_path)["primary"]
    assert r["saturated"]
    assert r["verdict"].startswith("SATURATED")
    # the decision rule is still evaluated for the record
    assert "success_criterion_met" in r and not r["success_criterion_met"]


def test_not_saturated_if_one_arm_below_level(tmp_path):
    write_eval_fixture(tmp_path, [0.995] * 5, [0.90] * 5)
    r = agg.aggregate(tmp_path)["primary"]
    assert not r["saturated"]
    assert r["success_criterion_met"]  # +0.095, 5/5


def test_aggregator_refuses_on_missing_input(tmp_path):
    write_eval_fixture(tmp_path, [0.5] * 5, [0.4] * 5)
    (tmp_path / "eval_a/WB-PRE/split3_seed43.json").unlink()
    with pytest.raises(SystemExit, match="REFUSED"):
        agg.aggregate(tmp_path)


def test_no_inferential_test_in_output(tmp_path):
    write_eval_fixture(tmp_path, [0.5] * 5, [0.4] * 5)
    r = agg.aggregate(tmp_path)["primary"]
    assert "p_value" not in json.dumps(r)
    assert "NONE" in r["inference"]


# ------------------------------------------------------- §4.4 selection
def write_probe_fixture(full_dir: Path, arm: str, val: float,
                        test: float) -> None:
    for seed in SEEDS:
        p = full_dir / "probe" / arm / f"seed{seed}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"arm": arm, "seed": seed, "splits": {
            f"split{k}": {"validation": {"macro_f1": val},
                          "test": {"macro_f1": test}}
            for k in range(5)}}) + "\n")


def test_selection_reads_validation_only(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "FULL_DIR", tmp_path)
    # adversarial: S1p wins on TEST, S4p wins on VALIDATION
    vals = {"S1p": (0.80, 0.99), "S4p": (0.90, 0.10),
            "S5p": (0.70, 0.95), "S6p": (0.60, 0.99), "S7p": (0.50, 0.99)}
    for arm, (v, t) in vals.items():
        write_probe_fixture(tmp_path, arm, v, t)
    runner.stage_select()
    doc = json.loads((tmp_path / "selected_arm.json").read_text())
    assert doc["selected"] == "S4p"
    assert "validation" in doc["rule"]


def test_selection_tie_breaks_to_frozen_order(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "FULL_DIR", tmp_path)
    for arm in runner.FAMILY:
        write_probe_fixture(tmp_path, arm, 0.75, 0.5)
    runner.stage_select()
    doc = json.loads((tmp_path / "selected_arm.json").read_text())
    assert doc["selected"] == "S1p"  # earliest in the frozen order


def test_selection_refuses_on_missing_probe(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "FULL_DIR", tmp_path)
    write_probe_fixture(tmp_path, "S1p", 0.8, 0.5)  # others missing
    with pytest.raises(SystemExit, match="REFUSED"):
        runner.stage_select()


def test_selection_refuses_to_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "FULL_DIR", tmp_path)
    for arm in runner.FAMILY:
        write_probe_fixture(tmp_path, arm, 0.75, 0.5)
    runner.stage_select()
    with pytest.raises(SystemExit, match="REFUSED"):
        runner.stage_select()


# -------------------------------------------------- runner refusal cases
def test_completed_training_refuses(tmp_path):
    d = tmp_path / "S1p_seed42"
    d.mkdir()
    (d / "record.json").write_text("{}")
    with pytest.raises(SystemExit, match="REFUSED"):
        runner.refuse_if_done(d)


def test_tampered_checkpoint_refuses(tmp_path):
    d = tmp_path / "S1p_seed42"
    d.mkdir()
    torch.save({"w": torch.zeros(2)}, d / "encoder.pt")
    (d / "record.json").write_text(json.dumps(
        {"files_sha256": {"encoder.pt": "0" * 64}}))
    with pytest.raises(SystemExit, match="hash mismatch"):
        runner._rv.load_pinned(d, ["encoder"])


def test_missing_checkpoint_refuses(tmp_path):
    with pytest.raises(SystemExit, match="REFUSED"):
        runner._rv.load_pinned(tmp_path / "absent", ["encoder"])


def test_lodo_refuses_without_go_decision(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "META_LODO", tmp_path)  # no GO file
    with pytest.raises(SystemExit, match="GO_DECISION"):
        runner.stage_lodo("minus_cwru", 42, None, None, None)


def test_lodo_refuses_on_nogo(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "META_LODO", tmp_path)
    (tmp_path / "GO_DECISION.json").write_text(
        json.dumps({"decision": "NO-GO", "date": "2026-08-08"}))
    with pytest.raises(SystemExit, match="enumerated-and-not-run"):
        runner.stage_lodo("minus_cwru", 42, None, None, None)


def test_aggregator_lodo_pending_without_decision_file():
    # the repo has no GO_DECISION.json before 2026-08-08
    section = agg.lodo_section(REPO_ROOT / "results/withinbearing_v1/full")
    assert "pending" in section["status"] or "enumerated" in section["status"]
