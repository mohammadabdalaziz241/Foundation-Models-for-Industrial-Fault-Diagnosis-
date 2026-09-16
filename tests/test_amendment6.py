"""Amendment 6 pre-freeze test battery (docs/amendment6_strict_protocol.md
§9). All tests run BEFORE any Amendment 6 training exists. Run file-by-file:

    .venv/bin/python -m pytest tests/test_amendment6.py -q
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.bearing_generalisation import LEGACY_COMPROMISED_BEARINGS  # noqa: E402


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        name, REPO_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


runner = _load("run_amendment6")
agg = _load("aggregate_amendment6")

META = REPO_ROOT / "metadata/vibrationclip_v1/downstream"
SEEDS = (42, 43, 44)


def _rows(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


# ------------------------------------------- benchmark byte-identity (V4)
def test_benchmark_manifests_match_committed_checksums():
    recorded = json.loads(
        (REPO_ROOT / "metadata/vibrationclip_v1/manifest_checksums.json")
        .read_text())
    entries = recorded.get("files", recorded)
    checked = 0
    for inst in runner.FOLD_INSTANCES:
        for name in ("label_blocks/Rfull.csv", "validation.csv",
                     "setting_b_eval.csv"):
            rel = f"downstream/{inst}/{name}"
            assert rel in entries, f"missing checksum entry for {rel}"
            want = entries[rel]
            want_sha = want["sha256"] if isinstance(want, dict) else want
            assert _sha(REPO_ROOT / "metadata/vibrationclip_v1" / rel) == \
                want_sha, f"benchmark manifest altered: {rel}"
            checked += 1
    assert checked == 36


# --------------------------------------- pairing / disjointness / sealed
def test_bearing_and_sample_disjointness_all_folds():
    sealed = set(LEGACY_COMPROMISED_BEARINGS)
    for inst in runner.FOLD_INSTANCES:
        d = META / inst
        train = _rows(d / "label_blocks/Rfull.csv")
        val = _rows(d / "validation.csv")
        test = _rows(d / "setting_b_eval.csv")
        b = {k: {r["bearing_id"] for r in v}
             for k, v in (("train", train), ("val", val), ("test", test))}
        s = {k: {r["sample_id"] for r in v}
             for k, v in (("train", train), ("val", val), ("test", test))}
        assert not (b["train"] | b["val"]) & b["test"], inst
        assert not (s["train"] | s["val"]) & s["test"], inst
        assert not (b["train"] | b["val"] | b["test"]) & sealed, inst


def test_planted_sealed_row_trips_quarantine():
    for sealed in LEGACY_COMPROMISED_BEARINGS:
        with pytest.raises(SystemExit, match="quarantine"):
            runner.quarantine_guard(
                [{"bearing_id": sealed}], "negative control")


def test_clean_rows_pass_quarantine():
    runner.quarantine_guard([{"bearing_id": "K001"}], "control")


def test_planted_bearing_overlap_fails_disjointness(tmp_path, monkeypatch):
    # copy one fold, plant a test bearing into the training manifest
    inst = runner.FOLD_INSTANCES[0]
    src = META / inst
    dst = tmp_path / inst
    (dst / "label_blocks").mkdir(parents=True)
    for name in ("label_blocks/Rfull.csv", "validation.csv",
                 "setting_b_eval.csv"):
        (dst / name).write_bytes((src / name).read_bytes())
    test_rows = _rows(dst / "setting_b_eval.csv")
    with (dst / "label_blocks/Rfull.csv").open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(test_rows[0].keys()))
        w.writerow(test_rows[0])
    monkeypatch.setattr(runner, "META", tmp_path)
    with pytest.raises(SystemExit, match="bearing overlap"):
        runner.assert_fold_disjointness(inst)


def test_real_folds_pass_disjointness_assertion():
    for inst in runner.FOLD_INSTANCES:
        runner.assert_fold_disjointness(inst)


# ------------------------------------------------- S5p pins and refusals
def test_s5p_pins_match_records_and_files():
    for seed in SEEDS:
        d = REPO_ROOT / f"models/saved_withinbearing/S5p_seed{seed}"
        record = json.loads((d / "record.json").read_text())
        recorded = record["files_sha256"]["encoder.pt"]
        assert recorded == runner.S5P_PINS[seed]
        assert _sha(d / "encoder.pt") == runner.S5P_PINS[seed]


def test_selected_arm_is_s5p():
    doc = json.loads(runner.SELECTED_ARM_FILE.read_text())
    assert doc["selected"] == "S5p"


def test_tampered_s5p_pin_refuses(tmp_path, monkeypatch):
    d = tmp_path / "S5p_seed42"
    d.mkdir()
    torch.save({"w": torch.zeros(1)}, d / "encoder.pt")
    (d / "record.json").write_text(json.dumps(
        {"files_sha256": {"encoder.pt": _sha(d / "encoder.pt")}}))
    monkeypatch.setattr(runner, "WB_CKPT_ROOT", tmp_path)
    with pytest.raises(SystemExit):  # pin mismatch (record consistent,
        runner.build_model("STRICT-PRE", 42,        # frozen pin differs)
                           torch.device("cpu"))


# ------------------------------------ parameter / state-shape equality
def test_parameter_and_shape_equality_between_arms():
    rand, sha_r = runner.build_model("STRICT-RAND", 42, torch.device("cpu"))
    assert sha_r is None
    n_rand = sum(p.numel() for p in rand.parameters())
    assert n_rand == runner.EXPECTED_PARAMS == 12_451_395
    pre, sha_p = runner.build_model("STRICT-PRE", 42, torch.device("cpu"))
    assert sha_p == runner.S5P_PINS[42]
    n_pre = sum(p.numel() for p in pre.parameters())
    assert n_pre == n_rand
    assert ({k: tuple(v.shape) for k, v in rand.state_dict().items()}
            == {k: tuple(v.shape) for k, v in pre.state_dict().items()})


def test_strict_pre_actually_loads_s5p_weights():
    rand, _ = runner.build_model("STRICT-RAND", 42, torch.device("cpu"))
    pre, _ = runner.build_model("STRICT-PRE", 42, torch.device("cpu"))
    diffs = sum(
        (rand.state_dict()[k] != pre.state_dict()[k]).sum().item()
        for k in rand.state_dict() if k.startswith("encoder."))
    assert diffs > 0, "STRICT-PRE encoder identical to random init"


# ------------------------------------------------ evaluation lock / no-ow
def test_evaluate_refuses_without_flag():
    with pytest.raises(SystemExit, match="locked"):
        runner.stage_evaluate("STRICT-RAND", 42, None, None, confirmed=False)


def test_evaluate_refuses_with_flag_but_incomplete_training(monkeypatch,
                                                            tmp_path):
    monkeypatch.setattr(runner, "CKPT_ROOT", tmp_path)  # no records exist
    with pytest.raises(SystemExit, match="before all 72"):
        runner.stage_evaluate("STRICT-RAND", 42, None, None, confirmed=True)


def test_train_rerun_refuses(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "CKPT_ROOT", tmp_path)
    inst = runner.FOLD_INSTANCES[0]
    d = tmp_path / "STRICT-RAND/seed42" / inst.replace("/", "_")
    d.mkdir(parents=True)
    (d / "record.json").write_text("{}")
    with pytest.raises(SystemExit, match="REFUSED"):
        runner.stage_train("STRICT-RAND", 42, None, None)


# ------------------------------------------------- aggregator (fixtures)
def _metrics(macro: float) -> dict:
    return {"macro_f1": macro, "bb_f1": macro, "accuracy": macro,
            "outer_race_recall": macro,
            "confusion": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "n": 3}


def _write_fixture(eval_dir: Path, pre_vals: dict, rand_vals: dict) -> None:
    for arm, vals in (("STRICT-PRE", pre_vals), ("STRICT-RAND", rand_vals)):
        for inst in agg.FOLD_INSTANCES:
            for seed in SEEDS:
                p = (eval_dir / arm / f"seed{seed}"
                     / f"{inst.replace('/', '_')}.json")
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps({
                    "arm": arm, "seed": seed, "fold_instance": inst,
                    "test": _metrics(vals[inst]),
                    "per_bearing": {"K001": {"true_class": "normal",
                                             "n": 3, "n_correct": 2,
                                             "recall": 2 / 3}}}) + "\n")


def test_aggregator_win_case(tmp_path):
    pre = {inst: 0.50 + 0.03 for inst in agg.FOLD_INSTANCES}
    rand = {inst: 0.50 for inst in agg.FOLD_INSTANCES}
    r = agg.aggregate(_fix(tmp_path, pre, rand))
    assert r["primary"]["verdict"] == "WIN"
    assert r["primary"]["p_two_sided_exact"] < 0.05


def test_aggregator_loss_case(tmp_path):
    pre = {inst: 0.50 - 0.05 for inst in agg.FOLD_INSTANCES}
    rand = {inst: 0.50 for inst in agg.FOLD_INSTANCES}
    r = agg.aggregate(_fix(tmp_path, pre, rand))
    assert r["primary"]["verdict"] == "LOSS"


def test_aggregator_tie_nonsignificant(tmp_path):
    pre = {inst: 0.50 + (0.05 if i % 2 else -0.05)
           for i, inst in enumerate(agg.FOLD_INSTANCES)}
    rand = {inst: 0.50 for inst in agg.FOLD_INSTANCES}
    r = agg.aggregate(_fix(tmp_path, pre, rand))
    assert r["primary"]["verdict"] == "TIE"


def test_aggregator_significant_but_small_is_tie(tmp_path):
    pre = {inst: 0.50 + 0.005 for inst in agg.FOLD_INSTANCES}
    rand = {inst: 0.50 for inst in agg.FOLD_INSTANCES}
    r = agg.aggregate(_fix(tmp_path, pre, rand))
    assert r["primary"]["p_two_sided_exact"] < 0.05
    assert r["primary"]["verdict"] == "TIE"
    assert "fails the frozen meaningful-effect threshold" in \
        r["primary"]["interpretation"]


def test_aggregator_refuses_missing_input(tmp_path):
    pre = {inst: 0.53 for inst in agg.FOLD_INSTANCES}
    rand = {inst: 0.50 for inst in agg.FOLD_INSTANCES}
    eval_dir = _fix(tmp_path, pre, rand)
    victim = (eval_dir / "STRICT-PRE/seed43"
              / f"{agg.FOLD_INSTANCES[5].replace('/', '_')}.json")
    victim.unlink()
    with pytest.raises(SystemExit, match="REFUSED"):
        agg.aggregate(eval_dir)


def test_aggregator_framing_embedded(tmp_path):
    pre = {inst: 0.53 for inst in agg.FOLD_INSTANCES}
    rand = {inst: 0.50 for inst in agg.FOLD_INSTANCES}
    r = agg.aggregate(_fix(tmp_path, pre, rand))
    assert "not fully independent" in r["framing"]


def _fix(tmp_path: Path, pre: dict, rand: dict) -> Path:
    eval_dir = tmp_path / "eval"
    _write_fixture(eval_dir, pre, rand)
    return eval_dir


# --------------------------------------------------- overlap audit guard
def test_overlap_audit_exists_and_confirms():
    doc = json.loads(
        (REPO_ROOT / "results/amendment6/overlap_audit.json").read_text())
    assert doc["verdict"].startswith("OVERLAP CONFIRMED")
    assert doc["selection_surface_equals_development_pool"]
    assert all(v["fully_contained"]
               for v in doc["per_fold_instance"].values())
    assert doc["sealed_bearings_present_anywhere"] == []
