"""SMX-1 pre-run test battery (docs/strict_matrix_protocol.md §9).
All tests run BEFORE any SMX-1 training exists, CPU-only, file-by-file:

    .venv/bin/python -m pytest tests/test_strict_matrix.py -q

Planted positive controls prove every guard fires; zero-hit token scans
carry in-invocation positive controls (house rule: Python `re`, never bare
`grep`).
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import inspect
import json
import re
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


runner = _load("run_strict_matrix")
agg = _load("aggregate_strict_matrix")
a6 = _load("run_amendment6")

META = REPO_ROOT / "metadata/vibrationclip_v1/downstream"
META_PRE = REPO_ROOT / "metadata/vibrationclip_v1/pretraining"
SEEDS = (42, 43, 44)
CPU = torch.device("cpu")


def _rows(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ------------------------------------------- benchmark byte-identity (V4)
def test_benchmark_manifests_match_committed_checksums():
    recorded = json.loads(
        (REPO_ROOT / "metadata/vibrationclip_v1/manifest_checksums.json")
        .read_text())["files"]
    checked = 0
    for inst in runner.FOLD_INSTANCES:
        for name in ("label_blocks/Rfull.csv", "validation.csv",
                     "setting_b_eval.csv"):
            rel = f"downstream/{inst}/{name}"
            assert rel in recorded, f"missing checksum entry for {rel}"
            assert _sha(REPO_ROOT / "metadata/vibrationclip_v1" / rel) == \
                recorded[rel], f"benchmark manifest altered: {rel}"
            checked += 1
    assert checked == 36


def test_pretraining_manifests_match_committed_checksums():
    recorded = json.loads(
        (REPO_ROOT / "metadata/vibrationclip_v1/manifest_checksums.json")
        .read_text())["files"]
    for name in ("s1_train.csv", "s1_val.csv", "s2s3_train.csv",
                 "s2s3_val.csv"):
        rel = f"pretraining/{name}"
        assert _sha(REPO_ROOT / "metadata/vibrationclip_v1" / rel) == \
            recorded[rel], f"pretraining manifest altered: {rel}"


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


# --------------------------- Paderborn absence in pretraining manifests
PADERBORN_TOKEN = re.compile(r"(?:^|[^A-Za-z0-9])(K[AIB]?\d{2}|paderborn)",
                             re.IGNORECASE)


def _paderborn_hits(rows: list[dict]) -> list[str]:
    hits = []
    for r in rows:
        for col in ("bearing_id", "dataset", "recording_id", "sample_id"):
            v = r.get(col, "")
            if v and PADERBORN_TOKEN.search(v):
                hits.append(f"{col}={v}")
    return hits


def test_no_paderborn_in_pretraining_manifests_with_positive_control():
    # positive control first: a planted Paderborn row MUST be detected
    planted = [{"bearing_id": "KA04", "dataset": "paderborn",
                "recording_id": "N15_M07_F10_KA04_1", "sample_id": "x"}]
    assert _paderborn_hits(planted), "positive control failed to fire"
    for name in ("s1_train.csv", "s1_val.csv", "s2s3_train.csv",
                 "s2s3_val.csv"):
        hits = _paderborn_hits(_rows(META_PRE / name))
        assert not hits, f"Paderborn token in {name}: {hits[:3]}"


# ------------------------------------------------ pins and checkpoint IDs
def test_reused_arm_pins_match_records_and_files():
    for arm in runner.REUSED_ARMS:
        for seed in SEEDS:
            d = REPO_ROOT / f"models/saved_withinbearing/{arm}_seed{seed}"
            record = json.loads((d / "record.json").read_text())
            recorded = record["files_sha256"]["encoder.pt"]
            assert recorded == runner.REUSED_PINS[arm][seed]
            assert _sha(d / "encoder.pt") == recorded
            assert record["arm"] == arm and int(record["seed"]) == seed


def _synthetic_ckpt(d: Path, arm: str, seed: int, extra: dict | None = None):
    d.mkdir(parents=True)
    torch.save({"w": torch.zeros(1)}, d / "encoder.pt")
    d_record = {"arm": arm, "seed": seed,
                "files_sha256": {"encoder.pt": _sha(d / "encoder.pt")}}
    d_record.update(extra or {})
    (d / "record.json").write_text(json.dumps(d_record))
    return d_record


def test_tampered_pin_refuses(tmp_path, monkeypatch):
    _synthetic_ckpt(tmp_path / "S1p_seed42", "S1p", 42)
    monkeypatch.setattr(runner, "WB_CKPT_ROOT", tmp_path)
    with pytest.raises(SystemExit, match="pin"):
        runner.build_model("S1p", 42, CPU)


def test_wrong_objective_checkpoint_refuses(tmp_path, monkeypatch):
    _synthetic_ckpt(tmp_path / "S1p_seed42", "S4p", 42)  # wrong arm inside
    monkeypatch.setattr(runner, "WB_CKPT_ROOT", tmp_path)
    with pytest.raises(SystemExit, match="wrong[ -]objective|arm"):
        runner.build_model("S1p", 42, CPU)


def test_wrong_seed_checkpoint_refuses(tmp_path, monkeypatch):
    _synthetic_ckpt(tmp_path / "S1p_seed42", "S1p", 43)  # wrong seed inside
    monkeypatch.setattr(runner, "WB_CKPT_ROOT", tmp_path)
    with pytest.raises(SystemExit, match="wrong seed|seed"):
        runner.build_model("S1p", 42, CPU)


def test_s2p_train_refuses_without_pins_file(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "PINS_FILE", tmp_path / "nonexistent.json")
    with pytest.raises(SystemExit, match="pins"):
        runner.build_model("S2p", 42, CPU)


def test_tampered_s2p_pins_refuse(tmp_path, monkeypatch):
    rec = _synthetic_ckpt(tmp_path / "pre" / "S2p_seed42", "S2p", 42)
    pins = {"pins": {"S2p": {"42": "0" * 64}, "S3p": {}}}  # wrong pin
    (tmp_path / "pins.json").write_text(json.dumps(pins))
    monkeypatch.setattr(runner, "PINS_FILE", tmp_path / "pins.json")
    monkeypatch.setattr(runner, "PRETRAIN_ROOT", tmp_path / "pre")
    assert rec["files_sha256"]["encoder.pt"] != "0" * 64
    with pytest.raises(SystemExit, match="pin"):
        runner.build_model("S2p", 42, CPU)


def test_matrix_closed_unknown_arm_refuses():
    with pytest.raises(SystemExit, match="closed|unknown"):
        runner.build_model("S5p", 42, CPU)   # frozen reference, not SMX-1
    with pytest.raises(SystemExit, match="closed|unknown"):
        runner.build_model("S8p", 42, CPU)   # no arm beyond the six


# ------------------------------------ parameter / state-shape equality
def test_parameter_and_shape_equality_all_reused_arms():
    from src.withinbearing.encoder2 import Classifier2
    from src.cv_paderborn import seed_everything
    seed_everything(42)
    rand = Classifier2()
    ref_shapes = {k: tuple(v.shape) for k, v in rand.state_dict().items()}
    assert sum(p.numel() for p in rand.parameters()) == \
        runner.EXPECTED_PARAMS == 12_451_395
    for arm in runner.REUSED_ARMS:
        model, sha = runner.build_model(arm, 42, CPU)
        assert sha == runner.REUSED_PINS[arm][42]
        assert sum(p.numel() for p in model.parameters()) == \
            runner.EXPECTED_PARAMS
        assert {k: tuple(v.shape)
                for k, v in model.state_dict().items()} == ref_shapes
        diffs = sum(
            (rand.state_dict()[k] != model.state_dict()[k]).sum().item()
            for k in rand.state_dict() if k.startswith("encoder."))
        assert diffs > 0, f"{arm} encoder identical to random init"


# ------------------------------- recipe identity with frozen Amendment 6
def test_recipe_constants_equal_frozen_amendment6_runner():
    assert runner.FULL == a6.FULL
    assert runner.FOLD_INSTANCES == a6.FOLD_INSTANCES
    assert runner.EXPECTED_PARAMS == a6.EXPECTED_PARAMS
    assert runner.CLASS_TO_INDEX == a6.CLASS_TO_INDEX
    assert runner.SEEDS == a6.SEEDS
    assert runner.META == a6.META


def test_pretrain_config_matches_frozen_definition():
    from src.vibrationclip.objectives import LearnableTemperature
    assert runner.PRE == {"batch": 64, "lr": 1e-3, "steps": 6_000,
                          "eval_every": 250, "sampler_offset": 100}
    t = LearnableTemperature()
    assert abs(float(t().detach()) - 0.07) < 1e-6 and t.floor == 0.01


def test_training_never_references_test_manifest():
    assert "setting_b" not in inspect.getsource(runner.stage_train)
    assert "setting_b" not in inspect.getsource(runner.stage_pretrain)
    assert "setting_b_eval" in inspect.getsource(runner.stage_evaluate)
    assert "validation.csv" in inspect.getsource(runner.stage_train)


# ------------------------------------------- frozen sampler-stream policy
def test_s2p_s3p_stream_matches_frozen_s4p_s5p_all_seeds():
    lab_rows = _rows(META_PRE / "s2s3_train.csv")
    for seed in SEEDS:
        stream, got = runner.assert_stream_matches_frozen_pair(lab_rows,
                                                               seed)
        assert len(stream) == runner.PRE["steps"]
        assert got == runner.FROZEN_PAIR_STREAM_SHAS[seed]
        for arm in ("S4p", "S5p"):
            rec = json.loads(
                (REPO_ROOT / f"models/saved_withinbearing/{arm}_seed{seed}"
                 / "record.json").read_text())
            assert rec["labelled_stream_sha"] == got


def test_mutated_sampler_offset_is_detected(monkeypatch):
    lab_rows = _rows(META_PRE / "s2s3_train.csv")
    monkeypatch.setitem(runner.PRE, "sampler_offset", 101)  # planted drift
    with pytest.raises(SystemExit, match="stream"):
        runner.assert_stream_matches_frozen_pair(lab_rows, 42)


# ------------------------------------------------ evaluation lock / no-ow
def test_evaluate_refuses_without_flag():
    with pytest.raises(SystemExit, match="locked"):
        runner.stage_evaluate("S1p", 42, None, None, confirmed=False)


def test_evaluate_refuses_with_flag_but_incomplete_training(tmp_path,
                                                            monkeypatch):
    monkeypatch.setattr(runner, "DS_ROOT", tmp_path)  # no records exist
    with pytest.raises(SystemExit, match="before all 216"):
        runner.stage_evaluate("S1p", 42, None, None, confirmed=True)


def _plant_all_216_records(root: Path) -> None:
    for arm in runner.ARMS:
        for seed in SEEDS:
            for inst in runner.FOLD_INSTANCES:
                d = root / arm / f"seed{seed}" / inst.replace("/", "_")
                d.mkdir(parents=True, exist_ok=True)
                (d / "record.json").write_text("{}")


def test_evaluate_refuses_existing_output(tmp_path, monkeypatch):
    _plant_all_216_records(tmp_path / "ds")
    monkeypatch.setattr(runner, "DS_ROOT", tmp_path / "ds")
    results = tmp_path / "res"
    monkeypatch.setattr(runner, "RESULTS", results)
    # keep the real frozen-tree gate: point the snapshot at the real file
    inst0 = runner.FOLD_INSTANCES[0].replace("/", "_")
    out = results / "eval" / "S1p" / "seed42" / f"{inst0}.json"
    out.parent.mkdir(parents=True)
    out.write_text("{}")
    monkeypatch.setattr(
        runner, "FROZEN_A6_SNAPSHOT",
        REPO_ROOT / "results/strict_matrix_v1/frozen_amendment6_checksums.json")
    with pytest.raises(SystemExit, match="exactly once"):
        runner.stage_evaluate("S1p", 42, None, None, confirmed=True)


def test_train_rerun_refuses(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "DS_ROOT", tmp_path)
    inst = runner.FOLD_INSTANCES[0]
    d = tmp_path / "S1p/seed42" / inst.replace("/", "_")
    d.mkdir(parents=True)
    (d / "record.json").write_text("{}")
    with pytest.raises(SystemExit, match="REFUSED"):
        runner.stage_train("S1p", 42, None, None)


def test_pretrain_rerun_refuses(tmp_path):
    d = tmp_path / "S2p_seed42"
    d.mkdir(parents=True)
    (d / "record.json").write_text("{}")
    with pytest.raises(SystemExit, match="REFUSED"):
        runner.stage_pretrain("S2p", 42, None, None, None, ckpt_root=tmp_path)


def test_pretrain_refuses_non_new_arms():
    for arm in ("S1p", "S4p", "S5p", "S0"):
        with pytest.raises(SystemExit, match="REFUSED"):
            runner.stage_pretrain(arm, 42, None, None, None)


# ------------------------------------------- frozen Amendment 6 tree gate
def test_frozen_a6_snapshot_matches_live_tree():
    runner.verify_frozen_amendment6_tree()   # must pass on the real tree


def test_frozen_a6_tamper_detected(tmp_path, monkeypatch):
    src = REPO_ROOT / "results/amendment6"
    dst = tmp_path / "amendment6"
    for p in src.rglob("*"):
        if p.is_file():
            q = dst / p.relative_to(src)
            q.parent.mkdir(parents=True, exist_ok=True)
            q.write_bytes(p.read_bytes())
    victim = dst / "aggregate.json"
    victim.write_text(victim.read_text() + " ")   # planted 1-byte change
    monkeypatch.setattr(runner, "FROZEN_A6_RESULTS", dst)
    with pytest.raises(SystemExit, match="differs"):
        runner.verify_frozen_amendment6_tree()


def test_frozen_a6_missing_file_detected(tmp_path, monkeypatch):
    src = REPO_ROOT / "results/amendment6"
    dst = tmp_path / "amendment6"
    files = [p for p in src.rglob("*") if p.is_file()]
    for p in files[:-1]:                          # drop one file
        q = dst / p.relative_to(src)
        q.parent.mkdir(parents=True, exist_ok=True)
        q.write_bytes(p.read_bytes())
    monkeypatch.setattr(runner, "FROZEN_A6_RESULTS", dst)
    with pytest.raises(SystemExit, match="differs"):
        runner.verify_frozen_amendment6_tree()


def test_smx_roots_disjoint_from_frozen_roots():
    frozen = [REPO_ROOT / "models/saved_withinbearing",
              REPO_ROOT / "models/saved_amendment6",
              REPO_ROOT / "models/saved_vibrationclip",
              REPO_ROOT / "results/amendment6",
              REPO_ROOT / "results/withinbearing_v1",
              REPO_ROOT / "results/vibrationclip_v1",
              REPO_ROOT / "metadata"]
    writable = [runner.PRETRAIN_ROOT, runner.DS_ROOT, runner.RESULTS]
    for w in writable:
        for f in frozen:
            assert not w.is_relative_to(f), f"{w} inside frozen {f}"
            assert not f.is_relative_to(w), f"frozen {f} inside {w}"


# ----------------------------------------------------- freeze-pins stage
def _plant_pretrains(root: Path, stream_flag=True, skip=()):
    for arm in ("S2p", "S3p"):
        for seed in SEEDS:
            if (arm, seed) in skip:
                continue
            _synthetic_ckpt(
                root / f"{arm}_seed{seed}", arm, seed,
                {"labelled_stream_matches_frozen_s4p_s5p": stream_flag})


def test_freeze_pins_writes_and_runs_once(tmp_path):
    root, pins = tmp_path / "pre", tmp_path / "pins.json"
    _plant_pretrains(root)
    runner.stage_freeze_pins(ckpt_root=root, pins_file=pins)
    doc = json.loads(pins.read_text())
    for arm in ("S2p", "S3p"):
        for seed in SEEDS:
            want = json.loads(
                (root / f"{arm}_seed{seed}" / "record.json").read_text()
            )["files_sha256"]["encoder.pt"]
            assert doc["pins"][arm][str(seed)] == want
    with pytest.raises(SystemExit, match="exactly once"):
        runner.stage_freeze_pins(ckpt_root=root, pins_file=pins)


def test_freeze_pins_refuses_incomplete(tmp_path):
    root, pins = tmp_path / "pre", tmp_path / "pins.json"
    _plant_pretrains(root, skip={("S3p", 44)})
    with pytest.raises(SystemExit, match="incomplete"):
        runner.stage_freeze_pins(ckpt_root=root, pins_file=pins)


def test_freeze_pins_refuses_unverified_stream(tmp_path):
    root, pins = tmp_path / "pre", tmp_path / "pins.json"
    _plant_pretrains(root, stream_flag=None)
    with pytest.raises(SystemExit, match="stream gate"):
        runner.stage_freeze_pins(ckpt_root=root, pins_file=pins)


# ------------------------------------------------- aggregator (fixtures)
def _metrics(macro: float) -> dict:
    per_class = {c: {"precision": macro, "recall": macro, "f1": macro,
                     "support": 3} for c in
                 ("normal", "inner_race", "outer_race")}
    return {"macro_f1": macro, "bb_f1": macro, "accuracy": macro,
            "per_class": per_class, "outer_race_recall": macro,
            "confusion": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "n": 3}


def _eval_doc(arm, seed, inst, macro):
    return {"arm": arm, "seed": seed, "fold_instance": inst,
            "test": _metrics(macro),
            "per_bearing": {"K001": {"true_class": "normal", "n": 3,
                                     "n_correct": 2, "recall": 2 / 3}}}


def _fixture(tmp_path: Path, new_vals: dict, ref_vals: dict):
    """new_vals: {arm: {inst: macro}}; ref_vals: {'STRICT-PRE'|'STRICT-RAND':
    {inst: macro}}. Builds eval tree, frozen a6 tree + snapshot."""
    eval_dir = tmp_path / "eval"
    for arm, per_inst in new_vals.items():
        for inst, macro in per_inst.items():
            for seed in SEEDS:
                p = (eval_dir / arm / f"seed{seed}"
                     / f"{inst.replace('/', '_')}.json")
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps(
                    _eval_doc(arm, seed, inst, macro)) + "\n")
    a6_dir = tmp_path / "a6"
    matrix = {}
    for a6arm, per_inst in ref_vals.items():
        matrix[a6arm] = dict(per_inst)
        for inst, macro in per_inst.items():
            for seed in SEEDS:
                p = (a6_dir / "eval" / a6arm / f"seed{seed}"
                     / f"{inst.replace('/', '_')}.json")
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps(
                    _eval_doc(a6arm, seed, inst, macro)) + "\n")
    (a6_dir / "aggregate.json").write_text(json.dumps(
        {"arm_instance_matrix": matrix}))
    snap = tmp_path / "snapshot.json"
    files = {str(p.relative_to(a6_dir)):
             hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(a6_dir.rglob("*")) if p.is_file()}
    snap.write_text(json.dumps({"files": files}))
    return eval_dir, a6_dir, snap


def _full_fixture(tmp_path):
    new_vals = {arm: {inst: 0.40 + 0.01 * i
                      for inst in agg.FOLD_INSTANCES}
                for i, arm in enumerate(agg.ARMS)}
    ref_vals = {"STRICT-PRE": {inst: 0.39 for inst in agg.FOLD_INSTANCES},
                "STRICT-RAND": {inst: 0.28 for inst in agg.FOLD_INSTANCES}}
    return _fixture(tmp_path, new_vals, ref_vals)


def test_aggregator_structure_and_framing(tmp_path):
    eval_dir, a6_dir, snap = _full_fixture(tmp_path)
    r = agg.aggregate(eval_dir, a6_dir=a6_dir, snapshot=snap)
    assert r["inferential_tests"] == "none (frozen SMX-1 plan)"
    assert "scope was extended" in r["framing"]["scope_statement"]
    assert "does not alter the status" in r["framing"]["status_statement"]
    assert len(r["descriptive_ranking_by_mean_macro_f1"]) == 8
    for arm in agg.ARMS:
        for ref in ("S0", "S5p"):
            block = r["paired_differences"][arm][f"vs_{ref}"]
            assert {"mean_difference", "bootstrap", "paired_d_z",
                    "n_positive_instances"} <= set(block)
    # frozen-reference passthrough: verbatim copies of the frozen matrix
    assert all(abs(r["arm_instance_matrix_macro_f1"]["S5p"][i] - 0.39) < 1e-12
               for i in agg.FOLD_INSTANCES)
    assert all(abs(r["arm_instance_matrix_macro_f1"]["S0"][i] - 0.28) < 1e-12
               for i in agg.FOLD_INSTANCES)
    blob = json.dumps(r)
    for forbidden in ("wilcoxon", "p_value", "p_two_sided", "verdict",
                      "significan"):
        assert forbidden not in blob.lower(), forbidden


def test_aggregator_bootstrap_deterministic(tmp_path):
    eval_dir, a6_dir, snap = _full_fixture(tmp_path)
    r1 = agg.aggregate(eval_dir, a6_dir=a6_dir, snapshot=snap)
    r2 = agg.aggregate(eval_dir, a6_dir=a6_dir, snapshot=snap)
    assert r1["paired_differences"] == r2["paired_differences"]


def test_aggregator_refuses_missing_input(tmp_path):
    eval_dir, a6_dir, snap = _full_fixture(tmp_path)
    victim = (eval_dir / "S3p/seed43"
              / f"{agg.FOLD_INSTANCES[5].replace('/', '_')}.json")
    victim.unlink()
    with pytest.raises(SystemExit, match="REFUSED"):
        agg.aggregate(eval_dir, a6_dir=a6_dir, snapshot=snap)


def test_aggregator_refuses_tampered_frozen_tree(tmp_path):
    eval_dir, a6_dir, snap = _full_fixture(tmp_path)
    victim = a6_dir / "aggregate.json"
    victim.write_text(victim.read_text() + " ")
    with pytest.raises(SystemExit, match="REFUSED"):
        agg.aggregate(eval_dir, a6_dir=a6_dir, snapshot=snap)


def test_aggregator_refuses_inconsistent_frozen_aggregate(tmp_path):
    new_vals = {arm: {inst: 0.40 for inst in agg.FOLD_INSTANCES}
                for arm in agg.ARMS}
    ref_vals = {"STRICT-PRE": {inst: 0.39 for inst in agg.FOLD_INSTANCES},
                "STRICT-RAND": {inst: 0.28 for inst in agg.FOLD_INSTANCES}}
    eval_dir, a6_dir, snap = _fixture(tmp_path, new_vals, ref_vals)
    doc = json.loads((a6_dir / "aggregate.json").read_text())
    doc["arm_instance_matrix"]["STRICT-PRE"][agg.FOLD_INSTANCES[0]] = 0.50
    (a6_dir / "aggregate.json").write_text(json.dumps(doc))
    files = {str(p.relative_to(a6_dir)):
             hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(a6_dir.rglob("*")) if p.is_file()}
    snap.write_text(json.dumps({"files": files}))
    with pytest.raises(SystemExit, match="inconsistent"):
        agg.aggregate(eval_dir, a6_dir=a6_dir, snapshot=snap)


def test_aggregator_main_refuses_existing_output(tmp_path, monkeypatch):
    eval_dir, a6_dir, snap = _full_fixture(tmp_path)
    (eval_dir.parent / "aggregate.json").write_text("{}")
    monkeypatch.setattr(sys, "argv",
                        ["aggregate_strict_matrix.py",
                         "--eval-dir", str(eval_dir)])
    with pytest.raises(SystemExit, match="exactly once"):
        agg.main()
