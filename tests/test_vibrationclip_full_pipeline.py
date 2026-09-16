"""Gate D structural tests: checkpoint pinning/refusal, evaluation schema,
and the frozen aggregation - exercised on smoke-grade fixtures BEFORE any
real result exists (Gate C approval rider).

Run file-by-file (repo convention): pytest tests/test_vibrationclip_full_pipeline.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.aggregate_vibrationclip as agg  # noqa: E402
import scripts.run_vibrationclip as rv  # noqa: E402
from src.vibrationclip.encoder import (  # noqa: E402
    Classifier, ProjectionHead, SpectrogramEncoder, TextProjection,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
TWO_INSTANCES = ("seed42/fold0", "seed42/fold1")


# ------------------------------------------------------- pinning + refusal
def test_save_and_load_pinned_roundtrip(tmp_path):
    enc = SpectrogramEncoder()
    rv.save_pinned(tmp_path / "ck", {"encoder": enc.state_dict()},
                   {"arm": "T", "seed": 0})
    states, record = rv.load_pinned(tmp_path / "ck", ["encoder"])
    assert set(states["encoder"]) == set(enc.state_dict())
    assert "encoder.pt" in record["files_sha256"]


def test_load_pinned_refuses_missing(tmp_path):
    with pytest.raises(SystemExit, match="REFUSED"):
        rv.load_pinned(tmp_path / "absent", ["encoder"])


def test_load_pinned_refuses_tampered_checkpoint(tmp_path):
    enc = SpectrogramEncoder()
    rv.save_pinned(tmp_path / "ck", {"encoder": enc.state_dict()}, {})
    p = tmp_path / "ck" / "encoder.pt"
    data = bytearray(p.read_bytes())
    data[len(data) // 2] ^= 0xFF
    p.write_bytes(bytes(data))
    with pytest.raises(SystemExit, match="hash mismatch"):
        rv.load_pinned(tmp_path / "ck", ["encoder"])


def test_pretrain_refuses_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(rv, "CKPT_ROOT", tmp_path)
    (tmp_path / "S1_seed42").mkdir(parents=True)
    (tmp_path / "S1_seed42" / "record.json").write_text("{}")
    with pytest.raises(SystemExit, match="already exists"):
        rv.full_pretrain("S1", 42, None, None, torch.device("cpu"))


# ------------------------------------- evaluation schema via the real code
@pytest.fixture(scope="module")
def eval_tree(tmp_path_factory):
    """Random-weight pinned checkpoints -> real stage_evaluate -> schema
    JSONs for 2 fold-instances, 1 seed. No training happens."""
    root = tmp_path_factory.mktemp("gate_d_schema")
    ckpt_root, results = root / "ckpts", root / "results"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    for arm in ("S1", "S3"):
        tensors = {"encoder": SpectrogramEncoder().state_dict()}
        if arm == "S3":
            tensors["projection"] = ProjectionHead().state_dict()
            tensors["text_projection"] = TextProjection().state_dict()
        rv.save_pinned(ckpt_root / f"{arm}_seed42", tensors,
                       {"arm": arm, "seed": 42, "fixture": True})
    for inst in TWO_INSTANCES:
        for regime in ("R1", "R5", "Rfull"):
            rv.save_pinned(
                ckpt_root / "s0" / regime / "seed42" / inst.replace("/", "_"),
                {"classifier": Classifier().state_dict()},
                {"arm": "S0", "regime": regime, "seed": 42,
                 "val_macro_f1": 0.0, "fixture": True})

    import contextlib

    @contextlib.contextmanager
    def patched():
        old = (rv.CKPT_ROOT, rv.RESULTS, rv.FOLD_INSTANCES)
        rv.CKPT_ROOT, rv.RESULTS = ckpt_root, results
        rv.FOLD_INSTANCES = TWO_INSTANCES
        try:
            yield
        finally:
            rv.CKPT_ROOT, rv.RESULTS, rv.FOLD_INSTANCES = old

    store, text_store = rv.SpectrogramStore(), rv.TextStore()
    with patched():
        for arm in ("S1", "S3"):
            rv.stage_evaluate("probe", arm, 42, store, text_store, device,
                              smoke_subset=48)
        rv.stage_evaluate("s0", None, 42, store, text_store, device,
                          smoke_subset=48)
        rv.stage_evaluate("zeroshot", "S3", 42, store, text_store, device,
                          smoke_subset=48)
    return results / "full" / "eval"


def test_probe_schema(eval_tree):
    p = eval_tree / "probe" / "S1" / "seed42" / "seed42_fold0.json"
    d = json.loads(p.read_text())
    assert d["encoder_sha256"]
    for regime in ("R1", "R5", "Rfull"):
        for split in ("validation", "setting_a_eval", "setting_b_eval"):
            m = d["regimes"][regime][split]
            assert set(m) >= {"macro_f1", "bb_f1", "per_class",
                              "outer_race_recall", "confusion", "n"}
            assert m["outer_race_recall"] == \
                m["per_class"]["outer_race"]["recall"]
            assert np.asarray(m["confusion"]).shape == (3, 3)


def test_s0_and_zeroshot_schema(eval_tree):
    d = json.loads((eval_tree / "s0" / "Rfull" / "seed42"
                    / "seed42_fold1.json").read_text())
    assert d["classifier_sha256"] and "setting_b_eval" in d
    z = json.loads((eval_tree / "zeroshot" / "S3" / "seed42"
                    / "seed42_fold0.json").read_text())
    assert set(z["families"]) == {"basic", "physics"}


def test_aggregator_allow_partial_on_real_schema(eval_tree, tmp_path, capsys):
    out = tmp_path / "agg.json"
    sys.argv = ["aggregate_vibrationclip.py", "--eval-root", str(eval_tree),
                "--out", str(out), "--allow-partial"]
    agg.main()
    d = json.loads(out.read_text())
    assert d["status"].startswith("NOT-A-RESULT")
    assert d["n_missing_inputs"] > 0
    assert [t["contrast"] for t in d["pre_registered_inference"]["tests"]] == \
        ["S4 vs S0", "S3 vs S2", "S4 vs S1"]


def test_aggregator_refuses_partial_without_flag(eval_tree, tmp_path):
    sys.argv = ["aggregate_vibrationclip.py", "--eval-root", str(eval_tree),
                "--out", str(tmp_path / "x.json")]
    with pytest.raises(SystemExit, match="REFUSED"):
        agg.main()


# ------------------------------------------- frozen inference on full tree
def _synthetic_tree(root: Path, shift: dict[str, float]) -> None:
    rng = np.random.default_rng(0)
    for arm in agg.ARMS:
        for seed in agg.SEEDS:
            for inst in agg.FOLD_INSTANCES:
                base = 0.5 + shift.get(arm, 0.0) + rng.normal(0, 0.01)
                m = {"macro_f1": base, "bb_f1": base - 0.02,
                     "accuracy": base, "outer_race_recall": base - 0.1,
                     "per_class": {c: {"precision": base, "recall": base,
                                       "f1": base, "support": 10}
                                   for c in agg.CLASS_NAMES},
                     "confusion": [[10, 0, 0]] * 3, "n": 30}
                if arm == "S0":
                    p = (root / "s0" / "Rfull" / f"seed{seed}"
                         / f"{inst.replace('/', '_')}.json")
                    payload = {"setting_b_eval": m}
                else:
                    p = (root / "probe" / arm / f"seed{seed}"
                         / f"{inst.replace('/', '_')}.json")
                    payload = {"regimes": {"Rfull": {"setting_b_eval": m}}}
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps(payload))


def test_frozen_contrasts_on_synthetic_full_tree(tmp_path, capsys):
    # S4 clearly above S0 and S1; S3 == S2 in distribution
    _synthetic_tree(tmp_path, {"S4": 0.08, "S1": 0.01, "S2": 0.03, "S3": 0.03})
    out = tmp_path / "agg.json"
    sys.argv = ["aggregate_vibrationclip.py", "--eval-root", str(tmp_path),
                "--out", str(out)]
    agg.main()
    d = json.loads(out.read_text())
    assert d["status"] == "final"
    tests = {t["contrast"]: t for t in d["pre_registered_inference"]["tests"]}
    for t in tests.values():
        assert t["n_fold_instances"] == 12
        assert len(t["per_fold_differences"]) == 12
    # exact two-sided Wilcoxon floor at n=12 is 2/4096
    assert tests["S4 vs S0"]["p_raw"] == pytest.approx(2 / 4096, rel=1e-6)
    assert tests["S4 vs S1"]["p_raw"] == pytest.approx(2 / 4096, rel=1e-6)
    assert tests["S3 vs S2"]["p_raw"] > 0.05
    # Holm: smallest p multiplied by 3, monotone
    assert tests["S4 vs S0"]["p_holm"] == pytest.approx(3 * 2 / 4096, rel=1e-6)
    assert tests["S3 vs S2"]["p_holm"] >= tests["S4 vs S0"]["p_holm"]
    # OR-recall secondary extraction present for every arm
    orq = d["secondary_outer_race_question"]["primary_surface_or_recall"]
    assert set(orq) == set(agg.ARMS)
