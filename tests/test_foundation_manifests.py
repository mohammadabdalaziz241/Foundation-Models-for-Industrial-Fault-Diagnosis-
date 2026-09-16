import csv
import json
from pathlib import Path

from scripts.build_foundation_manifests import build, parse_args
from src.foundation_manifest import iter_cwru, iter_paderborn, iter_xjtu


def _rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_raw_adapters_preserve_required_metadata():
    c = next(iter_cwru(Path("data/raw"), 1))
    p = next(iter_paderborn(Path("data/raw_paderborn"), 1))
    x = next(iter_xjtu(Path("data/raw_xjtu_sy"), 1))
    assert c.dataset_class_label and c.bearing_id and c.recording_id and c.operating_condition
    assert p.dataset_class_label in {"normal", "inner_race", "outer_race"}
    assert p.bearing_id and p.snapshot_index is not None
    assert x.dataset_class_label is None and x.shared_class_label is None
    assert x.supervised_eligible is False and x.final_failure_element
    assert x.channel == "Horizontal_vibration_signals"


def test_small_manifest_is_deterministic_isolated_and_role_correct(tmp_path):
    target = tmp_path / "seed42"
    args = parse_args(["--seed", "42", "--limit-per-dataset", "1",
        "--max-near-comparisons", "10000", "--output-root", str(target)])
    path, first = build(args)
    _, second = build(args)  # immutable idempotent verification
    assert first == second
    checksums = json.loads((path / "manifest_checksums.json").read_text())
    assert checksums
    all_rows = _rows(path / "all_samples.csv")
    ssl = _rows(path / "train_ssl.csv")
    supervised = _rows(path / "train_supervised.csv") + _rows(path / "validation_supervised.csv") + _rows(path / "test_supervised.csv")
    assert len({r["sample_id"] for r in all_rows}) == len(all_rows)
    assert all(r["window_length"] == "1024" for r in all_rows)
    assert all(r["split"] == "train" and r["ssl_eligible"] == "true" for r in ssl)
    assert {r["dataset"] for r in ssl} == {"cwru", "paderborn", "xjtu_sy"}
    assert {r["dataset"] for r in supervised} == {"cwru", "paderborn"}
    xjtu = [r for r in all_rows if r["dataset"] == "xjtu_sy"]
    assert xjtu and all(not r["dataset_class_label"] and not r["shared_class_label"] for r in xjtu)
    assert all(r["final_failure_element"] for r in xjtu)
    assert first["raw_interval_cross_split_violations"] == 0
    assert first["exact_duplicate_cross_split_violations"] == 0
    stats = json.loads((path / "normalization_stats.json").read_text())
    assert all(not values["validation_or_test_used"] for channels in stats["datasets"].values() for values in channels.values())


def test_seed_changes_temporal_assignment(tmp_path):
    outputs = []
    for seed in (42, 43):
        args = parse_args(["--seed", str(seed), "--limit-per-dataset", "1",
            "--max-near-comparisons", "1000", "--output-root", str(tmp_path / f"seed{seed}")])
        path, _ = build(args)
        outputs.append({(r["sample_id"].split(":", 2)[1], r["split"], r["raw_start"]) for r in _rows(path / "all_samples.csv")})
    assert outputs[0] != outputs[1]


def test_small_grouped_manifest_is_group_isolated_and_s2_extends_s1(tmp_path):
    target=tmp_path/"grouped"
    args=parse_args(["--protocol","grouped_generalization_v1","--seed","42","--limit-per-dataset","5","--output-root",str(target)])
    path,summary=build(args)
    rows=_rows(path/"all_samples.csv")
    key=lambda r:r["recording_id"] if r["dataset"]=="cwru" else r["bearing_id"]
    seen={}
    for r in rows:
        identity=(r["dataset"],key(r));seen.setdefault(identity,set()).add(r["split"])
    assert all(len(v)==1 for v in seen.values())
    s1t=_rows(path/"train_ssl_s1.csv");s2t=_rows(path/"train_ssl_s2.csv")
    assert {r["sample_id"] for r in s1t}<={r["sample_id"] for r in s2t}
    assert all(r["dataset"]=="xjtu_sy" for r in s2t if r["sample_id"] not in {q["sample_id"] for q in s1t})
    assert summary["checks"]["s2_equals_s1_plus_xjtu"]
    subsets=[{r["sample_id"] for r in _rows(path/f"train_supervised_labelled_{p}pct.csv")} for p in (10,25,50,100)]
    assert all(a<=b for a,b in zip(subsets,subsets[1:]))
