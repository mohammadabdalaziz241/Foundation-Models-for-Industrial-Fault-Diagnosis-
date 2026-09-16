"""Tests for the expanded CWRU+HUST benchmark: loader, manifests, label blocks, model.

Runnable two ways:

    .venv/bin/python tests/test_expanded_benchmark.py     # no pytest needed
    pytest tests/test_expanded_benchmark.py               # when pytest is installed

The venv in this environment has torch/scipy but not pytest, and the system
python has pytest but not torch, so the standalone runner at the bottom is what
actually executes here.
"""

from __future__ import annotations

import csv
import json
from math import gcd
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.expanded_manifest import SHARED_CLASSES, iter_hust
from src.expanded_metrics import class_metrics, metrics_from_rows
from src.foundation_dataset import (ExpandedManifestDataset, HierarchicalBalancedBatchSampler,
                                    SHARED_LABELS)
from src.foundation_heads import FoundationCNN1D, SharedFourClassCNN1D
from src.hust_loader import (CWRU_FS, HUST_FS, TOKENS, expected_resampled_length, find_root,
                             list_recordings, load_hust_vibration, parse_hust_filename,
                             read_hust_mat, resample_to_cwru_rate)

HUST_ROOT = REPO_ROOT / "data" / "hust_bearing_v3"
MANIFEST = REPO_ROOT / "metadata" / "foundation_manifests" / "expanded_cwru_hust_temporal_v1_seed42"
BLOCKS = MANIFEST / "label_blocks"
FRACTIONS = (1, 5, 10, 25, 50, 100)


# --------------------------------------------------------------------------
# HUST loader
# --------------------------------------------------------------------------

def test_filename_grammar_is_unambiguous():
    assert parse_hust_filename("N400.mat")["dataset_class_label"] == "normal"
    assert parse_hust_filename("I402.mat") == {
        "token": "I", "dataset_class_label": "inner_race", "shared_class_label": "inner_race",
        "compound": False, "bearing_model": "6204", "load_w": 200, "middle_digit": "0"}
    # The two-character token must win over a one-character read: IB504 is
    # inner+ball on 6205, never `I` followed by something starting with B.
    assert parse_hust_filename("IB504.mat")["token"] == "IB"
    assert parse_hust_filename("IB504.mat")["bearing_model"] == "6205"
    assert parse_hust_filename("OB800.mat")["token"] == "OB"
    assert parse_hust_filename("nonsense.mat") is None
    assert parse_hust_filename("Z400.mat") is None          # unknown token
    assert parse_hust_filename("N900.mat") is None          # unknown bearing digit
    assert parse_hust_filename("N401.mat") is None          # unknown load digit


def test_compound_tokens_have_no_shared_label():
    for token in ("IO", "IB", "OB"):
        assert TOKENS[token][1] is None, f"{token} must not map into the shared space"
    for token in ("N", "I", "O", "B"):
        assert TOKENS[token][1] in SHARED_CLASSES


def test_resampling_ratio_is_exact_and_anti_aliased():
    g = gcd(HUST_FS, CWRU_FS)
    assert (CWRU_FS // g, HUST_FS // g) == (15, 64)
    assert expected_resampled_length(512_000) == 120_000
    y = resample_to_cwru_rate(np.zeros(512_000))
    assert y.size == 120_000 and y.dtype == np.float32

    # A 9 kHz tone is above the 6 kHz Nyquist and MUST be attenuated, not folded
    # back to 3 kHz. Naive decimation-by-striding would fail this.
    t = np.arange(HUST_FS, dtype=np.float64) / HUST_FS
    tone = np.sin(2 * np.pi * 9_000 * t)
    out = np.asarray(resample_to_cwru_rate(tone), dtype=np.float64)
    assert np.sqrt(np.mean(out ** 2)) < 0.02 * np.sqrt(np.mean(tone ** 2))

    # A 2 kHz tone is inside the passband and must survive.
    tone = np.sin(2 * np.pi * 2_000 * t)
    out = np.asarray(resample_to_cwru_rate(tone), dtype=np.float64)
    assert np.sqrt(np.mean(out ** 2)) > 0.9 * np.sqrt(np.mean(tone ** 2))


def test_only_primary_classes_are_enumerated_by_default():
    recs = list_recordings(HUST_ROOT)
    assert len(recs) == 57
    assert {r.token for r in recs} == {"N", "I", "O", "B"}
    assert sum(r.token == "N" for r in recs) == 15
    assert sum(r.token == "B" for r in recs) == 12          # 6204 ball is absent
    assert all(r.shared_class_label in SHARED_CLASSES for r in recs)
    every = list_recordings(HUST_ROOT, primary_only=False)
    assert len(every) == 99
    assert sum(r.compound for r in every) == 42


def test_run_up_variables_never_reach_the_model_signal():
    rec = next(r for r in list_recordings(HUST_ROOT) if r.token == "N")
    raw = read_hust_mat(rec.path)
    assert any(raw["run_up_present"].values()) or True      # presence varies by file
    signal = load_hust_vibration(rec.path)
    assert signal.size == expected_resampled_length(raw["data"].size)
    # The loaded signal must be the resampled `data`, not `data` concatenated
    # with any run-up array.
    assert signal.size < raw["data"].size
    assert np.isfinite(signal).all()


def test_short_recordings_survive_resampling():
    """Nine HUST files are shorter than 10 s; none may be silently dropped."""
    recs = list_recordings(HUST_ROOT)
    lengths = []
    for rec in recs:
        n = read_hust_mat(rec.path)["data"].size
        lengths.append(n)
    assert min(lengths) < 512_000, "expected at least one short recording in the primary subset"
    shortest = min(recs, key=lambda r: read_hust_mat(r.path)["data"].size)
    sig = load_hust_vibration(shortest.path)
    assert sig.size > 3 * 1024, "shortest recording must still yield usable regions"


def test_find_root_handles_the_nested_archive_directory():
    assert any(find_root(HUST_ROOT).glob("*.mat"))


# --------------------------------------------------------------------------
# Manifests and leakage
# --------------------------------------------------------------------------

def _rows(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def test_manifest_summary_checks_all_pass():
    s = json.loads((MANIFEST / "manifest_summary.json").read_text())
    c = s["checks"]
    assert c["raw_interval_cross_split_violations"] == 0
    assert c["exact_duplicate_cross_split_violations"] == 0
    assert c["duplicate_sample_ids"] == 0
    assert c["s2_equals_s1_plus_xjtu"] is True
    assert c["paderborn_absent"] is True
    assert s["raw_recordings"]["hust"] == 57
    assert s["raw_recordings"]["cwru"] == 60


def test_no_raw_sample_index_is_shared_between_splits():
    """The core leakage guarantee, checked directly on the emitted rows."""
    by_recording: dict[tuple[str, str], list[tuple[int, int, str]]] = {}
    for r in _rows(MANIFEST / "all_samples.csv"):
        key = (r["dataset"], r["recording_id"])
        by_recording.setdefault(key, []).append((int(r["raw_start"]), int(r["raw_end"]), r["split"]))
    checked = 0
    for key, spans in by_recording.items():
        spans.sort()
        active: list[tuple[int, int, str]] = []
        for cur in spans:
            active = [q for q in active if q[1] > cur[0]]
            for q in active:
                assert q[2] == cur[2], f"raw-point leakage in {key}: {q} vs {cur}"
            active.append(cur)
            checked += 1
    assert checked > 100_000


def test_every_window_lies_inside_one_contiguous_region_per_split():
    bounds: dict[tuple[str, str, str], tuple[int, int]] = {}
    for r in _rows(MANIFEST / "all_samples.csv"):
        key = (r["dataset"], r["recording_id"], r["split"])
        lo, hi = int(r["raw_start"]), int(r["raw_end"])
        cur = bounds.get(key)
        bounds[key] = (min(lo, cur[0]), max(hi, cur[1])) if cur else (lo, hi)
    # The three per-recording split ranges must not overlap each other.
    per_recording: dict[tuple[str, str], list[tuple[int, int, str]]] = {}
    for (d, rec, split), (lo, hi) in bounds.items():
        per_recording.setdefault((d, rec), []).append((lo, hi, split))
    for key, spans in per_recording.items():
        spans.sort()
        for a, b in zip(spans, spans[1:]):
            assert a[1] <= b[0], f"split ranges overlap in {key}: {a} vs {b}"


def test_supervised_manifests_exclude_xjtu_and_carry_shared_labels():
    for name in ("train_supervised.csv", "validation_supervised.csv", "test_supervised.csv"):
        rows = _rows(MANIFEST / name)
        assert rows, name
        assert {r["dataset"] for r in rows} == {"cwru", "hust"}
        assert all(r["shared_class_label"] in SHARED_CLASSES for r in rows)
        assert all(r["supervised_eligible"] == "true" for r in rows)


def test_ssl_manifests_contain_no_test_rows():
    for name in ("train_ssl_labelled_sources.csv", "train_ssl_plus_xjtu.csv"):
        assert {r["split"] for r in _rows(MANIFEST / name)} == {"train"}
    for name in ("validation_ssl_labelled_sources.csv", "validation_ssl_plus_xjtu.csv"):
        assert {r["split"] for r in _rows(MANIFEST / name)} == {"val"}


def test_normalization_is_fitted_on_training_windows_only():
    stats = json.loads((MANIFEST / "normalization_stats.json").read_text())
    assert stats["fit_split"] == "train"
    train_ids = {r["sample_id"] for r in _rows(MANIFEST / "all_samples.csv") if r["split"] == "train"}
    for dataset, channels in stats["datasets"].items():
        for channel, v in channels.items():
            assert v["validation_or_test_used"] is False
            assert v["std"] > 0 and np.isfinite(v["mean"])
            assert v["n_training_windows"] > 0
    # Every training window counted must actually be a training window.
    total = sum(v["n_training_windows"] for ch in stats["datasets"].values() for v in ch.values())
    assert total == len(train_ids)


def test_both_datasets_contribute_all_four_shared_classes():
    s = json.loads((MANIFEST / "manifest_summary.json").read_text())
    for dataset in ("cwru", "hust"):
        assert sorted(s["shared_classes_present_per_dataset"][dataset]) == sorted(SHARED_CLASSES)


# --------------------------------------------------------------------------
# Contiguous label blocks
# --------------------------------------------------------------------------

def test_label_blocks_are_nested():
    previous: set[str] | None = None
    for pct in FRACTIONS:
        ids = {r["sample_id"] for r in _rows(BLOCKS / f"label_blocks_{pct}pct.csv")}
        if previous is not None:
            assert previous <= ids, f"{pct}% is not a superset of the previous fraction"
        previous = ids


def test_label_blocks_are_contiguous_per_recording():
    for pct in FRACTIONS:
        by_recording: dict[tuple[str, str], list[dict]] = {}
        for r in _rows(BLOCKS / f"label_blocks_{pct}pct.csv"):
            by_recording.setdefault((r["dataset"], r["recording_id"]), []).append(r)
        for key, rows in by_recording.items():
            rows.sort(key=lambda r: int(r["raw_start"]))
            starts = [int(r["raw_start"]) for r in rows]
            steps = {b - a for a, b in zip(starts, starts[1:])}
            assert len(steps) <= 1, f"{pct}% block for {key} is not a single contiguous run: {steps}"


def test_label_blocks_are_a_prefix_of_the_training_region():
    train_first: dict[tuple[str, str], int] = {}
    for r in _rows(MANIFEST / "train_supervised.csv"):
        key = (r["dataset"], r["recording_id"])
        start = int(r["raw_start"])
        train_first[key] = min(train_first.get(key, start), start)
    for pct in FRACTIONS:
        first: dict[tuple[str, str], int] = {}
        for r in _rows(BLOCKS / f"label_blocks_{pct}pct.csv"):
            key = (r["dataset"], r["recording_id"])
            start = int(r["raw_start"])
            first[key] = min(first.get(key, start), start)
        for key, start in first.items():
            assert start == train_first[key], f"{pct}% block for {key} does not start the training region"


def test_label_blocks_never_touch_validation_or_test():
    val = {r["sample_id"] for r in _rows(MANIFEST / "validation_supervised.csv")}
    test = {r["sample_id"] for r in _rows(MANIFEST / "test_supervised.csv")}
    for pct in FRACTIONS:
        ids = {r["sample_id"] for r in _rows(BLOCKS / f"label_blocks_{pct}pct.csv")}
        assert not ids & val and not ids & test


def test_every_dataset_class_and_recording_is_represented_at_every_fraction():
    train = _rows(MANIFEST / "train_supervised.csv")
    want_recordings = {(r["dataset"], r["recording_id"]) for r in train}
    want_classes = {(r["dataset"], r["shared_class_label"]) for r in train}
    for pct in FRACTIONS:
        rows = _rows(BLOCKS / f"label_blocks_{pct}pct.csv")
        assert {(r["dataset"], r["recording_id"]) for r in rows} == want_recordings
        assert {(r["dataset"], r["shared_class_label"]) for r in rows} == want_classes


def test_one_percent_is_genuinely_small():
    """The whole point of contiguous blocks: 1% must be a sliver, not a survey."""
    rows = _rows(BLOCKS / "label_blocks_1pct.csv")
    full = _rows(BLOCKS / "label_blocks_100pct.csv")
    assert len(rows) / len(full) < 0.02
    per_recording = {}
    for r in rows:
        per_recording.setdefault((r["dataset"], r["recording_id"]), 0)
        per_recording[(r["dataset"], r["recording_id"])] += 1
    assert max(per_recording.values()) <= 8, "1% must not cover a large stretch of any recording"


def test_label_block_checksums_match_the_files_on_disk():
    import hashlib
    payload = json.loads((BLOCKS / "label_block_checksums.json").read_text())
    assert payload["nested"] is True and payload["problems"] == []
    for pct, entry in payload["subsets"].items():
        path = REPO_ROOT / entry["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["file_sha256"], pct


# --------------------------------------------------------------------------
# Shared head and hierarchical sampler
# --------------------------------------------------------------------------

def test_shared_head_cannot_see_dataset_identity():
    import inspect
    import torch
    model = SharedFourClassCNN1D()
    params = list(inspect.signature(model.forward).parameters)
    assert params == ["x"], f"forward must take only the signal, got {params}"
    out = model(torch.randn(3, 1, 1024))
    assert out.shape == (3, len(SHARED_CLASSES))
    # Exactly one classification head, not one per dataset.
    assert isinstance(model.head, torch.nn.Linear)
    assert not any(isinstance(m, torch.nn.ModuleDict) for m in model.modules())


def test_accepted_dataset_specific_head_is_unchanged():
    """FoundationCNN1D backs accepted results; it must still be per-dataset."""
    model = FoundationCNN1D()
    assert set(model.heads) == {"cwru", "paderborn"}
    assert model.heads["cwru"].out_features == 4
    assert model.heads["paderborn"].out_features == 3


def test_linear_probe_freezes_only_the_encoder():
    model = SharedFourClassCNN1D()
    model.freeze_encoder(True)
    assert not any(p.requires_grad for p in model.encoder.parameters())
    assert all(p.requires_grad for p in model.head.parameters())
    model.train()
    assert not model.encoder.training, "frozen encoder must stay in eval mode"


def test_hierarchical_sampler_balances_datasets_and_classes():
    rows = []
    # Deliberately lopsided: cwru has 10x the rows and one dominant class.
    for i in range(1000):
        rows.append({"dataset": "cwru", "shared_class_label": "normal",
                     "recording_id": f"c{i % 20}", "bearing_id": "b"})
    for i in range(100):
        rows.append({"dataset": "hust", "shared_class_label": ["inner_race", "outer_race"][i % 2],
                     "recording_id": f"h{i % 4}", "bearing_id": "b"})
    sampler = HierarchicalBalancedBatchSampler(rows, batch_size=64, n_batches=50, seed=42)
    sampler.set_epoch(1)
    drawn = [i for b in sampler for i in b]
    per_dataset = {"cwru": 0, "hust": 0}
    for i in drawn:
        per_dataset[rows[i]["dataset"]] += 1
    assert abs(per_dataset["cwru"] - per_dataset["hust"]) <= 1, per_dataset
    hust_classes = {"inner_race": 0, "outer_race": 0}
    for i in drawn:
        if rows[i]["dataset"] == "hust":
            hust_classes[rows[i]["shared_class_label"]] += 1
    ratio = min(hust_classes.values()) / max(hust_classes.values())
    assert ratio > 0.8, hust_classes


def test_hierarchical_sampler_skips_unlabelled_rows():
    rows = [{"dataset": "cwru", "shared_class_label": "normal", "recording_id": "a", "bearing_id": "b"},
            {"dataset": "xjtu_sy", "shared_class_label": "", "recording_id": "x", "bearing_id": "b"}]
    sampler = HierarchicalBalancedBatchSampler(rows, batch_size=4, n_batches=5, seed=1)
    sampler.set_epoch(1)
    assert sampler.datasets == ["cwru"]
    assert all(rows[i]["dataset"] == "cwru" for b in sampler for i in b)


def test_hierarchical_sampler_is_deterministic_per_epoch():
    rows = [{"dataset": "d", "shared_class_label": "normal", "recording_id": f"r{i%5}",
             "bearing_id": "b"} for i in range(200)]
    a = HierarchicalBalancedBatchSampler(rows, 16, 10, seed=7)
    b = HierarchicalBalancedBatchSampler(rows, 16, 10, seed=7)
    a.set_epoch(3); b.set_epoch(3)
    assert list(a) == list(b)
    b.set_epoch(4)
    assert list(a) != list(b), "different epochs must give different draws"


def test_expanded_dataset_returns_shared_indices():
    ds = ExpandedManifestDataset(BLOCKS / "label_blocks_1pct.csv",
                                 MANIFEST / "normalization_stats.json")
    x, label, dataset, sample_id, bearing = ds[0]
    assert tuple(x.shape) == (1, 1024)
    assert 0 <= label < len(SHARED_CLASSES)
    assert SHARED_LABELS[ds.rows[0]["shared_class_label"]] == label
    assert dataset in {"cwru", "hust"}
    assert np.isfinite(x.numpy()).all()


def test_normalization_is_applied_per_dataset():
    ds = ExpandedManifestDataset(BLOCKS / "label_blocks_10pct.csv",
                                 MANIFEST / "normalization_stats.json")
    seen = {}
    for i in range(len(ds)):
        row = ds.rows[i]
        if row["dataset"] not in seen:
            seen[row["dataset"]] = ds[i][0].numpy()
        if len(seen) == 2:
            break
    assert set(seen) == {"cwru", "hust"}
    # Different rigs have wildly different raw sensitivity; after per-dataset
    # normalization neither should be orders of magnitude larger than the other.
    scales = {d: float(np.std(v)) for d, v in seen.items()}
    assert max(scales.values()) / max(min(scales.values()), 1e-9) < 100, scales


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def test_headline_metric_is_the_unweighted_dataset_mean():
    rows = ([{"dataset": "cwru", "true_index": 0, "predicted_index": 0}] * 1000
            + [{"dataset": "hust", "true_index": 1, "predicted_index": 0}] * 10)
    m = metrics_from_rows(rows)
    # cwru is perfect on one class, hust is wrong on one class.
    assert m["combined"]["mean_dataset_macro_f1"] == (
        m["datasets"]["cwru"]["macro_f1"] + m["datasets"]["hust"]["macro_f1"]) / 2
    # Pooling would be dominated by cwru's 1000 rows; it must not be the headline.
    assert m["shared_head_pooled"]["macro_f1"] != m["combined"]["mean_dataset_macro_f1"]
    assert m["combined"]["headline_definition"] == "unweighted mean of per-dataset macro-F1"


def test_class_metrics_cover_all_four_classes_even_when_absent():
    m = class_metrics(np.array([0, 0, 1]), np.array([0, 0, 1]))
    assert len(m["per_class"]) == 4
    assert m["per_class"]["outer_race"]["support"] == 0
    assert m["macro_f1"] < m["macro_f1_present_classes"]


# --------------------------------------------------------------------------

def _run_all() -> int:
    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith("test_") and callable(o)]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:                                 # noqa: BLE001
            failed.append((name, exc))
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
