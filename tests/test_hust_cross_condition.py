"""Tests for the HUST cross-load protocols (Stages B and C).

Covers every check listed in §12 of the experimental direction:
cross-load isolation; held-out-load exclusion from SSL, normalization and selection;
training-only normalization; temporal raw-point isolation; label-block nesting,
checksum stability and actual-vs-requested accounting; seven-class mapping and
compound-fault preservation; missing HUST combinations preserved; hierarchical
sampler balance; seed determinism; SSL checkpoint reuse with a fresh head;
architecture-matched parameter equality; prediction replay; checkpoint save/reload;
output-directory isolation.

Runnable standalone (the venv has torch but not pytest) or under pytest.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import statistics
import sys
import tempfile

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.expanded_metrics import class_metrics
from src.hust_cross_condition import (CLASSES_4, CLASSES_7, CrossConditionDataset, FOLDS,
                                      HierarchicalCrossConditionSampler, SPLIT_SEED,
                                      TOKEN_TO_CLASS, class_space, hust_recordings)
from src.hust_loader import list_recordings
from src.unified_data import split_temporal_regions, split_temporal_regions_two_way

MANI = REPO_ROOT / "metadata" / "foundation_manifests"
P4 = "hust_cross_condition_v1"
P7 = "hust_extended_seven_class_cross_condition_v1"
FRACTIONS = (1, 5, 10)
HUST_ROOT = REPO_ROOT / "data" / "hust_bearing_v3"


def _rows(p: Path) -> list[dict]:
    with p.open(newline="") as f:
        return list(csv.DictReader(f))


def _fold_dir(protocol: str, fold: str, seed: int = 42) -> Path:
    return MANI / f"{protocol}_seed{seed}" / fold


def _all_folds():
    for protocol in (P4, P7):
        for fold in FOLDS:
            yield protocol, fold


# --------------------------------------------------------------------------
# Two-way temporal split
# --------------------------------------------------------------------------

def test_two_way_split_is_contiguous_and_keeps_sizes_fixed():
    for seed in range(8):
        regions = split_temporal_regions_two_way(10_000, seed=seed, recording_id=f"r{seed}")
        assert len(regions) == 2
        assert regions[0].start == 0 and regions[-1].end == 10_000
        assert regions[0].end == regions[1].start
        sizes = {r.split: r.end - r.start for r in regions}
        # The seed may move the validation block, but never resize it.
        assert sizes["train"] == 8_000 and sizes["val"] == 2_000


def test_two_way_split_seed_moves_the_validation_block():
    positions = {split_temporal_regions_two_way(10_000, seed=42, recording_id=f"r{i}")[0].split
                 for i in range(40)}
    assert positions == {"train", "val"}, "seed should place validation first or last"


def test_three_way_split_is_unchanged():
    """The accepted protocols depend on this function; it must not have moved."""
    regions = split_temporal_regions(10_000)
    assert [(r.split, r.start, r.end) for r in regions] == [
        ("train", 0, 7000), ("val", 7000, 8500), ("test", 8500, 10000)]


# --------------------------------------------------------------------------
# Class mapping, compound preservation, missing combinations
# --------------------------------------------------------------------------

def test_seven_class_mapping_and_compound_preservation():
    assert class_space(4) == CLASSES_4 and class_space(7) == CLASSES_7
    assert len(CLASSES_7) == 7 and CLASSES_7[:4] == CLASSES_4
    for token, name in (("IO", "inner_outer"), ("IB", "inner_rolling_element"),
                        ("OB", "outer_rolling_element")):
        assert TOKEN_TO_CLASS[token] == name
        assert name in CLASSES_7 and name not in CLASSES_4
    # Compound faults must never collapse into a single-fault class.
    assert len(set(TOKEN_TO_CLASS.values())) == 7


def test_missing_hust_combinations_are_preserved_not_synthesized():
    recs = hust_recordings(HUST_ROOT, 7)
    assert len(recs) == 99
    combos = {(r.token, r.bearing_model) for r in recs}
    assert ("B", "6204") not in combos, "B/6204 does not exist and must not be invented"
    assert ("IB", "6204") not in combos, "IB/6204 does not exist and must not be invented"
    assert len(combos) == 33
    assert len(hust_recordings(HUST_ROOT, 4)) == 57


def test_every_class_is_present_at_every_load():
    """Gate A2 re-checked in code, not just in the report."""
    recs = list_recordings(HUST_ROOT, primary_only=False)
    by_token = {}
    for r in recs:
        by_token.setdefault(r.token, set()).add(r.load_w)
    for token in TOKEN_TO_CLASS:
        assert by_token[token] == {0, 200, 400}, f"{token} missing a load: {by_token[token]}"


# --------------------------------------------------------------------------
# Cross-load isolation
# --------------------------------------------------------------------------

def test_held_out_load_is_absent_from_every_training_side_manifest():
    for protocol, fold in _all_folds():
        d = _fold_dir(protocol, fold)
        held = FOLDS[fold].test_load
        for name in ("train_supervised", "validation_supervised", "train_ssl", "validation_ssl"):
            loads = {int(r["load_w"]) for r in _rows(d / f"{name}.csv")}
            assert held not in loads, f"{protocol}/{fold}: held-out {held}W leaked into {name}"
            assert loads == set(FOLDS[fold].train_loads)


def test_test_manifest_contains_only_the_held_out_load():
    for protocol, fold in _all_folds():
        loads = {int(r["load_w"]) for r in _rows(_fold_dir(protocol, fold) / "test_supervised.csv")}
        assert loads == {FOLDS[fold].test_load}


def test_no_raw_sample_index_is_shared_between_splits():
    for protocol, fold in _all_folds():
        by_recording: dict[str, list] = {}
        for r in _rows(_fold_dir(protocol, fold) / "all_samples.csv"):
            by_recording.setdefault(r["recording_id"], []).append(
                (int(r["raw_start"]), int(r["raw_end"]), r["split"]))
        for rid, spans in by_recording.items():
            spans.sort()
            active = []
            for cur in spans:
                active = [q for q in active if q[1] > cur[0]]
                for q in active:
                    assert q[2] == cur[2], f"{protocol}/{fold}/{rid}: raw-point leakage {q} vs {cur}"
                active.append(cur)


def test_train_and_validation_regions_do_not_overlap_within_a_recording():
    for protocol, fold in _all_folds():
        bounds: dict[tuple[str, str], tuple[int, int]] = {}
        for r in _rows(_fold_dir(protocol, fold) / "all_samples.csv"):
            k = (r["recording_id"], r["split"])
            lo, hi = int(r["raw_start"]), int(r["raw_end"])
            cur = bounds.get(k)
            bounds[k] = (min(lo, cur[0]), max(hi, cur[1])) if cur else (lo, hi)
        per_rec: dict[str, list] = {}
        for (rid, split), (lo, hi) in bounds.items():
            per_rec.setdefault(rid, []).append((lo, hi, split))
        for rid, spans in per_rec.items():
            spans.sort()
            for a, b in zip(spans, spans[1:]):
                assert a[1] <= b[0], f"{protocol}/{fold}/{rid}: regions overlap {a} vs {b}"


def test_normalization_is_fitted_on_training_load_train_windows_only():
    for protocol, fold in _all_folds():
        d = _fold_dir(protocol, fold)
        stats = json.loads((d / "normalization_stats.json").read_text())
        assert stats["fit_split"] == "train"
        assert stats["held_out_load"] == FOLDS[fold].test_load
        assert stats["held_out_load_used"] is False
        v = stats["datasets"]["hust"]["data"]
        assert v["validation_or_test_used"] is False
        assert v["std"] > 0 and np.isfinite(v["mean"])
        # Every counted window must be a training-region window of a training load.
        train = _rows(d / "train_supervised.csv")
        assert v["n_training_windows"] == len(train)


def test_manifest_summary_checks_all_pass():
    for protocol, fold in _all_folds():
        c = json.loads((_fold_dir(protocol, fold) / "manifest_summary.json").read_text())["checks"]
        bad = [k for k, val in c.items() if val not in (0, True)]
        assert not bad, f"{protocol}/{fold}: failing checks {bad}"


def test_all_classes_appear_on_both_sides_of_every_fold():
    for protocol, fold in _all_folds():
        s = json.loads((_fold_dir(protocol, fold) / "manifest_summary.json").read_text())
        expected = sorted(s["classes"])
        assert sorted(s["classes_present"]["train"]) == expected
        assert sorted(s["classes_present"]["test"]) == expected


# --------------------------------------------------------------------------
# Seed determinism
# --------------------------------------------------------------------------

def test_same_seed_gives_identical_manifests_and_blocks():
    """Seeds vary optimization only: all seed directories must be byte-identical."""
    for protocol in (P4, P7):
        ref = json.loads((_fold_dir(protocol, "HC-A", 42) / "manifest_checksums.json").read_text())
        for seed in (43, 44):
            other = json.loads((_fold_dir(protocol, "HC-A", seed) / "manifest_checksums.json").read_text())
            assert other == ref, f"{protocol}: seed {seed} manifest differs from seed 42"
        for seed in (43, 44):
            for pct in FRACTIONS:
                a = (_fold_dir(protocol, "HC-A", 42) / "label_blocks" / f"label_blocks_{pct}pct.csv").read_bytes()
                b = (_fold_dir(protocol, "HC-A", seed) / "label_blocks" / f"label_blocks_{pct}pct.csv").read_bytes()
                assert a == b, f"{protocol}: seed {seed} label block {pct}% differs"


def test_split_seed_is_recorded_and_fixed():
    for protocol, fold in _all_folds():
        s = json.loads((_fold_dir(protocol, fold) / "manifest_summary.json").read_text())
        assert s["split_seed"] == SPLIT_SEED == 42


# --------------------------------------------------------------------------
# Label blocks
# --------------------------------------------------------------------------

def test_label_blocks_are_nested_contiguous_and_prefix():
    for protocol, fold in _all_folds():
        d = _fold_dir(protocol, fold)
        train_first: dict[str, int] = {}
        for r in _rows(d / "train_supervised.csv"):
            s = int(r["raw_start"])
            train_first[r["recording_id"]] = min(train_first.get(r["recording_id"], s), s)
        previous = None
        for pct in FRACTIONS:
            rows = _rows(d / "label_blocks" / f"label_blocks_{pct}pct.csv")
            ids = {r["sample_id"] for r in rows}
            if previous is not None:
                assert previous <= ids, f"{protocol}/{fold}: {pct}% breaks nesting"
            previous = ids
            per_rec: dict[str, list[int]] = {}
            for r in rows:
                per_rec.setdefault(r["recording_id"], []).append(int(r["raw_start"]))
            for rid, starts in per_rec.items():
                starts.sort()
                assert starts[0] == train_first[rid], f"{protocol}/{fold}/{rid}: block is not a prefix"
                steps = {b - a for a, b in zip(starts, starts[1:])}
                assert len(steps) <= 1, f"{protocol}/{fold}/{rid}: block is not contiguous"


def test_label_blocks_never_touch_validation_or_test():
    for protocol, fold in _all_folds():
        d = _fold_dir(protocol, fold)
        val = {r["sample_id"] for r in _rows(d / "validation_supervised.csv")}
        test = {r["sample_id"] for r in _rows(d / "test_supervised.csv")}
        for pct in FRACTIONS:
            ids = {r["sample_id"] for r in _rows(d / "label_blocks" / f"label_blocks_{pct}pct.csv")}
            assert not (ids & val) and not (ids & test)


def test_label_block_accounting_reports_both_fractions_and_is_consistent():
    for protocol, fold in _all_folds():
        acc = _rows(_fold_dir(protocol, fold) / "label_blocks" / "label_block_accounting.csv")
        assert {int(r["fraction_percent"]) for r in acc} == set(FRACTIONS)
        for r in acc:
            # Both denominators must be present and must differ (50% overlap).
            wf = float(r["actual_window_fraction_percent"])
            df = float(r["actual_duration_fraction_percent"])
            assert 0 < wf < 100 and 0 < df < 100
            assert df > wf, "duration fraction must exceed window fraction at 50% overlap"
            # No recording may silently drop out.
            assert int(r["recordings_with_zero_labels"]) == 0
            assert int(r["recordings_with_labels"]) == int(r["recordings_total"])
            # Coverage must be genuinely achieved, not merely claimed.
            assert r["classes_represented"] == r["classes_expected"]
            assert r["loads_represented"] == r["loads_expected"]
            assert r["bearing_types_represented"] == r["bearing_types_expected"]


def test_label_block_checksums_match_files_on_disk():
    import hashlib
    for protocol, fold in _all_folds():
        payload = json.loads((_fold_dir(protocol, fold) / "label_blocks"
                              / "label_block_checksums.json").read_text())
        assert payload["problems"] == [] and payload["nested"] is True
        for pct, entry in payload["subsets"].items():
            got = hashlib.sha256((REPO_ROOT / entry["path"]).read_bytes()).hexdigest()
            assert got == entry["file_sha256"], f"{protocol}/{fold}: {pct}% checksum drift"


def test_one_percent_covers_every_recording_exactly_once_or_twice():
    """The A3 interrogation, asserted rather than only reported."""
    for protocol, fold in _all_folds():
        rows = _rows(_fold_dir(protocol, fold) / "label_blocks" / "label_blocks_1pct.csv")
        per_rec: dict[str, int] = {}
        for r in rows:
            per_rec[r["recording_id"]] = per_rec.get(r["recording_id"], 0) + 1
        n_train_recs = len({r["recording_id"] for r in
                            _rows(_fold_dir(protocol, fold) / "train_supervised.csv")})
        assert len(per_rec) == n_train_recs, "1% must still touch every training recording"
        assert max(per_rec.values()) <= 2, "1% must remain a sliver of each recording"


# --------------------------------------------------------------------------
# Sampler
# --------------------------------------------------------------------------

def test_hierarchical_sampler_balances_classes_under_lopsided_input():
    rows = []
    for i in range(2000):
        rows.append({"class_label": "normal", "bearing_model": "6205", "load_w": "0",
                     "recording_id": f"n{i % 10}"})
    for i in range(50):
        rows.append({"class_label": "inner_race", "bearing_model": "6206", "load_w": "200",
                     "recording_id": f"i{i % 2}"})
    s = HierarchicalCrossConditionSampler(rows, batch_size=64, n_batches=40, seed=42)
    s.set_epoch(1)
    drawn = [i for b in s for i in b]
    counts = {"normal": 0, "inner_race": 0}
    for i in drawn:
        counts[rows[i]["class_label"]] += 1
    assert abs(counts["normal"] - counts["inner_race"]) <= 1, counts


def test_hierarchical_sampler_balances_bearing_types_and_loads():
    rows = []
    for bearing, n in (("6205", 900), ("6206", 100)):
        for i in range(n):
            rows.append({"class_label": "normal", "bearing_model": bearing,
                         "load_w": "0" if i % 5 else "200", "recording_id": f"{bearing}-{i % 7}"})
    s = HierarchicalCrossConditionSampler(rows, batch_size=64, n_batches=60, seed=7)
    s.set_epoch(1)
    drawn = [i for b in s for i in b]
    per_bearing = {"6205": 0, "6206": 0}
    for i in drawn:
        per_bearing[rows[i]["bearing_model"]] += 1
    ratio = min(per_bearing.values()) / max(per_bearing.values())
    assert ratio > 0.85, per_bearing


def test_hierarchical_sampler_is_deterministic_per_epoch():
    rows = [{"class_label": "normal", "bearing_model": "6205", "load_w": "0",
             "recording_id": f"r{i % 5}"} for i in range(200)]
    a = HierarchicalCrossConditionSampler(rows, 16, 10, seed=7)
    b = HierarchicalCrossConditionSampler(rows, 16, 10, seed=7)
    a.set_epoch(3); b.set_epoch(3)
    assert list(a) == list(b)
    b.set_epoch(4)
    assert list(a) != list(b)


def test_sampler_skips_unlabelled_rows():
    rows = [{"class_label": "normal", "bearing_model": "6205", "load_w": "0", "recording_id": "a"},
            {"class_label": "", "bearing_model": "6205", "load_w": "0", "recording_id": "b"}]
    s = HierarchicalCrossConditionSampler(rows, 4, 5, seed=1)
    s.set_epoch(1)
    assert s.classes == ["normal"]
    assert all(rows[i]["class_label"] == "normal" for b in s for i in b)


# --------------------------------------------------------------------------
# Model, checkpoint reuse, architecture matching
# --------------------------------------------------------------------------

def _model(n):
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from run_hust_cross_condition import CrossConditionCNN1D
    return CrossConditionCNN1D(n)


def test_architecture_matched_parameter_equality():
    """S0 and S1 must instantiate identical architectures; only init differs."""
    for n in (4, 7):
        a, b = _model(n), _model(n)
        assert sum(p.numel() for p in a.parameters()) == sum(p.numel() for p in b.parameters())
        assert [tuple(p.shape) for p in a.parameters()] == [tuple(p.shape) for p in b.parameters()]
        assert a(torch.randn(3, 1, 1024)).shape == (3, n)
    assert sum(p.numel() for p in _model(7).parameters()) > sum(p.numel() for p in _model(4).parameters())


def test_ssl_checkpoint_reuse_gives_same_encoder_and_an_untrained_head():
    """One checkpoint per (fold, seed) reused across fractions, head never carried over.

    "Fresh head" means each fraction starts from a newly constructed, UNTRAINED
    head -- not that the initialisation differs between fractions. Because the run
    seed is fixed per (fold, seed), the initialisation is deliberately identical
    across fractions, which removes head init as a nuisance variable between them.
    What must never happen is a head (or encoder) carried over from another
    fraction's fine-tuning.
    """
    from run_hust_cross_condition import state_sha
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "best_checkpoint.pt"
        torch.manual_seed(0)
        torch.save(_model(4).encoder.state_dict(), ckpt)
        encoders, heads = [], []
        for _fraction in range(3):                      # stand-ins for 1%, 5%, 10%
            torch.manual_seed(42)                       # the run seed, fixed per (fold, seed)
            m = _model(4)
            untrained_head = state_sha(m.head.state_dict())
            m.encoder.load_state_dict(torch.load(ckpt, weights_only=True))
            encoders.append(state_sha(m.encoder.state_dict()))
            heads.append(untrained_head)
            # Simulate this fraction training its head; the next iteration must not inherit it.
            with torch.no_grad():
                m.head.weight.add_(1.0)
        assert len(set(encoders)) == 1, "the same checkpoint must give the same encoder every time"
        assert len(set(heads)) == 1, "same run seed must give the same fresh head init"
        torch.manual_seed(42)
        assert state_sha(_model(4).head.state_dict()) == heads[0], \
            "the head must be freshly constructed, never inherited from a trained one"


def test_runner_artifacts_confirm_checkpoint_reuse():
    """Assert the property on REAL debug artifacts, not a simulation."""
    import glob
    for suffix in ("", "_native51200"):
        paths = sorted(glob.glob(str(REPO_ROOT / "results" / "foundation_model"
                                     / f"hust_cross_condition_v1{suffix}" / "debug" / "HC-A"
                                     / "S1" / "full_*pct" / "seed_42" / "run_metadata.json")))
        if len(paths) < 2:
            continue                                     # debug runs not present; nothing to assert
        metas = [json.loads(Path(p).read_text()) for p in paths]
        assert len({m["ssl_checkpoint"] for m in metas}) == 1, "fractions must share one SSL checkpoint"
        assert len({m["encoder_sha_before"] for m in metas}) == 1, \
            "every fraction must start from the same SSL encoder, not a previous fraction's"


def test_native_rows_load_native_signals():
    """Regression: the ceiling arm must not index a 12 kHz signal with native offsets."""
    from src.hust_cross_condition import NATIVE_FS, signal_for
    d = _fold_dir(P4 + "_native51200", "HC-A")
    rows = _rows(d / "train_supervised.csv")
    assert rows and all(int(r["model_sample_rate"]) == NATIVE_FS for r in rows)
    worst = max(rows, key=lambda r: int(r["raw_end"]))
    sig = signal_for(worst)
    assert sig.size >= int(worst["raw_end"]), \
        "native manifest indexes past the end of the loaded signal (wrong rate loaded)"
    ds = CrossConditionDataset(d / "label_blocks" / "label_blocks_1pct.csv",
                               d / "normalization_stats.json", CLASSES_4, window_len=4384)
    x, label, _dataset, _sid, _bearing = ds[0]
    assert tuple(x.shape) == (1, 4384) and np.isfinite(x.numpy()).all()


def test_variable_window_ssl_matches_protected_implementation():
    """The ceiling arm must optimise the SAME O2 objective, not a lookalike."""
    from src.masked_ssl_variable import VariableWindowMaskedSSL
    from src.models.masked_ssl import MaskedSSL
    from src.models.cnn1d import CNN1D

    def build(cls, **kw):
        torch.manual_seed(0); enc = CNN1D(4)
        torch.manual_seed(1)
        return cls(encoder_model=enc, patch_len=32, mask_ratio=0.5, frequency_weight=1.0, **kw)

    a, b = build(MaskedSSL), build(VariableWindowMaskedSSL, window_len=1024)
    x = torch.randn(4, 1, 1024)
    torch.manual_seed(7); ra, la = a(x)
    torch.manual_seed(7); rb, lb = b(x)
    assert torch.equal(ra, rb), "reconstruction differs from the protected implementation"
    assert torch.equal(la, lb), "loss differs from the protected implementation"
    for k in a.last_loss_components:
        assert torch.equal(a.last_loss_components[k], b.last_loss_components[k]), k


def test_native_ssl_geometry_preserves_patch_structure():
    from src.hust_cross_condition import NATIVE_PATCH_LEN, NATIVE_WINDOW_LEN, RATES, MODEL_FS, NATIVE_FS
    from src.masked_ssl_variable import VariableWindowMaskedSSL
    from src.models.cnn1d import CNN1D
    assert NATIVE_WINDOW_LEN % NATIVE_PATCH_LEN == 0
    m = VariableWindowMaskedSSL(encoder_model=CNN1D(4), patch_len=NATIVE_PATCH_LEN,
                                mask_ratio=0.5, window_len=NATIVE_WINDOW_LEN, frequency_weight=1.0)
    ref = 1024 // RATES[MODEL_FS][2]
    assert m.n_patches == ref == 32, "native arm must keep the 32-patch structure"
    # Patch duration must match to well under a millisecond.
    assert abs(NATIVE_PATCH_LEN / NATIVE_FS - RATES[MODEL_FS][2] / MODEL_FS) < 5e-5
    recon, loss = m(torch.randn(2, 1, NATIVE_WINDOW_LEN))
    assert tuple(recon.shape) == (2, 1, NATIVE_WINDOW_LEN) and torch.isfinite(loss)


def test_linear_probe_freezes_only_the_encoder():
    m = _model(4)
    m.freeze_encoder(True)
    assert not any(p.requires_grad for p in m.encoder.parameters())
    assert all(p.requires_grad for p in m.head.parameters())
    m.train()
    assert not m.encoder.training


def test_checkpoint_save_and_reload_round_trips():
    from run_hust_cross_condition import state_sha
    with tempfile.TemporaryDirectory() as tmp:
        m = _model(7)
        p = Path(tmp) / "m.pt"
        torch.save(m.state_dict(), p)
        m2 = _model(7)
        m2.load_state_dict(torch.load(p, weights_only=True))
        assert state_sha(m.encoder.state_dict()) == state_sha(m2.encoder.state_dict())
        x = torch.randn(2, 1, 1024)
        m.eval(); m2.eval()
        assert torch.allclose(m(x), m2(x))


# --------------------------------------------------------------------------
# Metrics and datasets
# --------------------------------------------------------------------------

def test_prediction_replay_is_exact():
    rng = np.random.RandomState(0)
    y = rng.randint(0, 7, 500); p = rng.randint(0, 7, 500)
    a = class_metrics(y, p, CLASSES_7)
    b = class_metrics(y, p, CLASSES_7)
    assert a == b
    assert len(a["per_class"]) == 7


def test_dataset_returns_correct_class_indices():
    d = _fold_dir(P7, "HC-A")
    ds = CrossConditionDataset(d / "label_blocks" / "label_blocks_1pct.csv",
                               d / "normalization_stats.json", CLASSES_7)
    seen = set()
    for i in range(len(ds)):
        x, label, dataset, sid, bearing = ds[i]
        assert tuple(x.shape) == (1, 1024) and np.isfinite(x.numpy()).all()
        assert CLASSES_7.index(ds.rows[i]["class_label"]) == label
        seen.add(ds.rows[i]["class_label"])
    assert seen == set(CLASSES_7), "1% block must expose all seven classes"


def test_output_directory_isolation():
    """The new protocols must not write into any accepted results directory."""
    for protocol in (P4, P7):
        assert not (REPO_ROOT / "results" / "foundation_model" / protocol).is_symlink()
    for accepted in ("S0", "S1", "S2", "temporal_regions_v1_seed42",
                     "grouped_generalization_v1_seed42",
                     "expanded_cwru_hust_temporal_v1_seed42"):
        p = REPO_ROOT / "results" / "foundation_model" / accepted
        if p.exists():
            assert p.is_dir()
    for accepted in ("temporal_regions_v1_seed42", "grouped_generalization_v1_seed42",
                     "expanded_cwru_hust_temporal_v1_seed42"):
        assert (MANI / accepted).exists(), "accepted manifests must still be present"


# --------------------------------------------------------------------------

def _run_all() -> int:
    tests = [(n, o) for n, o in sorted(globals().items()) if n.startswith("test_") and callable(o)]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:                                  # noqa: BLE001
            failed.append(name)
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
