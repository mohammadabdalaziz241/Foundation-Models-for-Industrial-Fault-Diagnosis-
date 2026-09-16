"""Protocol tests for `bearing_generalisation_v1`.

Three groups:

1. **Unit** — the protocol primitives: DC removal, the role plan, fold structure,
   label-block construction.
2. **Manifest** — properties of the manifests actually written to disk.
3. **Negative controls** — deliberately corrupted copies of the manifest, one per
   leakage assertion. An assertion that cannot be made to fail is not an assertion,
   so each of these injects a specific violation and requires the audit to catch it.

The negative controls run against a *miniature* copy of the real manifest: every
recording, role, bearing and fold is preserved, but only the first window row of
each recording is kept. That keeps the structure complete while making the fixture
about ninety times smaller and fast enough to rebuild per test.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.bearing_generalisation import (
    BEARING_REGIMES, CLASS_TO_INDEX, DC_TOLERANCE, FIELDS,
    LEGACY_COMPROMISED_BEARINGS, MODEL_FS, PROTOCOL, RECORDING_REGIMES,
    RECORDING_ROLE_PLAN, ROLE_ADAPTATION, ROLE_SETTING_A, ROLE_SETTING_B,
    ROLE_SOURCE_TRAIN, ROLE_SOURCE_VAL, ROLE_SSL, ROLE_VALIDATION, SHARED_CLASSES,
    SPLIT_SEED, WINDOW_LEN, assert_dc_removed, assert_no_legacy_bearings,
    build_folds, cwru_role, dc_remove, exact_window_hash, make_sample_id,
    paderborn_class, recording_role, segment, select_bearing_block,
    select_recording_block,
)
from src.cv_paderborn import DEVELOPMENT_POOL, TEST_BEARINGS

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from assert_bearing_generalisation_manifests import audit  # noqa: E402

MANIFEST_ROOT = REPO_ROOT / "metadata" / "foundation_manifests" / f"{PROTOCOL}_seed{SPLIT_SEED}"
manifest_required = pytest.mark.skipif(
    not MANIFEST_ROOT.exists(),
    reason="run scripts/build_bearing_generalisation_manifests.py first",
)


# ===========================================================================
# 1. Unit — DC removal (Stage 2 Decision D)
# ===========================================================================

def test_dc_removal_makes_every_window_zero_mean():
    rng = np.random.default_rng(0)
    # deliberately large, bearing-like offsets on top of the signal
    x = rng.standard_normal((64, WINDOW_LEN)).astype(np.float32) * 0.5
    x += rng.choice([-0.017, 0.041], size=(64, 1)).astype(np.float32)
    out = dc_remove(x)
    worst = float(np.abs(np.asarray(out, np.float64).mean(axis=-1)).max())
    assert worst <= DC_TOLERANCE, f"max |window mean| {worst:.3e} exceeds {DC_TOLERANCE:.1e}"


def test_dc_removal_preserves_amplitude_and_is_not_z_scoring():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((32, WINDOW_LEN)).astype(np.float32) * rng.uniform(
        0.3, 3.0, size=(32, 1)).astype(np.float32)
    out = dc_remove(x)
    # standard deviation is untouched: this removes the offset, not the scale
    assert np.allclose(x.std(axis=-1), out.std(axis=-1), atol=1e-4)
    # and the per-window std is NOT driven to 1, which is what z-scoring would do
    assert out.std(axis=-1).std() > 0.1


def test_dc_removal_handles_one_dimensional_input():
    x = np.linspace(-1.0, 1.0, WINDOW_LEN, dtype=np.float32) + 7.0
    out = dc_remove(x)
    assert out.shape == x.shape
    assert abs(float(np.asarray(out, np.float64).mean())) <= DC_TOLERANCE


def test_assert_dc_removed_rejects_an_offset_window():
    x = np.ones((4, WINDOW_LEN), dtype=np.float32) * 0.04
    with pytest.raises(AssertionError, match="DC removal failed"):
        assert_dc_removed(x, "unit test")


def test_assert_dc_removed_accepts_and_reports_the_worst_case():
    out = dc_remove(np.random.default_rng(2).standard_normal((8, WINDOW_LEN)) + 3.0)
    assert assert_dc_removed(out, "unit test") <= DC_TOLERANCE


def test_exact_hash_is_computed_on_the_dc_removed_window():
    rng = np.random.default_rng(3)
    base = rng.standard_normal(WINDOW_LEN).astype(np.float32)
    offset = base + np.float32(0.041)
    # hashing happens AFTER demeaning, so the offset cannot survive into the digest
    assert exact_window_hash(offset) != exact_window_hash(dc_remove(offset))
    # the digest is reproducible from the same canonical single application, which is
    # what the manifest records and what the A8 audit re-derives from raw
    assert exact_window_hash(dc_remove(offset)) == exact_window_hash(dc_remove(offset))
    # demeaning is numerically -- but NOT bit -- idempotent in float32: the residual
    # mean after one pass is ~1e-8, and subtracting it again perturbs low-order bits.
    # The protocol therefore hashes exactly one application; re-applying stays well
    # inside tolerance but is not required to reproduce the digest.
    twice = dc_remove(dc_remove(offset))
    assert np.allclose(twice, dc_remove(offset), atol=1e-6)
    assert abs(float(np.asarray(twice, np.float64).mean())) <= DC_TOLERANCE
    # the same waveform under two different offsets lands on the same signal
    assert np.allclose(dc_remove(offset), dc_remove(base - np.float32(0.017)), atol=1e-6)


# ===========================================================================
# 1. Unit — roles, folds, blocks
# ===========================================================================

def test_recording_role_plan_partitions_all_twenty_repeats():
    seen: set[int] = set()
    for repeats in RECORDING_ROLE_PLAN.values():
        assert not (seen & set(repeats)), "roles overlap"
        seen |= set(repeats)
    assert seen == set(range(1, 21))


def test_legacy_bearing_set_matches_the_original_sealed_set():
    assert LEGACY_COMPROMISED_BEARINGS == frozenset(TEST_BEARINGS)
    assert LEGACY_COMPROMISED_BEARINGS == frozenset(
        {"K005", "K006", "KA08", "KA09", "KA30", "KI08", "KI18", "KI21"})
    assert not (LEGACY_COMPROMISED_BEARINGS & set(DEVELOPMENT_POOL))


def test_recording_role_refuses_a_legacy_bearing():
    with pytest.raises(AssertionError, match="legacy compromised"):
        recording_role("K005", 1, [])


def test_assert_no_legacy_bearings_detects_each_of_the_eight():
    for b in sorted(LEGACY_COMPROMISED_BEARINGS):
        with pytest.raises(AssertionError, match=b):
            assert_no_legacy_bearings(["K001", b], "unit test")
    assert_no_legacy_bearings(sorted(DEVELOPMENT_POOL), "unit test")


def test_folds_are_bearing_disjoint_and_cover_the_pool_exactly_once():
    folds = build_folds()
    assert len(folds) == 4
    counted: dict[str, int] = {}
    for f in folds:
        assert not (set(f["setting_b_bearings"]) & set(f["population_1_bearings"]))
        assert set(f["setting_b_bearings"]) | set(f["population_1_bearings"]) == set(DEVELOPMENT_POOL)
        for b in f["setting_b_bearings"]:
            counted[b] = counted.get(b, 0) + 1
    assert set(counted) == set(DEVELOPMENT_POOL)
    assert set(counted.values()) == {1}, "each bearing must be Setting B exactly once"


def test_every_fold_has_all_three_classes_and_both_damage_origins_in_setting_b():
    for f in build_folds():
        classes = {paderborn_class(b) for b in f["setting_b_bearings"]}
        assert classes == set(SHARED_CLASSES), f["fold_id"]
        from src.bearing_generalisation import damage_origin
        origins = {damage_origin(b) for b in f["setting_b_bearings"]}
        assert {"artificial", "real", "none"} <= origins, f["fold_id"]


def test_cwru_role_split_is_exhaustive_over_the_four_loads():
    assert {cwru_role(l) for l in (0, 1, 2)} == {ROLE_SOURCE_TRAIN}
    assert cwru_role(3) == ROLE_SOURCE_VAL
    with pytest.raises(AssertionError):
        cwru_role(4)


def _synthetic_adaptation_pool(fold) -> dict[str, list[tuple[str, int]]]:
    pool: dict[str, list[tuple[str, int]]] = {c: [] for c in SHARED_CLASSES}
    for b in fold["population_1_bearings"]:
        for r in RECORDING_ROLE_PLAN[ROLE_ADAPTATION]:
            pool[paderborn_class(b)].append((b, r))
    return pool


def test_recording_label_blocks_are_nested_and_correctly_sized():
    for fold in build_folds():
        pool = _synthetic_adaptation_pool(fold)
        previous: set = set()
        for name, n in RECORDING_REGIMES.items():
            block = set(select_recording_block(pool, n))
            assert previous <= block, f"{fold['fold_id']}/{name} is not nested"
            previous = block
            for c in SHARED_CLASSES:
                got = sum(1 for b, _ in block if paderborn_class(b) == c)
                assert got == (len(pool[c]) if n is None else n), (fold["fold_id"], name, c)


def test_recording_blocks_maximise_bearing_spread_at_small_n():
    for fold in build_folds():
        pool = _synthetic_adaptation_pool(fold)
        block = select_recording_block(pool, 5)
        for c in SHARED_CLASSES:
            bearings = {b for b, _ in block if paderborn_class(b) == c}
            available = {b for b, _ in pool[c]}
            assert len(bearings) == min(5, len(available)), (fold["fold_id"], c)


def test_bearing_label_blocks_are_nested_and_take_whole_bearings():
    for fold in build_folds():
        pool = _synthetic_adaptation_pool(fold)
        previous: set = set()
        for name, n in BEARING_REGIMES.items():
            block = set(select_bearing_block(pool, n))
            assert previous <= block, f"{fold['fold_id']}/{name} is not nested"
            previous = block
            for c in SHARED_CLASSES:
                bearings = {b for b, _ in block if paderborn_class(b) == c}
                assert len(bearings) == n
                for b in bearings:
                    got = {r for bb, r in block if bb == b}
                    assert got == set(RECORDING_ROLE_PLAN[ROLE_ADAPTATION])


def test_label_blocks_are_deterministic_across_calls():
    fold = build_folds()[0]
    pool = _synthetic_adaptation_pool(fold)
    assert select_recording_block(pool, 5) == select_recording_block(pool, 5)
    assert select_bearing_block(pool, 2) == select_bearing_block(pool, 2)


def test_a_regime_larger_than_the_pool_is_rejected_rather_than_silently_truncated():
    fold = build_folds()[0]
    pool = _synthetic_adaptation_pool(fold)
    with pytest.raises(AssertionError, match="only"):
        select_recording_block(pool, 10_000)
    with pytest.raises(AssertionError, match="only"):
        select_bearing_block(pool, 99)


def test_segment_never_crosses_a_recording_boundary():
    sig = np.arange(48_000, dtype=np.float32)
    spans = [(s, e) for s, e, _ in segment(sig)]
    assert all(e <= len(sig) for _, e in spans)
    assert all(e - s == WINDOW_LEN for s, e in spans)
    assert spans[1][0] - spans[0][0] == WINDOW_LEN // 2


def test_sample_id_is_fold_independent():
    a = make_sample_id("paderborn", "N15_M07_F10_K001_1", 0, 1024)
    assert "fold" not in a and a.endswith("00000000-00001024")


# ===========================================================================
# 2. Manifest properties
# ===========================================================================

def _read(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _fold_dirs(root: Path) -> list[Path]:
    """Fold directories only — `fold_assignments.json` must not be swept up."""
    return sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("fold"))


@manifest_required
def test_manifest_summary_reports_dc_removal_passed():
    s = json.loads((MANIFEST_ROOT / "manifest_summary.json").read_text())
    assert s["dc_removal"]["applied"] is True
    assert s["dc_removal"]["passed"] is True
    assert s["dc_removal"]["max_abs_window_mean"] <= DC_TOLERANCE
    assert s["checks"]["legacy_bearings_read"] == 0
    assert s["checks"]["p1_equals_p4_cwru_source_rows"] is True


@manifest_required
def test_manifest_contains_no_legacy_bearing_anywhere():
    for path in MANIFEST_ROOT.rglob("*.csv"):
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            if "bearing_id" not in (reader.fieldnames or []):
                continue
            for row in reader:
                assert row["bearing_id"] not in LEGACY_COMPROMISED_BEARINGS, path


@manifest_required
def test_every_fold_uses_all_twenty_recordings_of_every_bearing_exactly_once():
    for fdir in sorted(_fold_dirs(MANIFEST_ROOT)):
        roles = _read(fdir / "recording_roles.csv")
        assert len(roles) == 21 * 20
        seen: dict[str, set] = {}
        for r in roles:
            seen.setdefault(r["bearing_id"], set()).add(int(r["repeat_index"]))
        assert len(seen) == 21
        for bearing, repeats in seen.items():
            assert repeats == set(range(1, 21)), bearing


@manifest_required
def test_setting_b_bearings_hold_every_recording_and_nothing_else():
    assignments = {f["fold_id"]: set(f["setting_b_bearings"])
                   for f in json.loads((MANIFEST_ROOT / "fold_assignments.json").read_text())["folds"]}
    for fdir in sorted(_fold_dirs(MANIFEST_ROOT)):
        rows = _read(fdir / "recording_roles.csv")
        sb = {r["bearing_id"] for r in rows if r["role"] == ROLE_SETTING_B}
        assert sb == assignments[fdir.name]
        for b in sb:
            reps = {int(r["repeat_index"]) for r in rows
                    if r["bearing_id"] == b and r["role"] == ROLE_SETTING_B}
            assert reps == set(range(1, 21)), b


@manifest_required
def test_population_1_recording_roles_follow_the_declared_plan():
    for fdir in sorted(_fold_dirs(MANIFEST_ROOT)):
        for r in _read(fdir / "recording_roles.csv"):
            if r["population"] != "population_1":
                continue
            assert int(r["repeat_index"]) in RECORDING_ROLE_PLAN[r["role"]], r


@manifest_required
def test_normalisation_is_fitted_on_ssl_rows_only():
    for fdir in sorted(_fold_dirs(MANIFEST_ROOT)):
        norm = json.loads((fdir / "normalization.json").read_text())
        ssl_ids = [row["sample_id"] for row in _read(fdir / "ssl.csv")]
        digest = hashlib.sha256("\n".join(sorted(ssl_ids)).encode()).hexdigest()
        assert norm["fitted_on_role"] == ROLE_SSL
        assert norm["training_sample_ids_sha256"] == digest
        assert norm["validation_rows_used"] is False
        assert norm["setting_a_eval_rows_used"] is False
        assert norm["setting_b_rows_used"] is False
        # the pooled mean is recorded but must be ~0, proving DC removal upstream
        assert abs(norm["pooled_mean_not_applied"]) <= DC_TOLERANCE
        assert norm["std"] > 0


@manifest_required
def test_cwru_source_excludes_ball_and_is_shared_by_p1_and_p4():
    summary = json.loads((MANIFEST_ROOT / "cwru_source" / "cwru_source_summary.json").read_text())
    assert summary["excluded_classes"] == ["ball"]
    assert summary["n_recordings"] == 44
    assert set(summary["recordings_per_class"]) == set(SHARED_CLASSES)
    assert summary["p1_equals_p4_source_rows"] is True
    for role in (ROLE_SOURCE_TRAIN, ROLE_SOURCE_VAL):
        for row in _read(MANIFEST_ROOT / "cwru_source" / f"{role}.csv"):
            assert row["class_label"] in SHARED_CLASSES
            assert row["class_index"] == str(CLASS_TO_INDEX[row["class_label"]])
            assert row["source_sample_rate"] == str(MODEL_FS), "CWRU must never be resampled"


@manifest_required
def test_label_blocks_are_shared_by_every_arm():
    for fdir in sorted(_fold_dirs(MANIFEST_ROOT)):
        index = json.loads((fdir / "label_blocks" / "label_block_index.json").read_text())
        assert set(index["blocks"]) == set(RECORDING_REGIMES) | set(BEARING_REGIMES)
        assert "every arm" in index["shared_by"]
        for name, entry in index["blocks"].items():
            rows = _read(fdir / "label_blocks" / f"{name}.csv")
            got = hashlib.sha256(
                "\n".join(sorted(r["sample_id"] for r in rows)).encode()).hexdigest()
            assert got == entry["sample_ids_sha256"], (fdir.name, name)


@manifest_required
def test_r1_block_is_exactly_one_recording_per_class():
    for fdir in sorted(_fold_dirs(MANIFEST_ROOT)):
        entry = json.loads(
            (fdir / "label_blocks" / "label_block_index.json").read_text())["blocks"]["R1"]
        assert entry["recordings_per_class"] == {c: 1 for c in sorted(SHARED_CLASSES)}
        assert entry["n_recordings"] == 3


@manifest_required
def test_manifest_rows_carry_the_full_declared_schema():
    for path in (MANIFEST_ROOT / "fold0" / "ssl.csv",
                 MANIFEST_ROOT / "cwru_source" / "source_train.csv"):
        with path.open(newline="") as f:
            assert csv.DictReader(f).fieldnames == FIELDS, path


# ===========================================================================
# 3. Negative controls — every assertion must be capable of failing
# ===========================================================================

def _miniaturise(src: Path, dst: Path) -> None:
    """Copy the manifest, keeping only the first window row of each recording."""
    for path in sorted(src.rglob("*")):
        rel = path.relative_to(src)
        target = dst / rel
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".csv" and path.name != "recording_roles.csv" \
                and path.name != "damage_origin_balance.csv":
            rows = _read(path)
            keep, seen = [], set()
            for r in rows:
                if r["recording_id"] not in seen:
                    seen.add(r["recording_id"])
                    keep.append(r)
            with target.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                w.writeheader()
                w.writerows(keep)
        else:
            shutil.copy2(path, target)
    _refresh(dst)


def _refresh(root: Path) -> None:
    """Recompute the digests that the audit cross-checks, so only the injected
    violation is left for it to find."""
    for fdir in sorted(_fold_dirs(root)):
        ssl_ids = [r["sample_id"] for r in _read(fdir / "ssl.csv")]
        norm = json.loads((fdir / "normalization.json").read_text())
        norm["training_sample_ids_sha256"] = hashlib.sha256(
            "\n".join(sorted(ssl_ids)).encode()).hexdigest()
        norm["n_training_windows"] = len(ssl_ids)
        (fdir / "normalization.json").write_text(json.dumps(norm, indent=2, sort_keys=True) + "\n")
        index = json.loads((fdir / "label_blocks" / "label_block_index.json").read_text())
        for name, entry in index["blocks"].items():
            rows = _read(fdir / "label_blocks" / f"{name}.csv")
            entry["sample_ids_sha256"] = hashlib.sha256(
                "\n".join(sorted(r["sample_id"] for r in rows)).encode()).hexdigest()
            entry["n_windows"] = len(rows)
        (fdir / "label_blocks" / "label_block_index.json").write_text(
            json.dumps(index, indent=2, sort_keys=True) + "\n")
    ids = {r["sample_id"] for role in (ROLE_SOURCE_TRAIN, ROLE_SOURCE_VAL)
           for r in _read(root / "cwru_source" / f"{role}.csv")}
    summary = json.loads((root / "cwru_source" / "cwru_source_summary.json").read_text())
    digest = hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()
    summary["p1_ssl_sample_ids_sha256"] = digest
    summary["p4_supervised_sample_ids_sha256"] = digest
    summary["p1_equals_p4_source_rows"] = True
    (root / "cwru_source" / "cwru_source_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    checks = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name != "manifest_checksums.json":
            h = hashlib.sha256()
            h.update(p.read_bytes())
            checks[str(p.relative_to(root))] = h.hexdigest()
    (root / "manifest_checksums.json").write_text(json.dumps(checks, indent=2, sort_keys=True) + "\n")


@pytest.fixture(scope="module")
def mini(tmp_path_factory):
    if not MANIFEST_ROOT.exists():
        pytest.skip("manifests not built")
    root = tmp_path_factory.mktemp("mini") / "manifest"
    _miniaturise(MANIFEST_ROOT, root)
    return root


def _fresh(mini: Path, tmp_path: Path) -> Path:
    out = tmp_path / "case"
    shutil.copytree(mini, out)
    return out


def _run(root: Path) -> dict:
    return audit(root, full_dc_check=False, dc_sample=6, seed=SPLIT_SEED)


def _codes_failing(report: dict) -> set[str]:
    return {c["code"] for c in report["checks"] if not c["passed"]}


def _append(path: Path, rows: list[dict]) -> None:
    existing = _read(path)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(existing + rows)


@manifest_required
def test_baseline_miniature_manifest_passes(mini):
    report = _run(mini)
    assert report["passed"], _codes_failing(report)


@manifest_required
def test_A1_detects_a_recording_in_two_roles(mini, tmp_path):
    root = _fresh(mini, tmp_path)
    row = _read(root / "fold0" / "ssl.csv")[0]
    _append(root / "fold0" / "validation.csv", [dict(row, role=ROLE_VALIDATION)])
    _refresh(root)
    assert "A1" in _codes_failing(_run(root))


@manifest_required
def test_A2_detects_a_setting_a_recording_used_for_adaptation(mini, tmp_path):
    root = _fresh(mini, tmp_path)
    row = _read(root / "fold0" / "setting_a_eval.csv")[0]
    _append(root / "fold0" / "adaptation_pool.csv", [dict(row, role=ROLE_ADAPTATION)])
    _refresh(root)
    assert "A2" in _codes_failing(_run(root))


@manifest_required
def test_A2_detects_a_setting_a_recording_inside_a_label_block(mini, tmp_path):
    root = _fresh(mini, tmp_path)
    row = _read(root / "fold0" / "setting_a_eval.csv")[0]
    _append(root / "fold0" / "label_blocks" / "R1.csv", [dict(row, role=ROLE_ADAPTATION)])
    _refresh(root)
    assert "A2" in _codes_failing(_run(root))


@manifest_required
def test_A3_detects_a_setting_b_bearing_entering_ssl(mini, tmp_path):
    root = _fresh(mini, tmp_path)
    row = _read(root / "fold0" / "setting_b_eval.csv")[0]
    _append(root / "fold0" / "ssl.csv", [dict(row, role=ROLE_SSL)])
    _refresh(root)
    assert "A3" in _codes_failing(_run(root))


@manifest_required
def test_A4_detects_a_legacy_compromised_bearing(mini, tmp_path):
    root = _fresh(mini, tmp_path)
    row = _read(root / "fold0" / "ssl.csv")[0]
    _append(root / "fold0" / "ssl.csv",
            [dict(row, bearing_id="KA30", sample_id=row["sample_id"] + ":injected")])
    _refresh(root)
    assert "A4" in _codes_failing(_run(root))


@manifest_required
def test_A5_detects_p1_and_p4_source_rows_diverging(mini, tmp_path):
    root = _fresh(mini, tmp_path)
    path = root / "cwru_source" / "cwru_source_summary.json"
    summary = json.loads(path.read_text())
    summary["p4_supervised_sample_ids_sha256"] = "0" * 64
    summary["p1_equals_p4_source_rows"] = False
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    # refresh checksums only, so the injected divergence survives
    _checksum_only(root)
    assert "A5" in _codes_failing(_run(root))


@manifest_required
def test_A6_detects_a_label_block_edited_after_its_digest_was_recorded(mini, tmp_path):
    root = _fresh(mini, tmp_path)
    rows = _read(root / "fold0" / "label_blocks" / "R5.csv")
    with (root / "fold0" / "label_blocks" / "R5.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows[:-1])
    _checksum_only(root)
    assert "A6" in _codes_failing(_run(root))


@manifest_required
def test_A7_detects_a_scaler_fitted_on_evaluation_rows(mini, tmp_path):
    root = _fresh(mini, tmp_path)
    path = root / "fold0" / "normalization.json"
    norm = json.loads(path.read_text())
    norm["fitted_on_role"] = "setting_a_eval"
    path.write_text(json.dumps(norm, indent=2, sort_keys=True) + "\n")
    _checksum_only(root)
    assert "A7" in _codes_failing(_run(root))


@manifest_required
def test_A8_detects_a_window_that_was_not_dc_removed(mini, tmp_path):
    root = _fresh(mini, tmp_path)
    rows = _read(root / "fold0" / "ssl.csv")
    rows[0]["exact_hash"] = "f" * 64          # hash of a window nobody produced
    with (root / "fold0" / "ssl.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    _refresh(root)
    report = audit(root, full_dc_check=True, dc_sample=0, seed=SPLIT_SEED)
    assert "A8" in _codes_failing(report)


@manifest_required
def test_A9_detects_overlapping_raw_intervals_across_roles(mini, tmp_path):
    root = _fresh(mini, tmp_path)
    row = _read(root / "fold0" / "ssl.csv")[0]
    clash = dict(row, role=ROLE_VALIDATION, raw_start=str(int(row["raw_start"]) + 256),
                 raw_end=str(int(row["raw_end"]) + 256),
                 sample_id=row["sample_id"] + ":overlap")
    _append(root / "fold0" / "validation.csv", [clash])
    _refresh(root)
    assert "A9" in _codes_failing(_run(root))


@manifest_required
def test_A10_detects_a_label_block_escaping_the_adaptation_pool(mini, tmp_path):
    root = _fresh(mini, tmp_path)
    row = _read(root / "fold0" / "validation.csv")[0]
    _append(root / "fold0" / "label_blocks" / "Rfull.csv", [dict(row, role=ROLE_ADAPTATION)])
    _refresh(root)
    assert "A10" in _codes_failing(_run(root))


@manifest_required
def test_A11_detects_a_file_edited_after_checksumming(mini, tmp_path):
    root = _fresh(mini, tmp_path)
    (root / "fold0" / "fold_summary.json").write_text('{"tampered": true}\n')
    assert "A11" in _codes_failing(_run(root))


def _checksum_only(root: Path) -> None:
    checks = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name != "manifest_checksums.json":
            checks[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
    (root / "manifest_checksums.json").write_text(json.dumps(checks, indent=2, sort_keys=True) + "\n")
