"""Cache tests for `bearing_generalisation_v1` (Stage 3.1).

Every safeguard the loader claims is exercised by a **negative control**: a cache that
has been deliberately broken in one specific way, which the loader must refuse. A guard
that has never been made to fire is not a guard.

The fixture is a genuine cache built over a handful of real manifest rows, so the
hashes, the raw files and the resampling path are all real — just small enough to
rebuild per test.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import stat
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.bearing_generalisation import (
    DC_TOLERANCE, FIELDS, PROTOCOL, SPLIT_SEED, WINDOW_LEN, dc_remove, exact_window_hash,
)
from src.bearing_generalisation_cache import (
    CACHE_VERSION, PREPROCESSING_POLICY, REQUIRED_METADATA_FIELDS,
    CacheValidationError, WindowCache, build_cache, cache_identity,
    collect_declared_rows, default_cache_dir, loader_source_digest, manifest_digest,
    preprocessing_digest,
)

MANIFEST_ROOT = REPO_ROOT / "metadata" / "foundation_manifests" / f"{PROTOCOL}_seed{SPLIT_SEED}"
manifest_required = pytest.mark.skipif(
    not MANIFEST_ROOT.exists(), reason="build the manifests first")

N_FIXTURE_ROWS = 24


# ===========================================================================
# fixtures
# ===========================================================================

def _checksums(root: Path) -> None:
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name != "manifest_checksums.json":
            out[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
    (root / "manifest_checksums.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")


def _mini_manifest(dst: Path, n_rows: int = N_FIXTURE_ROWS) -> Path:
    """A minimal but real manifest: a few rows per dataset plus a checksum table."""
    dst.mkdir(parents=True, exist_ok=True)
    picked: list[dict] = []
    for rel in ("fold0/ssl.csv", "cwru_source/source_train.csv"):
        with (MANIFEST_ROOT / rel).open(newline="") as f:
            rows = list(csv.DictReader(f))
        step = max(1, len(rows) // (n_rows // 2))
        picked.extend(rows[::step][: n_rows // 2])
    with (dst / "rows.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(picked)
    _checksums(dst)
    return dst


@pytest.fixture(scope="module")
def mini_manifest(tmp_path_factory):
    if not MANIFEST_ROOT.exists():
        pytest.skip("manifests not built")
    return _mini_manifest(tmp_path_factory.mktemp("mini_manifest") / "m")


@pytest.fixture(scope="module")
def reference_cache(mini_manifest, tmp_path_factory):
    out = tmp_path_factory.mktemp("refcache") / "cache"
    build_cache(mini_manifest, out, creation_command="pytest",
                git_commit="0" * 40, created_at="1970-01-01T00:00:00+00:00")
    return out


def _writable_copy(src: Path, dst: Path) -> Path:
    shutil.copytree(src, dst)
    for p in dst.rglob("*"):
        p.chmod(p.stat().st_mode | stat.S_IWUSR)
    dst.chmod(dst.stat().st_mode | stat.S_IWUSR)
    return dst


def _rewrite_meta(cache: Path, **updates) -> None:
    meta = json.loads((cache / "cache_metadata.json").read_text())
    meta.update(updates)
    (cache / "cache_metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")


# ===========================================================================
# identity
# ===========================================================================

@manifest_required
def test_cache_identity_is_content_addressed(mini_manifest):
    ident = cache_identity(mini_manifest)
    assert len(ident) == 64
    assert cache_identity(mini_manifest) == ident, "identity must be deterministic"
    assert default_cache_dir(mini_manifest).name.endswith(ident[:16])


@manifest_required
def test_identity_changes_when_the_manifest_set_changes(mini_manifest, tmp_path):
    other = _writable_copy(mini_manifest, tmp_path / "m2")
    rows = list(csv.DictReader((other / "rows.csv").open(newline="")))
    with (other / "rows.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows[:-1])
    _checksums(other)
    assert cache_identity(other) != cache_identity(mini_manifest)


def test_identity_changes_when_the_preprocessing_policy_changes(monkeypatch):
    before = preprocessing_digest()
    monkeypatch.setitem(PREPROCESSING_POLICY, "model_sample_rate_hz", 24_000)
    assert preprocessing_digest() != before


def test_identity_changes_when_window_shaping_code_changes(monkeypatch):
    before = loader_source_digest()
    import src.bearing_generalisation_cache as mod

    def fake_dc_remove(x):  # different source text
        x = np.asarray(x, dtype=np.float32)
        return (x - x.mean(axis=-1, keepdims=True)).astype(np.float32)

    monkeypatch.setattr(mod, "_WINDOW_SHAPING_FUNCTIONS",
                        (fake_dc_remove,) + mod._WINDOW_SHAPING_FUNCTIONS[1:])
    assert loader_source_digest() != before


# ===========================================================================
# construction requirements
# ===========================================================================

@manifest_required
def test_cache_contains_dc_removed_unscaled_windows(reference_cache, mini_manifest):
    cache = WindowCache(reference_cache, mini_manifest)
    declared = collect_declared_rows(mini_manifest)
    for sid in cache.sample_ids:
        w = cache.get(sid)
        assert w.dtype == np.float32
        assert w.shape == (WINDOW_LEN,)
        assert abs(float(np.asarray(w, np.float64).mean())) <= DC_TOLERANCE
        assert exact_window_hash(w) == declared[sid].exact_hash
    # amplitude was preserved: windows are not unit variance
    stds = np.array([cache.get(s).std() for s in cache.sample_ids])
    assert stds.std() > 1e-3, "per-window scale was destroyed; z-scoring is forbidden"


@manifest_required
def test_cache_holds_no_fold_scaling_labels_or_training_statistics(reference_cache):
    meta = json.loads((reference_cache / "cache_metadata.json").read_text())
    policy = meta["preprocessing_policy"]
    assert policy["fold_scaling_in_cache"] is False
    assert policy["labels_in_sample_array"] is False
    assert policy["augmentation"] is False
    assert set(p.name for p in reference_cache.iterdir()) == {
        "windows.npy", "sample_ids.json", "index.json", "cache_metadata.json"}


@manifest_required
def test_cache_metadata_is_complete(reference_cache):
    meta = json.loads((reference_cache / "cache_metadata.json").read_text())
    for field in REQUIRED_METADATA_FIELDS:
        assert field in meta, field
    assert meta["cache_version"] == CACHE_VERSION
    assert meta["dtype"] == "float32"
    assert meta["entry_count"] == meta["array_shape"][0]
    assert meta["validation"]["passed"] is True


@manifest_required
def test_published_cache_is_read_only(reference_cache):
    for p in reference_cache.iterdir():
        assert not (p.stat().st_mode & stat.S_IWUSR), f"{p.name} is writable"


@manifest_required
def test_refuses_to_overwrite_an_existing_cache(reference_cache, mini_manifest):
    with pytest.raises(FileExistsError, match="immutable"):
        build_cache(mini_manifest, reference_cache)


@manifest_required
def test_build_refuses_a_manifest_whose_hash_disagrees(mini_manifest, tmp_path):
    broken = _writable_copy(mini_manifest, tmp_path / "broken_manifest")
    rows = list(csv.DictReader((broken / "rows.csv").open(newline="")))
    rows[0]["exact_hash"] = "a" * 64
    with (broken / "rows.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    _checksums(broken)
    with pytest.raises(AssertionError, match="does not match the manifest"):
        build_cache(broken, tmp_path / "cache_from_broken")


@manifest_required
def test_collect_declared_rows_rejects_disagreeing_manifests(mini_manifest, tmp_path):
    two = _writable_copy(mini_manifest, tmp_path / "two")
    rows = list(csv.DictReader((two / "rows.csv").open(newline="")))
    clash = dict(rows[0], raw_start=str(int(rows[0]["raw_start"]) + 512))
    with (two / "other.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows([clash])
    with pytest.raises(AssertionError, match="manifest disagreement"):
        collect_declared_rows(two)


# ===========================================================================
# negative controls — one per declared loader safeguard
# ===========================================================================

@manifest_required
def test_baseline_cache_loads(reference_cache, mini_manifest):
    cache = WindowCache(reference_cache, mini_manifest, verify_hashes=True)
    assert len(cache) == N_FIXTURE_ROWS


@manifest_required
def test_rejects_a_corrupted_sample(reference_cache, mini_manifest, tmp_path):
    bad = _writable_copy(reference_cache, tmp_path / "c")
    arr = np.load(bad / "windows.npy", mmap_mode="r+")
    # Corrupt in a MEAN-PRESERVING way, so the DC guard cannot see it and the hash
    # guard is the only thing standing between this cache and a training run.
    arr[0, 0] = np.float32(arr[0, 0] + 1.0)
    arr[0, 1] = np.float32(arr[0, 1] - 1.0)
    arr.flush()
    del arr
    check = WindowCache(bad, mini_manifest)
    assert abs(float(np.asarray(check.get(check.sample_ids[0]), np.float64).mean())) \
        <= DC_TOLERANCE, "the corruption should be invisible to the DC guard"
    with pytest.raises(CacheValidationError, match="fail their manifest hash"):
        WindowCache(bad, mini_manifest, verify_hashes=True)


@manifest_required
def test_rejects_a_missing_entry(reference_cache, mini_manifest, tmp_path):
    cache = _writable_copy(reference_cache, tmp_path / "c")
    ids = json.loads((cache / "sample_ids.json").read_text())
    index = json.loads((cache / "index.json").read_text())
    dropped = ids.pop()
    index.pop(dropped)
    (cache / "sample_ids.json").write_text(json.dumps(ids, indent=0) + "\n")
    (cache / "index.json").write_text(json.dumps(index, sort_keys=True) + "\n")
    arr = np.load(cache / "windows.npy")
    np.save(cache / "windows.npy", arr[:-1])
    _rewrite_meta(cache, entry_count=len(ids), array_shape=[len(ids), WINDOW_LEN])
    with pytest.raises(CacheValidationError, match="no cache entry"):
        WindowCache(cache, mini_manifest)


@manifest_required
def test_rejects_an_extra_undeclared_entry(reference_cache, mini_manifest, tmp_path):
    cache = _writable_copy(reference_cache, tmp_path / "c")
    ids = json.loads((cache / "sample_ids.json").read_text())
    index = json.loads((cache / "index.json").read_text())
    ids.append("paderborn:INJECTED_RECORDING:00000000-00001024")
    index[ids[-1]] = len(ids) - 1
    (cache / "sample_ids.json").write_text(json.dumps(ids, indent=0) + "\n")
    (cache / "index.json").write_text(json.dumps(index, sort_keys=True) + "\n")
    arr = np.load(cache / "windows.npy")
    np.save(cache / "windows.npy", np.vstack([arr, np.zeros((1, WINDOW_LEN), np.float32)]))
    _rewrite_meta(cache, entry_count=len(ids), array_shape=[len(ids), WINDOW_LEN])
    with pytest.raises(CacheValidationError, match="not declared by any manifest row"):
        WindowCache(cache, mini_manifest)


@manifest_required
def test_rejects_a_stale_preprocessing_digest(reference_cache, mini_manifest, tmp_path):
    cache = _writable_copy(reference_cache, tmp_path / "c")
    _rewrite_meta(cache, preprocessing_digest="b" * 64)
    with pytest.raises(CacheValidationError, match="preprocessing digest differs"):
        WindowCache(cache, mini_manifest)


@manifest_required
def test_rejects_a_stale_manifest_digest(reference_cache, mini_manifest, tmp_path):
    cache = _writable_copy(reference_cache, tmp_path / "c")
    _rewrite_meta(cache, manifest_digest="c" * 64)
    with pytest.raises(CacheValidationError, match="manifest digest differs"):
        WindowCache(cache, mini_manifest)


@manifest_required
def test_rejects_a_dc_removal_violation(reference_cache, mini_manifest, tmp_path):
    cache = _writable_copy(reference_cache, tmp_path / "c")
    arr = np.load(cache / "windows.npy", mmap_mode="r+")
    arr[3] = arr[3] + np.float32(0.041)      # exactly the offset Stage 1 measured
    arr.flush()
    del arr
    loaded = WindowCache(cache, mini_manifest)          # structural checks still pass
    with pytest.raises(CacheValidationError, match="violate the DC tolerance"):
        loaded.verify_all_hashes(check_hashes=False)     # isolate the DC guard


@manifest_required
def test_rejects_an_incorrect_dtype(reference_cache, mini_manifest, tmp_path):
    cache = _writable_copy(reference_cache, tmp_path / "c")
    arr = np.load(cache / "windows.npy")
    np.save(cache / "windows.npy", arr.astype(np.float64))
    with pytest.raises(CacheValidationError, match="dtype"):
        WindowCache(cache, mini_manifest)


@manifest_required
def test_rejects_an_incorrect_shape(reference_cache, mini_manifest, tmp_path):
    cache = _writable_copy(reference_cache, tmp_path / "c")
    arr = np.load(cache / "windows.npy")
    np.save(cache / "windows.npy", arr[:, :512])
    with pytest.raises(CacheValidationError, match="shape"):
        WindowCache(cache, mini_manifest)


@manifest_required
def test_rejects_a_wrong_cache_version(reference_cache, mini_manifest, tmp_path):
    cache = _writable_copy(reference_cache, tmp_path / "c")
    _rewrite_meta(cache, cache_version="0.0.1-not-this-one")
    with pytest.raises(CacheValidationError, match="cache version"):
        WindowCache(cache, mini_manifest)


@manifest_required
def test_rejects_incomplete_metadata(reference_cache, mini_manifest, tmp_path):
    cache = _writable_copy(reference_cache, tmp_path / "c")
    meta = json.loads((cache / "cache_metadata.json").read_text())
    del meta["loader_source_digest"]
    (cache / "cache_metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    with pytest.raises(CacheValidationError, match="incomplete"):
        WindowCache(cache, mini_manifest)


@manifest_required
def test_rejects_missing_metadata_file(reference_cache, mini_manifest, tmp_path):
    cache = _writable_copy(reference_cache, tmp_path / "c")
    (cache / "cache_metadata.json").unlink()
    with pytest.raises(CacheValidationError, match="cache_metadata.json is missing"):
        WindowCache(cache, mini_manifest)


@manifest_required
def test_rejects_a_missing_cache_directory(mini_manifest, tmp_path):
    with pytest.raises(CacheValidationError, match="does not exist"):
        WindowCache(tmp_path / "nope", mini_manifest)


@manifest_required
def test_unknown_sample_id_is_refused_not_silently_zero(reference_cache, mini_manifest):
    cache = WindowCache(reference_cache, mini_manifest)
    with pytest.raises(CacheValidationError, match="not in the cache"):
        cache.get("paderborn:NOT_A_RECORDING:00000000-00001024")


@manifest_required
def test_independent_raw_rederivation_matches(reference_cache, mini_manifest):
    cache = WindowCache(reference_cache, mini_manifest)
    result = cache.rederive_from_raw(cache.sample_ids)
    assert result["mismatches"] == 0
    assert result["entries_rederived"] == len(cache)


@manifest_required
def test_second_build_is_byte_identical(mini_manifest, tmp_path):
    a, _ = build_cache(mini_manifest, tmp_path / "a", creation_command="x",
                       git_commit="0" * 40, created_at="1970-01-01T00:00:00+00:00")
    b, _ = build_cache(mini_manifest, tmp_path / "b", creation_command="y",
                       git_commit="0" * 40, created_at="2000-01-01T00:00:00+00:00")
    for name in ("windows.npy", "sample_ids.json", "index.json"):
        assert (a / name).read_bytes() == (b / name).read_bytes(), name


# ===========================================================================
# the real published cache, if it exists
# ===========================================================================

REAL_CACHE = default_cache_dir(MANIFEST_ROOT) if MANIFEST_ROOT.exists() else None


@pytest.mark.skipif(REAL_CACHE is None or not REAL_CACHE.exists(),
                    reason="build the full window cache first")
def test_published_cache_matches_the_committed_manifests():
    cache = WindowCache(REAL_CACHE, MANIFEST_ROOT)
    meta = cache.metadata
    assert meta["entry_count"] == len(cache) == 51_561
    assert meta["dataset_counts"] == {"cwru": 12_777, "paderborn": 38_784}
    assert meta["manifest_digest"] == manifest_digest(MANIFEST_ROOT)
    assert meta["preprocessing_digest"] == preprocessing_digest()
    assert meta["validation"]["passed"] is True
    assert meta["validation"]["hash_mismatches"] == 0
    assert meta["validation"]["dc_violations"] == 0
    rng = np.random.default_rng(SPLIT_SEED)
    sample = [cache.sample_ids[i] for i in rng.choice(len(cache), 200, replace=False)]
    assert cache.rederive_from_raw(sample)["mismatches"] == 0
