import numpy as np
import pytest

from src.unified_data import SampleMetadata, assert_no_exact_duplicates, assert_no_raw_overlap, exact_window_hash, split_temporal_regions, window_region


def _meta(sample_id, split, start, end, digest):
    return SampleMetadata(sample_id, "demo", "normal", "normal", "b1", "r1", "op", 1, split, "horizontal", "x", start, end, 0, digest)


def test_temporal_regions_and_windows_do_not_cross_boundaries():
    x = np.arange(10_000, dtype=np.float32)
    regions = split_temporal_regions(len(x))
    assert [(r.start, r.end) for r in regions] == [(0, 7000), (7000, 8500), (8500, 10000)]
    for region in regions:
        for window, start, end in window_region(x, region, window_len=1024, step=512):
            assert region.start <= start < end <= region.end
            assert len(window) == 1024


def test_overlap_and_duplicate_guards():
    a = _meta("a", "train", 0, 1024, "same")
    b = _meta("b", "val", 512, 1536, "other")
    with pytest.raises(AssertionError, match="raw-point leakage"):
        assert_no_raw_overlap([a, b])
    c = _meta("c", "test", 2048, 3072, "same")
    with pytest.raises(AssertionError, match="exact duplicate"):
        assert_no_exact_duplicates([a, c])


def test_exact_hash_is_dtype_stable_for_numeric_values():
    assert exact_window_hash(np.arange(8, dtype=np.float32)) == exact_window_hash(np.arange(8, dtype=np.float64))
