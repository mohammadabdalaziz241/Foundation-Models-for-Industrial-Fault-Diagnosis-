"""Tests for the frozen vibrationclip_v1 preprocessing (Amendment 4 §5)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.vibrationclip.preprocessing import (  # noqa: E402
    DC_TOLERANCE, MODEL_FS, N_FRAMES, NATIVE_FS, OUT_SHAPE, RESAMPLE_RATIOS,
    WINDOW_LEN, WINDOW_STEP, assert_dc_removed, dc_remove,
    exact_window_hash, expected_resampled_length, log_spectrogram, n_windows,
    resample_to_model_rate, segment, spectrogram_metadata,
)

RNG = np.random.default_rng(42)


def test_frozen_constants():
    assert MODEL_FS == 24_000
    assert WINDOW_LEN == 24_000 and WINDOW_STEP == 12_000
    assert N_FRAMES == 125
    assert OUT_SHAPE == (128, 128)
    for ds, (up, down) in RESAMPLE_RATIOS.items():
        assert NATIVE_FS[ds] * up == MODEL_FS * down


@pytest.mark.parametrize("dataset", sorted(RESAMPLE_RATIOS))
def test_resample_length_closed_form(dataset):
    n = NATIVE_FS[dataset] * 3 + 17  # awkward length on purpose
    y = resample_to_model_rate(RNG.normal(size=n), dataset)
    assert len(y) == expected_resampled_length(n, dataset)


def test_segment_never_crosses_boundary():
    sig = np.arange(WINDOW_LEN * 2 + 5_000, dtype=np.float32)
    wins = list(segment(sig))
    assert len(wins) == n_windows(len(sig)) == 3
    for start, end, w in wins:
        assert end - start == WINDOW_LEN and end <= len(sig)
        assert w[0] == start  # identity signal: window really starts at start
    sig_short = np.zeros(WINDOW_LEN - 1)
    assert list(segment(sig_short)) == [] and n_windows(len(sig_short)) == 0


def test_dc_removal_and_tolerance():
    w = 3.7 + 2.5 * RNG.normal(size=WINDOW_LEN)
    d = dc_remove(w)
    assert abs(d.mean()) < DC_TOLERANCE
    assert_dc_removed(d[None, :])
    with pytest.raises(ValueError):
        assert_dc_removed(w[None, :])
    # amplitude preserved: not z-scored
    assert not np.isclose(d.std(), 1.0, atol=0.05)


def test_spectrogram_shape_dtype_determinism():
    w = dc_remove(RNG.normal(size=WINDOW_LEN))
    s1, s2 = log_spectrogram(w), log_spectrogram(w.copy())
    assert s1.shape == OUT_SHAPE and s1.dtype == np.float32
    assert np.isfinite(s1).all()
    assert np.array_equal(s1, s2)
    # trailing time padding is exactly zero
    assert np.all(s1[:, N_FRAMES:] == 0.0)
    assert np.any(s1[:, N_FRAMES - 1] != 0.0)


def test_spectrogram_rejects_nonwindow_input():
    with pytest.raises(ValueError):
        log_spectrogram(np.zeros(1000))
    with pytest.raises(ValueError):
        log_spectrogram(np.full(WINDOW_LEN, 2.5))  # DC not removed


@pytest.mark.parametrize("dataset", sorted(RESAMPLE_RATIOS))
def test_commensurate_frequency_axes(dataset):
    """A 1.5 kHz tone at each native rate lands in the SAME output row.

    This is the Gate B axis-commensurability guarantee: after resampling to
    the common 24 kHz, physical frequency -> spectrogram row is one mapping
    for every dataset (93.75 Hz per row => 1500 Hz -> row 16).
    """
    fs = NATIVE_FS[dataset]
    t = np.arange(int(fs * 1.6)) / fs
    tone = np.sin(2 * np.pi * 1_500.0 * t)
    y = resample_to_model_rate(tone, dataset)
    w = dc_remove(y[:WINDOW_LEN])
    s = log_spectrogram(w)
    row_energy = s[:, :N_FRAMES].mean(axis=1)
    assert int(np.argmax(row_energy)) == 16
    # and a 4.5 kHz tone -> row 48, still identical across datasets
    tone2 = np.sin(2 * np.pi * 4_500.0 * t)
    w2 = dc_remove(resample_to_model_rate(tone2, dataset)[:WINDOW_LEN])
    assert int(np.argmax(log_spectrogram(w2)[:, :N_FRAMES].mean(axis=1))) == 48


def test_exact_hash_is_float32_bytes():
    w = dc_remove(RNG.normal(size=WINDOW_LEN))
    import hashlib
    assert exact_window_hash(w) == hashlib.sha256(
        w.astype(np.float32).tobytes()).hexdigest()


def test_metadata_records_all_frozen_parameters():
    md = spectrogram_metadata()
    for key in ("model_sample_rate", "window_duration_s", "n_fft", "hop",
                "stft_window", "output_shape", "resample_ratios",
                "freq_hz_per_row", "time_s_per_frame"):
        assert key in md
    assert md["window_duration_s"] == 1.0
    assert md["freq_hz_per_row"] == 93.75
