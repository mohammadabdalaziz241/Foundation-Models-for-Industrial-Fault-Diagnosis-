"""Final frozen representation reader — methodology_v2 Part 4C.

    get_representation(window_id, fold_id) -> (tensor, meta)

Given a sealed Part-3B window identity and a fold, returns the exact
model-domain tensor a Part-5 model receives:

  raw 1-s native-rate window (Part-3B lazy reader)
  -> approved dataset-specific STFT (periodic Hann, center=False,
     no padding, one-sided)
  -> log1p magnitude
  -> N2 normalization with the sealed normalizer[fold][dataset]
  -> float32, layout (freq_bins, time_frames)

meta exposes frequency_hz, time_seconds, shape, dataset, split and
provenance. All upstream seals (Part-2, Part-3B, Part-4C) are verified
fail-closed once per process before any representation is served.
Validation/test access is legal (preprocessing is frozen) and NEVER
updates statistics — the normalizer arrays are loaded read-only and
their on-disk artifacts stay sealed.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from .part2_builder import verify_frozen_hashes
from .part3b_reader import read_window
from .part3b_windows import PART3B_DIR, verify_part3b_hashes
from .part4b_freeze import TF_CONFIG, RATES, rep_of
from .part4c_normalizers import (Part4CError, load_normalizer,
                                 verify_part4c_hashes)

_SEALS_OK = False


def _verify_seals_once() -> None:
    global _SEALS_OK
    if not _SEALS_OK:
        verify_frozen_hashes()
        verify_part3b_hashes()
        verify_part4c_hashes()
        _SEALS_OK = True


@lru_cache(maxsize=3)
def _manifest(fold: int) -> pd.DataFrame:
    df = pd.read_csv(PART3B_DIR / f"window_manifest_fold_{fold}.csv")
    return df.set_index("window_id")


@lru_cache(maxsize=12)
def _normalizer(fold: int, dataset: str) -> dict:
    n = load_normalizer(fold, dataset)
    n["mean"].setflags(write=False)
    n["std_denominator"].setflags(write=False)
    n["frequency_hz"].setflags(write=False)
    return n


def get_representation(window_id: str,
                       fold_id: int) -> tuple[np.ndarray, dict]:
    _verify_seals_once()
    man = _manifest(fold_id)
    if window_id not in man.index:
        raise Part4CError(f"{window_id}: not in fold-{fold_id} manifest")
    row = man.loc[window_id]
    ds = row["dataset"]
    n_fft, hop = TF_CONFIG[ds]
    fs = RATES[ds]

    rep = rep_of(read_window(row), ds)          # (frames, bins), float64
    norm = _normalizer(fold_id, ds)
    x = (rep - norm["mean"]) / norm["std_denominator"]
    tensor = np.ascontiguousarray(x.T, dtype=np.float32)  # (bins, frames)

    n_frames = rep.shape[0]
    meta = {
        "window_id": window_id, "fold_id": fold_id, "dataset": ds,
        "split": row["split"], "original_label": row["original_label"],
        "channel": row["channel"],
        "sampling_rate_hz": fs, "n_fft": n_fft, "hop": hop,
        "shape": tensor.shape,
        "frequency_hz": norm["frequency_hz"],
        "time_seconds": np.arange(n_frames) * hop / fs,
        "normalization": "N2 per-fold per-dataset per-bin (frozen)",
        "n_floored_bins": int(norm["floored_bins"].size),
        "source_file": row["source_file"],
        "group_id": row["group_id"],
    }
    if not np.isfinite(tensor).all():
        raise Part4CError(f"{window_id}: non-finite representation")
    return tensor, meta
