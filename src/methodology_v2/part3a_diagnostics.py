"""Part 3A fold-1 TRAIN-ONLY signal diagnostics (corroborative evidence).

SCOPE AND LEAKAGE DECLARATION
-----------------------------
These diagnostics read raw signal content and are therefore restricted to
recordings whose split == 'train' in GLOBAL FOLD 1. They are labelled
fold-1-train-specific, are used only to corroborate metadata/physics-based
recommendations, and must never silently drive a parameter for another
fold. Because the union of TRAIN groups across folds covers all groups,
pooling such diagnostics across folds would de facto touch every test
partition — so no cross-fold pooling is performed.

The guard `_assert_train_fold1` fails closed if any non-train recording is
requested. JNU is read with max_rows so validation/test temporal regions
of fold 1 are never loaded into memory. HIT is sliced via mmap to train
recordings only. Every recording actually read is written to the
diagnostics JSON for the leakage test to audit.

Computed:
  - cumulative energy fractions below 5 / 8 / 10 / 12.5 kHz per dataset
    (native-rate one-sided energy spectrum, candidate primary channel);
  - HIT adjacent-series continuity check (boundary first-difference vs
    within-series first-difference), evidence for whether future windows
    may cross 20480-sample series boundaries inside one recording.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .part2_builder import PART2_DIR
from .registry import DATA_ROOT, OUTPUT_DIR as PART1_DIR, REPO_ROOT

EDGES_HZ = (5_000.0, 8_000.0, 10_000.0, 12_000.0, 12_500.0, 16_000.0)
MAFAULDA_SUBSAMPLE_STEP = 5  # deterministic: every 5th sorted train rec


class SealedDataError(AssertionError):
    """Raised if a diagnostic would touch non-train fold-1 signal data."""


def _fold1() -> pd.DataFrame:
    return pd.read_csv(PART2_DIR / "global_fold_1.csv")


def _assert_train_fold1(fold1: pd.DataFrame, recording_id: str) -> None:
    rows = fold1[fold1["recording_id"] == recording_id]
    if rows.empty:
        raise SealedDataError(f"{recording_id}: not in fold-1 manifest")
    splits = set(rows["split"]) - {"guard"}
    if splits != {"train"}:
        raise SealedDataError(
            f"{recording_id}: fold-1 split(s) {splits} — raw signal "
            "content is sealed for design diagnostics (train only)")


def _energy_fractions(x: np.ndarray, rate: float) -> dict:
    x = np.asarray(x, dtype=np.float64).ravel()
    x = x - x.mean()  # DC removed so fractions describe dynamic content
    spec = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(x.size, d=1.0 / rate)
    total = spec.sum()
    return {f"below_{e/1000:g}kHz":
            float(spec[freqs <= e].sum() / total) for e in EDGES_HZ}


def _agg(per_rec: list[dict]) -> dict:
    out = {}
    for key in per_rec[0]:
        vals = [d[key] for d in per_rec]
        out[key] = {"mean": round(float(np.mean(vals)), 4),
                    "min": round(float(np.min(vals)), 4)}
    return out


def run_diagnostics(out_dir: Path) -> dict:
    import scipy.io as sio
    fold1 = _fold1()
    read_log: dict[str, list[str]] = {}
    result: dict = {"scope": "GLOBAL FOLD 1 TRAIN ONLY — corroborative; "
                             "never a silent cross-fold parameter source"}

    # ---- CWRU: DE channel of the 20 fold-1 train recordings -------------
    cw = fold1[(fold1["dataset"] == "CWRU") & (fold1["split"] == "train")]
    fr = []
    for _, r in cw.iterrows():
        _assert_train_fold1(fold1, r["recording_id"])
        mat = sio.loadmat(str(REPO_ROOT / r["source_file"]))
        pid = r["recording_id"].removeprefix("cwru_")
        x = mat[f"{pid}_DE_time"]
        fr.append(_energy_fractions(x, 48_000))
    result["CWRU"] = {"channel": "DE_time", "native_rate": 48_000,
                      "n_recordings": len(fr), "energy_fraction": _agg(fr)}
    read_log["CWRU"] = cw["recording_id"].tolist()

    # ---- JNU: train macro-blocks A+B+C only (first 3N/5 samples) --------
    # JNU seals at TEMPORAL-REGION level, not recording level: a recording
    # legitimately spans all three splits through disjoint blocks. The
    # sealed rule here is: only samples inside fold-1 TRAIN blocks may be
    # read. In fold 1 the train blocks are A,B,C — contiguous from sample
    # 0 — so reading with max_rows = end(C) provably never loads the
    # validation (D) or test (E) regions.
    jn = fold1[(fold1["dataset"] == "JNU") & (fold1["split"] == "train")]
    fr = []
    jn_ids = []
    for rec_id, grp in jn.groupby("recording_id"):
        blocks = set(grp["temporal_block_id"])
        if blocks != {"A", "B", "C"}:
            raise SealedDataError(
                f"JNU fold-1 train blocks unexpected: {blocks}")
        if int(grp["temporal_start_sample"].min()) != 0:
            raise SealedDataError(
                f"{rec_id}: fold-1 train region does not start at 0; "
                "max_rows strategy would touch sealed samples")
        n_train = int(grp["temporal_end_sample"].max())
        path = REPO_ROOT / grp.iloc[0]["source_file"]
        x = np.loadtxt(path, max_rows=n_train)  # D/E never loaded
        fr.append(_energy_fractions(x, 50_000))
        jn_ids.append(rec_id)
    result["JNU"] = {"channel": "acc_vertical (blocks A-C)",
                     "native_rate": 50_000, "n_recordings": len(fr),
                     "energy_fraction": _agg(fr)}
    read_log["JNU"] = jn_ids

    # ---- HIT: ch3 of the 94 fold-1 train recordings + continuity --------
    hit = fold1[(fold1["dataset"] == "HIT") & (fold1["split"] == "train")]
    arrays = {s: np.load(DATA_ROOT / "raw_hit" / "gdrive_full"
                         / "HIT-dataset" / f"{s.removeprefix('hit_')}.npy",
                         mmap_mode="r")
              for s in sorted(hit["session"].unique())}
    fr, boundary_ratios = [], []
    for _, r in hit.iterrows():
        _assert_train_fold1(fold1, r["recording_id"])
        k = int(r["recording_id"].rsplit("rec", 1)[1])
        arr = arrays[r["session"]]
        block = np.asarray(arr[18 * k:18 * (k + 1), 2, :], dtype=np.float64)
        fr.append(_energy_fractions(block.ravel(), 25_000))
        within_p999 = np.percentile(
            np.abs(np.diff(block, axis=1)), 99.9)
        jumps = np.abs(block[1:, 0] - block[:-1, -1])
        boundary_ratios.extend((jumps / within_p999).tolist())
    br = np.asarray(boundary_ratios)
    result["HIT"] = {
        "channel": "ch3 (first casing accelerometer)",
        "native_rate": 25_000, "n_recordings": int(len(hit)),
        "energy_fraction": _agg(fr),
        "series_boundary_continuity": {
            "n_boundaries": int(br.size),
            "median_jump_ratio": round(float(np.median(br)), 3),
            "p95_jump_ratio": round(float(np.percentile(br, 95)), 3),
            "max_jump_ratio": round(float(br.max()), 3),
            "interpretation": "ratio ~<=1 means adjacent 20480-sample "
                              "series join like ordinary neighbouring "
                              "samples (contiguous segmentation)",
        }}
    read_log["HIT"] = hit["recording_id"].tolist()

    # ---- MaFaulDa: col3 (underhang radial), every 5th train recording ---
    mf = (fold1[(fold1["dataset"] == "MAFAULDA")
                & (fold1["split"] == "train")]
          .sort_values("recording_id"))
    sel = mf.iloc[::MAFAULDA_SUBSAMPLE_STEP]
    fr = []
    for _, r in sel.iterrows():
        _assert_train_fold1(fold1, r["recording_id"])
        a = pd.read_csv(REPO_ROOT / r["source_file"], header=None,
                        usecols=[2], dtype=np.float64).to_numpy().ravel()
        fr.append(_energy_fractions(a, 50_000))
    result["MAFAULDA"] = {
        "channel": "col3 underhang radial",
        "native_rate": 50_000,
        "n_recordings": len(fr),
        "subsample_rule": f"every {MAFAULDA_SUBSAMPLE_STEP}th of sorted "
                          "fold-1 train recordings (deterministic)",
        "energy_fraction": _agg(fr)}
    read_log["MAFAULDA"] = sel["recording_id"].tolist()

    result["recordings_read"] = read_log
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "fold1_train_diagnostics.json", "w") as f:
        json.dump(result, f, indent=1)
    return result
