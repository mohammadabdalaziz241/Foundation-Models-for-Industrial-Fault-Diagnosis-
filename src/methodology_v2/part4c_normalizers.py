"""Part 4C — fit, seal and register the fold-specific N2 normalizers.

Implements the APPROVED frozen representation (Part 4B): per fold x
dataset x frequency-bin TRAIN statistics over the log1p(|STFT|) values,
computed with a numerically stable streaming (parallel Welford) merge in
float64 — no full-dataset tensors are ever held in memory and no
spectrogram files are written.

Fold independence and TRAIN-only fitting are enforced by construction
and by a fail-closed guard. Near-zero-variance bins follow the
predeclared rule: bins with std < STD_TOL keep their raw std recorded,
but their normalization DENOMINATOR is replaced by exactly 1.0 and every
affected bin is reported. No frequency bin is ever deleted.

Artifacts are byte-deterministic (.npz written with fixed zip metadata)
and sealed under a Part-4C master representation hash; Part-5 code must
fail closed via verify_part4c_hashes().
"""
from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .integrity import sha256_file
from .part3b_windows import PART3B_DIR
from .part4b_freeze import TF_CONFIG, RATES, rep_of, frequency_hz
from .registry import REPO_ROOT

PART4C_DIR = REPO_ROOT / "methodology_v2" / "part4_representation_final"
NORM_DIR = PART4C_DIR / "normalizers"

STD_TOL = 1e-6          # predeclared near-zero-variance tolerance (std of
FLOOR_DENOMINATOR = 1.0  # log1p values); affected denominators := 1.0
DATASETS = ("CWRU", "JNU", "HIT", "MAFAULDA")
FOLD_IDS = (1, 2, 3)


class Part4CError(AssertionError):
    """Raised loudly on protocol violation or seal failure."""


# ---------------------------------------------------------------------------
# streaming statistics (parallel Welford merge, float64)
# ---------------------------------------------------------------------------

class StreamingBinStats:
    """Per-frequency-bin running count/mean/M2 over spectrogram frames."""

    def __init__(self, n_bins: int):
        self.n = 0
        self.mean = np.zeros(n_bins, dtype=np.float64)
        self.m2 = np.zeros(n_bins, dtype=np.float64)
        self.n_windows = 0

    def add_window(self, rep: np.ndarray) -> None:
        """rep: (frames, bins) float64 — one window's representation."""
        nb = rep.shape[0]
        mb = rep.mean(axis=0)
        m2b = ((rep - mb) ** 2).sum(axis=0)
        if self.n == 0:
            self.n, self.mean, self.m2 = nb, mb, m2b
        else:
            delta = mb - self.mean
            tot = self.n + nb
            self.mean = self.mean + delta * (nb / tot)
            self.m2 = self.m2 + m2b + delta ** 2 * (self.n * nb / tot)
            self.n = tot
        self.n_windows += 1

    def finalize(self) -> dict:
        if self.n < 2:
            raise Part4CError("insufficient frames for statistics")
        std_raw = np.sqrt(self.m2 / self.n)   # population std
        floored = np.where(std_raw < STD_TOL)[0]
        denom = std_raw.copy()
        denom[floored] = FLOOR_DENOMINATOR
        return {"mean": self.mean, "std_raw": std_raw,
                "std_denominator": denom,
                "floored_bins": floored.astype(np.int64),
                "n_frames": self.n, "n_windows": self.n_windows}


# ---------------------------------------------------------------------------
# deterministic .npz writer (fixed zip metadata -> byte-stable hashing)
# ---------------------------------------------------------------------------

def deterministic_savez(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as z:
        for name in sorted(arrays):
            buf = io.BytesIO()
            np.save(buf, np.asarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy",
                                   date_time=(1980, 1, 1, 0, 0, 0))
            z.writestr(info, buf.getvalue())


# ---------------------------------------------------------------------------
# fitting
# ---------------------------------------------------------------------------

def train_windows(fold: int, dataset: str) -> pd.DataFrame:
    """TRAIN rows of one fold/dataset — the ONLY legal fitting input."""
    man = pd.read_csv(PART3B_DIR / f"window_manifest_fold_{fold}.csv")
    sub = man[(man["dataset"] == dataset) & (man["split"] == "train")]
    if sub.empty:
        raise Part4CError(f"no TRAIN windows for fold {fold} {dataset}")
    bad = sub[(sub["fold_id"] != fold) | (sub["split"] != "train")]
    if len(bad):
        raise Part4CError("non-TRAIN row reached the fitting path")
    return sub.sort_values(["recording_id", "start_sample"])


def assert_train_row(row: pd.Series, fold: int) -> None:
    if row["split"] != "train" or int(row["fold_id"]) != fold:
        raise Part4CError(
            f"{row['window_id']}: split={row['split']} fold="
            f"{row['fold_id']} — normalizer fitting is TRAIN-only and "
            "fold-isolated (fail closed)")


def fit_one(fold: int, dataset: str) -> dict:
    """Fit mu/std for one (fold, dataset) from its TRAIN windows only."""
    from .part3b_reader import read_window
    sub = train_windows(fold, dataset)
    n_bins = TF_CONFIG[dataset][0] // 2 + 1
    stats = StreamingBinStats(n_bins)
    for _, row in sub.iterrows():
        assert_train_row(row, fold)
        rep = rep_of(read_window(row), dataset)
        stats.add_window(rep)
    out = stats.finalize()
    out |= {"fold": fold, "dataset": dataset,
            "frequency_hz": frequency_hz(dataset)}
    return out


def _fit_job(args: tuple[int, str]) -> tuple[int, str, dict]:
    fold, dataset = args
    return fold, dataset, fit_one(fold, dataset)


def fit_all(n_workers: int = 3) -> dict:
    from multiprocessing import Pool
    jobs = [(f, d) for f in FOLD_IDS for d in DATASETS]
    if n_workers > 1:
        with Pool(n_workers) as pool:
            results = pool.map(_fit_job, jobs)
    else:
        results = [_fit_job(j) for j in jobs]
    return {(f, d): r for f, d, r in results}


# ---------------------------------------------------------------------------
# artifacts, registry, sealing
# ---------------------------------------------------------------------------

def write_artifacts(fitted: dict, spec_path: Path) -> dict:
    NORM_DIR.mkdir(parents=True, exist_ok=True)
    spec_hash = sha256_file(spec_path)
    man_hashes = {f: sha256_file(PART3B_DIR
                                 / f"window_manifest_fold_{f}.csv")
                  for f in FOLD_IDS}
    registry_rows, fitstat_rows = [], []
    for (fold, ds), r in sorted(fitted.items()):
        d = NORM_DIR / f"fold_{fold}"
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{ds.lower()}.npz"
        n_fft, hop = TF_CONFIG[ds]
        deterministic_savez(path, {
            "mean": r["mean"], "std_raw": r["std_raw"],
            "std_denominator": r["std_denominator"],
            "frequency_hz": r["frequency_hz"],
            "floored_bins": r["floored_bins"],
            "n_frames": np.int64(r["n_frames"]),
            "n_windows": np.int64(r["n_windows"]),
            "fold": np.int64(fold),
            "n_fft": np.int64(n_fft), "hop": np.int64(hop),
            "sampling_rate_hz": np.int64(RATES[ds]),
        })
        registry_rows.append({
            "fold": fold, "dataset": ds,
            "file": str(path.relative_to(PART4C_DIR)),
            "n_fft": n_fft, "hop": hop,
            "sampling_rate_hz": RATES[ds],
            "freq_bins": r["mean"].size,
            "bin_spacing_hz": round(RATES[ds] / n_fft, 6),
            "n_train_windows": r["n_windows"],
            "n_frames": r["n_frames"],
            "n_floored_bins": int(r["floored_bins"].size),
            "std_floor_rule": f"std < {STD_TOL} -> denominator "
                              f"{FLOOR_DENOMINATOR}",
            "train_manifest_sha256": man_hashes[fold],
            "representation_spec_sha256": spec_hash,
        })
        fitstat_rows.append({
            "fold": fold, "dataset": ds,
            "n_train_windows": r["n_windows"],
            "n_frames": r["n_frames"],
            "frames_per_bin": r["n_frames"],
            "mean_of_bin_means": round(float(r["mean"].mean()), 6),
            "min_bin_std": float(r["std_raw"].min()),
            "max_bin_std": float(r["std_raw"].max()),
            "n_floored_bins": int(r["floored_bins"].size),
        })
    registry = pd.DataFrame(registry_rows)
    registry.to_csv(PART4C_DIR / "normalizer_registry.csv", index=False)
    pd.DataFrame(fitstat_rows).to_csv(
        PART4C_DIR / "normalizer_fit_statistics.csv", index=False)

    hash_rows = []
    for name in ["representation_spec.yaml", "normalizer_registry.csv"]:
        hash_rows.append({"file": name,
                          "sha256": sha256_file(PART4C_DIR / name)})
    for (fold, ds) in sorted(fitted):
        rel = f"normalizers/fold_{fold}/{ds.lower()}.npz"
        hash_rows.append({"file": rel,
                          "sha256": sha256_file(PART4C_DIR / rel)})
    master_src = "".join(f"{r['file']}:{r['sha256']}\n"
                         for r in sorted(hash_rows,
                                         key=lambda r: r["file"]))
    master = hashlib.sha256(master_src.encode()).hexdigest()
    hash_rows.append({"file": "PART4C_MASTER_REPRESENTATION_HASH",
                      "sha256": master})
    pd.DataFrame(hash_rows).to_csv(PART4C_DIR / "normalizer_hashes.csv",
                                   index=False)
    return {"master": master, "registry": registry}


def verify_part4c_hashes(base: Path | None = None) -> None:
    """FAIL CLOSED if any sealed Part-4C artifact changed."""
    base = Path(base) if base else PART4C_DIR
    rec = pd.read_csv(base / "normalizer_hashes.csv")
    stored = {r["file"]: r["sha256"] for _, r in rec.iterrows()}
    entries = []
    for name, expect in stored.items():
        if name == "PART4C_MASTER_REPRESENTATION_HASH":
            continue
        got = sha256_file(base / name)
        if got != expect:
            raise Part4CError(
                f"FROZEN PART-4C ARTIFACT CHANGED: {name} (fail closed)")
        entries.append((name, got))
    master_src = "".join(f"{n}:{h}\n" for n, h in sorted(entries))
    if hashlib.sha256(master_src.encode()).hexdigest() \
            != stored["PART4C_MASTER_REPRESENTATION_HASH"]:
        raise Part4CError("Part-4C master hash mismatch (fail closed)")


def load_normalizer(fold: int, dataset: str) -> dict:
    path = NORM_DIR / f"fold_{fold}" / f"{dataset.lower()}.npz"
    with np.load(path) as z:
        return {k: z[k].copy() for k in z.files}
