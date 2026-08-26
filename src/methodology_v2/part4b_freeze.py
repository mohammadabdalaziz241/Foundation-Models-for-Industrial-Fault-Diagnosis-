"""Part 4B — final representation verification and N1/N2 normalization
comparison (Fold-1 TRAIN only; nothing frozen without approval).

FROZEN physically matched analysis configuration (approved):
    CWRU     48 kHz : n_fft 1024, hop 256
    JNU      50 kHz : n_fft 1024, hop 256
    HIT      25 kHz : n_fft  512, hop 128   (physically matched: ~20.5 ms
                                             frame, ~5.1 ms hop, ~48.8 Hz)
    MAFAULDA 50 kHz : n_fft 1024, hop 256
Conventions inherited from Part 4A (periodic Hann, center=False, no
padding, one-sided rfft). FINAL value transform: log1p(|TF|). Frequency
bins keep physical Hz: f_k = k * fs / n_fft (metadata only; no embedding
implemented here).

Normalization candidates compared on Fold-1 TRAIN content only:
  N1  per-window:  (X - mean(X)) / max(std(X), STD_FLOOR)
  N2  per-dataset, per-frequency-bin TRAIN statistics:
      (X[:, f] - mu[D, f]) / max(std[D, f], STD_FLOOR)
      DEV-SAMPLE estimate only in Part 4B — the real fold-specific
      normalizers are fitted AFTER approval, one per fold.
STD_FLOOR = 1e-8; windows/bins hitting the floor are counted and
reported, never silently altered beyond the documented floor.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .part3b_windows import PART3B_DIR
from .part4a_repdesign import (apply_transform, assert_fold1_train,
                               load_fold1_train, select_dev_windows,
                               tf_map)
from .registry import REPO_ROOT

PART4B_DIR = REPO_ROOT / "methodology_v2" / "part4_representation_freeze"

TF_CONFIG = {"CWRU": (1024, 256), "JNU": (1024, 256),
             "HIT": (512, 128), "MAFAULDA": (1024, 256)}
RATES = {"CWRU": 48_000, "JNU": 50_000, "HIT": 25_000, "MAFAULDA": 50_000}
FINAL_TRANSFORM = "log1p"
STD_FLOOR = 1e-8
STATS_SAMPLE_STEP = 20   # every 20th Fold-1 TRAIN window (sorted) for the
                         # N2 dev-sample statistics


def rep_of(x: np.ndarray, ds: str) -> np.ndarray:
    """Final frozen representation candidate: log1p(|TF|), (frames, bins)."""
    n_fft, hop = TF_CONFIG[ds]
    return apply_transform(tf_map(x, n_fft, hop), FINAL_TRANSFORM)


def verification_table() -> pd.DataFrame:
    rows = []
    for ds, (n_fft, hop) in TF_CONFIG.items():
        fs = RATES[ds]
        frames = (fs - n_fft) // hop + 1
        rows.append({
            "dataset": ds, "sampling_rate_hz": fs, "n_fft": n_fft,
            "hop": hop, "analysis_ms": round(1000 * n_fft / fs, 3),
            "hop_ms": round(1000 * hop / fs, 3),
            "freq_resolution_hz": round(fs / n_fft, 3),
            "freq_bins": n_fft // 2 + 1, "time_frames": frames,
            "nyquist_hz": fs / 2,
            "tensor_shape": f"({n_fft // 2 + 1}, {frames})",
            "transform": FINAL_TRANSFORM,
            "window_fn": "periodic Hann, center=False, no padding, "
                         "one-sided rfft",
        })
    return pd.DataFrame(rows)


def frequency_hz(ds: str) -> np.ndarray:
    """Physical frequency of every bin (metadata for the future encoder)."""
    n_fft, _ = TF_CONFIG[ds]
    return np.arange(n_fft // 2 + 1) * RATES[ds] / n_fft


# ---------------------------------------------------------------------------
# normalizers
# ---------------------------------------------------------------------------

def n1_normalize(rep: np.ndarray) -> tuple[np.ndarray, bool]:
    mu, sd = float(rep.mean()), float(rep.std())
    floored = sd < STD_FLOOR
    return (rep - mu) / max(sd, STD_FLOOR), floored


class N2Stats:
    """Per-dataset, per-frequency-bin statistics from Fold-1 TRAIN
    dev-sample windows only (estimate for comparison — NOT the final
    fold normalizers, which require approval and per-fold fitting)."""

    def __init__(self):
        self._acc: dict[str, list] = {}

    def add(self, ds: str, rep: np.ndarray) -> None:
        n_bins = rep.shape[1]
        if ds not in self._acc:
            self._acc[ds] = [np.zeros(n_bins), np.zeros(n_bins), 0]
        s, s2, n = self._acc[ds]
        s += rep.sum(axis=0)
        s2 += (rep ** 2).sum(axis=0)
        self._acc[ds][2] = n + rep.shape[0]

    def finalize(self) -> dict[str, dict]:
        out = {}
        for ds, (s, s2, n) in self._acc.items():
            mu = s / n
            var = np.maximum(s2 / n - mu ** 2, 0.0)
            sd = np.sqrt(var)
            out[ds] = {"mu": mu, "std": sd, "n_frames": int(n),
                       "n_floored_bins": int((sd < STD_FLOOR).sum())}
        return out


def n2_normalize(rep: np.ndarray, stats: dict) -> np.ndarray:
    return (rep - stats["mu"]) / np.maximum(stats["std"], STD_FLOOR)


# ---------------------------------------------------------------------------
# development windows (Fold-1 TRAIN only, deterministic, guarded)
# ---------------------------------------------------------------------------

def build_dev_sets() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (train_manifest, qualitative_66, stats_sample)."""
    train = load_fold1_train()
    qual = select_dev_windows(train)          # identical to Part-4A set
    sample = (train.sort_values("window_id").iloc[::STATS_SAMPLE_STEP])
    for _, w in sample.iterrows():
        assert_fold1_train(w)
    return train, qual, sample


# ---------------------------------------------------------------------------
# the bounded comparison study
# ---------------------------------------------------------------------------

def run_study(out_dir: Path, read_window) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    train, qual, sample = build_dev_sets()
    train_by_id = train.set_index("window_id")

    dev_rows = [{"window_id": w["window_id"], "dataset": w["dataset"],
                 "class": w["original_label"], "role": "stats_sample"}
                for _, w in sample.iterrows()]
    dev_rows += [{"window_id": r["window_id"], "dataset": r["dataset"],
                  "class": r["class"], "role": "qualitative"}
                 for _, r in qual.iterrows()]
    pd.DataFrame(dev_rows).to_csv(out_dir / "part4b_development_windows.csv",
                                  index=False)

    verification_table().to_csv(
        out_dir / "physically_matched_stft_verification.csv", index=False)

    # ---- pass 1: N2 dev-sample statistics + per-window scalars ----------
    stats = N2Stats()
    scalars = []
    profiles: dict[str, np.ndarray] = {}
    for _, w in sample.iterrows():
        assert_fold1_train(w)
        rep = rep_of(read_window(w), w["dataset"])
        stats.add(w["dataset"], rep)
        profiles[w["window_id"]] = rep.mean(axis=0)
        scalars.append({"window_id": w["window_id"],
                        "dataset": w["dataset"],
                        "pre_mean": float(rep.mean()),
                        "pre_std": float(rep.std()),
                        "pre_p99": float(np.percentile(rep, 99))})
    n2 = stats.finalize()
    sc = pd.DataFrame(scalars)

    # analytic post-normalization window means
    post2 = []
    for _, r in sc.iterrows():
        st = n2[r["dataset"]]
        prof = profiles[r["window_id"]]
        post2.append(float(np.mean((prof - st["mu"])
                                   / np.maximum(st["std"], STD_FLOOR))))
    sc["post_mean_n2"] = post2
    sc["post_mean_n1"] = 0.0   # exact, by construction

    # ---- amplitude retention (rank correlations over the sample) -------
    from scipy.stats import spearmanr
    amp_rows = []
    for ds, grp in sc.groupby("dataset"):
        rho2 = spearmanr(grp["pre_mean"], grp["post_mean_n2"]).statistic
        big = grp.loc[grp["pre_mean"].idxmax()]
        amp_rows.append({
            "dataset": ds, "n_windows": len(grp),
            "rank_corr_pre_energy_vs_postN1": 0.0,
            "rank_corr_pre_energy_vs_postN2": round(float(rho2), 4),
            "window_mean_spread_preN": round(float(grp["pre_mean"].std()),
                                             4),
            "window_mean_spread_N1": 0.0,
            "window_mean_spread_N2":
                round(float(grp["post_mean_n2"].std()), 4),
            "largest_window_postN2_mean":
                round(float(big["post_mean_n2"]), 3),
        })
    amp = pd.DataFrame(amp_rows)
    amp.to_csv(out_dir / "amplitude_retention_metrics.csv", index=False)

    # ---- dataset-scale shortcut table -----------------------------------
    scale_rows = []
    for ds, grp in sc.groupby("dataset"):
        st = n2[ds]
        scale_rows.append({
            "dataset": ds,
            "unnormalized_mean": round(float(grp["pre_mean"].mean()), 4),
            "unnormalized_std": round(float(grp["pre_std"].mean()), 4),
            "N1_mean": 0.0, "N1_std": 1.0,
            "N2_mean": round(float(grp["post_mean_n2"].mean()), 4),
            "N2_within_dataset_mean_spread":
                round(float(grp["post_mean_n2"].std()), 4),
        })
    pd.DataFrame(scale_rows).to_csv(out_dir / "dataset_scale_metrics.csv",
                                    index=False)

    # ---- pass 2: full metrics on the 66 qualitative windows -------------
    comp_rows, num_rows = [], []
    for _, d in qual.iterrows():
        w = train_by_id.loc[d["window_id"]]
        assert_fold1_train(w)
        rep = rep_of(read_window(w), d["dataset"])
        st = n2[d["dataset"]]
        x1, floored = n1_normalize(rep)
        x2 = n2_normalize(rep, st)
        fprof = rep.mean(axis=0)
        tprof = rep.mean(axis=1)

        def contrast(a):
            return float((np.percentile(a, 99) - np.median(a))
                         / (a.std() + 1e-12))
        for name, xn in (("N1", x1), ("N2", x2)):
            comp_rows.append({
                "window_id": d["window_id"], "dataset": d["dataset"],
                "class": d["class"], "strategy": name,
                "post_mean": round(float(xn.mean()), 4),
                "post_std": round(float(xn.std()), 4),
                "peak_contrast_pre": round(contrast(rep), 3),
                "peak_contrast_post": round(contrast(xn), 3),
                "freq_profile_corr_pre_post":
                    round(float(np.corrcoef(fprof,
                                            xn.mean(axis=0))[0, 1]), 4),
                "temporal_profile_corr_pre_post":
                    round(float(np.corrcoef(tprof,
                                            xn.mean(axis=1))[0, 1]), 4),
                "n1_window_floored": bool(floored) if name == "N1" else
                    None,
            })
            num_rows.append({
                "window_id": d["window_id"], "dataset": d["dataset"],
                "strategy": name,
                "post_min": round(float(xn.min()), 3),
                "post_p1": round(float(np.percentile(xn, 1)), 3),
                "post_p99": round(float(np.percentile(xn, 99)), 3),
                "post_max": round(float(xn.max()), 3),
                "finite": bool(np.isfinite(xn).all()),
            })
    pd.DataFrame(comp_rows).to_csv(out_dir / "normalization_comparison.csv",
                                   index=False)
    pd.DataFrame(num_rows).to_csv(out_dir / "normalization_numerics.csv",
                                  index=False)

    access = {
        "rule": "Fold-1 TRAIN only; guard fail-closed",
        "n_stats_sample": int(len(sample)),
        "n_qualitative": int(len(qual)),
        "stats_sample_step": STATS_SAMPLE_STEP,
        "window_ids_stats_sample": sample["window_id"].tolist(),
        "window_ids_qualitative": qual["window_id"].tolist(),
    }
    with open(out_dir / "part4b_signal_access_log.json", "w") as f:
        json.dump(access, f, indent=1)

    n2_summary = {ds: {"n_frames": v["n_frames"],
                       "n_floored_bins": v["n_floored_bins"]}
                  for ds, v in n2.items()}
    return {"n2_stats": n2, "n2_summary": n2_summary, "scalars": sc,
            "qual": qual, "sample": sample, "train_by_id": train_by_id}
