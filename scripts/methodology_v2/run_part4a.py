#!/usr/bin/env python
"""methodology_v2 Part 4A — training-only STFT design study.

Fail-closed on Part-2/Part-3B seals; all raw values via the frozen lazy
reader from Fold-1 TRAIN windows only. Produces the numeric study tables
plus a bounded deterministic figure set. No full-dataset spectrograms,
no normalization fitting, no models, no training.
"""
from __future__ import annotations

import datetime as dt
import json
import platform
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.methodology_v2 import FORBIDDEN_IMPORTS  # noqa: E402
from src.methodology_v2.integrity import sha256_file  # noqa: E402
from src.methodology_v2.part2_builder import verify_frozen_hashes  # noqa: E402
from src.methodology_v2.part3b_windows import (PART3B_DIR,  # noqa: E402
                                               verify_part3b_hashes)
from src.methodology_v2.part3b_reader import read_window  # noqa: E402
from src.methodology_v2.part4a_repdesign import (PART4A_DIR,  # noqa: E402
                                                 apply_transform,
                                                 assert_fold1_train,
                                                 run_numeric_studies,
                                                 tf_map)
from src.methodology_v2.registry import REPO_ROOT  # noqa: E402

FIG_DIR = PART4A_DIR / "figures"
N_FFT_VIS, HOP_VIS = 1024, 256   # visualisation reference config

# deterministic per-dataset representative classes for figures
FIG_CLASSES = {
    "CWRU": ["IR007", "B014", "OR021@6"],
    "JNU": ["n", "ib", "ob", "tb"],
    "HIT": ["0", "1", "2"],
    "MAFAULDA": ["normal", "imbalance", "horizontal-misalignment",
                 "vertical-misalignment", "underhang/outer_race"],
}


def _title(w, n_fft, hop, transform) -> str:
    fs = int(w["native_sampling_rate_hz"])
    return (f"{w['dataset']} | {w['class']} | {w['window_id']}\n"
            f"fs={fs} Hz, rpm={w['rpm']}, n_fft={n_fft}, hop={hop}, "
            f"df={fs / n_fft:.1f} Hz, {transform}")


def _imshow(ax, rep, fs, n_fft, hop):
    """Visualisation image (colormap for HUMAN inspection only — the
    model-domain tensor stays numeric, no colormap/RGB/resize)."""
    ax.imshow(rep.T, origin="lower", aspect="auto",
              extent=[0, rep.shape[0] * hop / fs, 0, fs / 2 / 1000],
              interpolation="none", cmap="magma")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("freq [kHz]")


def class_figures(dev, signals):
    made = []
    for ds, classes in FIG_CLASSES.items():
        for cls in classes:
            cand = dev[(dev["dataset"] == ds) & (dev["class"] == cls)]
            if cand.empty:
                continue
            w = cand.sort_values("window_id").iloc[0]
            x = signals[w["window_id"]]
            fs = int(w["native_sampling_rate_hz"])
            z = tf_map(x, N_FFT_VIS, HOP_VIS)
            mag = apply_transform(z, "magnitude")
            lg = apply_transform(z, "log1p")
            fig, axes = plt.subplots(1, 4, figsize=(18, 3.6),
                                     constrained_layout=True)
            t = np.arange(x.size) / fs
            axes[0].plot(t, x, lw=0.3)
            axes[0].set_title("raw waveform")
            axes[0].set_xlabel("time [s]")
            _imshow(axes[1], mag, fs, N_FFT_VIS, HOP_VIS)
            axes[1].set_title("raw |TF| magnitude")
            _imshow(axes[2], lg, fs, N_FFT_VIS, HOP_VIS)
            axes[2].set_title("log1p(|TF|) + shaft-freq harmonics")
            try:
                frot = float(w["rpm"]) / 60.0
                for h in range(1, 6):
                    axes[2].axhline(h * frot / 1000, color="cyan",
                                    lw=0.5, alpha=0.6)
            except (TypeError, ValueError):
                pass
            axes[3].axis("off")
            axes[3].text(0.02, 0.5,
                         "encoder-input tensor (hypothetical):\n"
                         f"float32, shape ({lg.shape[1]}, {lg.shape[0]})\n"
                         "= (freq bins, time frames)\n"
                         "native resolution — NO resize,\n"
                         "NO RGB, NO colormap, NO quantization",
                         fontsize=10, va="center", family="monospace")
            fig.suptitle(_title(w, N_FFT_VIS, HOP_VIS, "magnitude/log1p"),
                         fontsize=9)
            name = f"class_{ds}_{cls}".replace("/", "-").replace("@", "")
            fig.savefig(FIG_DIR / f"{name}.png", dpi=110)
            plt.close(fig)
            made.append(name)
    return made


def multires_figures(dev, signals):
    made = []
    for ds in FIG_CLASSES:
        w = dev[dev["dataset"] == ds].sort_values("window_id").iloc[0]
        x = signals[w["window_id"]]
        fs = int(w["native_sampling_rate_hz"])
        fig, axes = plt.subplots(1, 2, figsize=(12, 4),
                                 constrained_layout=True)
        for ax, n_fft in zip(axes, (512, 4096)):
            rep = apply_transform(tf_map(x, n_fft, n_fft // 4), "log1p")
            _imshow(ax, rep, fs, n_fft, n_fft // 4)
            ax.set_title(f"n_fft={n_fft} "
                         f"(df={fs / n_fft:.1f} Hz, "
                         f"frame={1000 * n_fft / fs:.1f} ms)")
        fig.suptitle(f"{w['dataset']} | {w['class']} | {w['window_id']}\n"
                     f"fs={fs} Hz, rpm={w['rpm']} — short (512) vs long "
                     f"(4096) analysis frames, log1p", fontsize=9)
        name = f"multires_{ds}"
        fig.savefig(FIG_DIR / f"{name}.png", dpi=110)
        plt.close(fig)
        made.append(name)
    return made


def hit_boundary_figures(train_by_id, audit):
    made = []
    picks = audit.sort_values("window_id").iloc[[0, len(audit) // 2]]
    for i, (_, a) in enumerate(picks.iterrows()):
        w = train_by_id.loc[a["window_id"]]
        assert_fold1_train(w)
        x = read_window(w)
        fs = int(w["native_sampling_rate_hz"])
        b = int(a["boundary_sample_in_window"])
        lg = apply_transform(tf_map(x, N_FFT_VIS, HOP_VIS), "log1p")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4),
                                 constrained_layout=True)
        lo, hi = max(0, b - 400), min(x.size, b + 400)
        axes[0].plot(np.arange(lo, hi), x[lo:hi], lw=0.6)
        axes[0].axvline(b, color="red", lw=1,
                        label=f"fragment joint (jump ratio "
                              f"{a['jump_ratio']:.2f})")
        axes[0].legend(fontsize=8)
        axes[0].set_title("waveform around fragment joint")
        _imshow(axes[1], lg, fs, N_FFT_VIS, HOP_VIS)
        axes[1].axvline(b / fs, color="red", lw=0.8)
        axes[1].set_title("log1p TF map, joint time marked")
        fig.suptitle(f"HIT fragment-joint audit | {a['window_id']} | "
                     f"boundary@{b} | max frame-energy z="
                     f"{a['max_frame_energy_zscore']}", fontsize=9)
        name = f"hit_boundary_{i}"
        fig.savefig(FIG_DIR / f"{name}.png", dpi=110)
        plt.close(fig)
        made.append(name)
    return made


def negative_control_figure(dev, signals):
    """NEGATIVE CONTROL ONLY: shows what a crude 128x128 block-average
    'image resize' would destroy. Not a pipeline component."""
    w = dev[dev["dataset"] == "CWRU"].sort_values("window_id").iloc[0]
    x = signals[w["window_id"]]
    fs = int(w["native_sampling_rate_hz"])
    rep = apply_transform(tf_map(x, N_FFT_VIS, HOP_VIS), "log1p")
    fb = rep.shape[1] // 128
    tb = rep.shape[0] // 128
    pooled = rep[:128 * tb, :128 * fb].reshape(128, tb, 128, fb) \
        .mean(axis=(1, 3))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4),
                             constrained_layout=True)
    _imshow(axes[0], rep, fs, N_FFT_VIS, HOP_VIS)
    axes[0].set_title(f"native resolution {rep.shape[1]}x{rep.shape[0]}")
    axes[1].imshow(pooled.T, origin="lower", aspect="auto",
                   interpolation="none", cmap="magma")
    axes[1].set_title("NEGATIVE CONTROL: 128x128 block-mean 'resize' — "
                      "NOT used anywhere")
    fig.suptitle(f"resize negative control | {w['window_id']}", fontsize=9)
    fig.savefig(FIG_DIR / "negative_control_resize.png", dpi=110)
    plt.close(fig)
    return ["negative_control_resize"]


def main() -> None:
    start = dt.datetime.now(dt.timezone.utc)
    verify_frozen_hashes()
    verify_part3b_hashes()
    print("Part-2 and Part-3B seals verified (pre)")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    res = run_numeric_studies(PART4A_DIR, read_window)
    dev, signals = res["dev"], res["signals"]
    print(f"development windows: {len(dev)} "
          f"({dev.groupby('dataset').size().to_dict()})")
    audit = res["boundary_audit"]
    print(f"HIT boundary audit: {len(audit)} joints in "
          f"{audit['window_id'].nunique()} train windows | "
          f"max frame-energy z={audit['max_frame_energy_zscore'].max()} | "
          f"max jump ratio={audit['jump_ratio'].max():.2f}")

    # per-dataset scale summary for the normalization study (dev set only)
    scale = {}
    for ds in FIG_CLASSES:
        vals = [apply_transform(tf_map(signals[w], N_FFT_VIS, HOP_VIS),
                                "log1p")
                for w in dev.loc[dev["dataset"] == ds, "window_id"]]
        allv = np.concatenate([v.ravel() for v in vals])
        scale[ds] = {"log1p_mean": round(float(allv.mean()), 4),
                     "log1p_std": round(float(allv.std()), 4),
                     "log1p_p99": round(float(np.percentile(allv, 99)), 4)}
    print("per-dataset log1p scale (dev):", scale)

    figs = (class_figures(dev, signals) + multires_figures(dev, signals)
            + hit_boundary_figures(res["train_by_id"], audit)
            + negative_control_figure(dev, signals))
    print(f"figures written: {len(figs)}")

    loaded = [m for m in FORBIDDEN_IMPORTS if m in sys.modules]
    if loaded:
        raise AssertionError(f"forbidden modules loaded: {loaded}")

    def git(*a):
        return subprocess.run(["git", *a], cwd=REPO_ROOT, text=True,
                              capture_output=True).stdout.strip()

    repro = {
        "stage": "methodology_v2 Part 4A STFT design study",
        "started_utc": start.isoformat(),
        "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "python": sys.version, "platform": platform.platform(),
        "git": {"commit": git("rev-parse", "HEAD"),
                "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
                "n_untracked_or_modified":
                    len(git("status", "--short").splitlines())},
        "part2_seal_verified": True, "part3b_seal_verified": True,
        "part3b_window_hashes_sha256":
            sha256_file(PART3B_DIR / "window_hashes.csv"),
        "data_access": "Fold-1 TRAIN only (see "
                       "part4a_signal_access_log.json)",
        "per_dataset_log1p_scale_dev": scale,
        "n_figures": len(figs),
    }
    with open(PART4A_DIR / "part4a_reproducibility.json", "w") as f:
        json.dump(repro, f, indent=1)

    verify_frozen_hashes()
    verify_part3b_hashes()
    print("seals re-verified (post); no forbidden code executed")


if __name__ == "__main__":
    main()
