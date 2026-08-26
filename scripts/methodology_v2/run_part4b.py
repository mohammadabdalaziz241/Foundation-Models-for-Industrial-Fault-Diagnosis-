#!/usr/bin/env python
"""methodology_v2 Part 4B — representation verification + N1/N2 study.

Fail-closed on Part-2/Part-3B seals. Fold-1 TRAIN content only. No
full-dataset STFT generation, no final normalizer fitting, no models.

Usage: .venv/bin/python scripts/methodology_v2/run_part4b.py
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.methodology_v2 import FORBIDDEN_IMPORTS  # noqa: E402
from src.methodology_v2.integrity import sha256_file  # noqa: E402
from src.methodology_v2.part2_builder import verify_frozen_hashes  # noqa: E402
from src.methodology_v2.part3b_windows import (PART3B_DIR,  # noqa: E402
                                               verify_part3b_hashes)
from src.methodology_v2.part3b_reader import read_window  # noqa: E402
from src.methodology_v2.part4a_repdesign import assert_fold1_train  # noqa: E402
from src.methodology_v2.part4b_freeze import (PART4B_DIR,  # noqa: E402
                                              n1_normalize, n2_normalize,
                                              rep_of, run_study,
                                              verification_table)
from src.methodology_v2.registry import REPO_ROOT  # noqa: E402

FIG_DIR = PART4B_DIR / "figures"


def comparison_figures(res):
    """One TRAIN window per dataset: raw log1p vs N1 vs N2 (colormaps are
    for human inspection only; the model-domain tensor stays numeric)."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    made = []
    for ds in ("CWRU", "JNU", "HIT", "MAFAULDA"):
        d = res["qual"][res["qual"]["dataset"] == ds] \
            .sort_values("window_id").iloc[0]
        w = res["train_by_id"].loc[d["window_id"]]
        assert_fold1_train(w)
        rep = rep_of(read_window(w), ds)
        x1, _ = n1_normalize(rep)
        x2 = n2_normalize(rep, res["n2_stats"][ds])
        fig, axes = plt.subplots(1, 3, figsize=(15, 3.6),
                                 constrained_layout=True)
        for ax, (name, arr) in zip(axes, [("log1p (pre-norm)", rep),
                                          ("N1 per-window", x1),
                                          ("N2 per-dataset-per-bin", x2)]):
            im = ax.imshow(arr.T, origin="lower", aspect="auto",
                           interpolation="none", cmap="magma")
            ax.set_title(name, fontsize=9)
            fig.colorbar(im, ax=ax, shrink=0.8)
        fig.suptitle(f"{ds} | {d['class']} | {d['window_id']} — "
                     "normalization comparison (Fold-1 TRAIN)", fontsize=9)
        name = f"norm_compare_{ds}"
        fig.savefig(FIG_DIR / f"{name}.png", dpi=110)
        plt.close(fig)
        made.append(name)
    return made


def main() -> None:
    start = dt.datetime.now(dt.timezone.utc)
    verify_frozen_hashes()
    verify_part3b_hashes()
    print("Part-2 and Part-3B seals verified (pre)")

    print(verification_table().to_string(index=False))
    res = run_study(PART4B_DIR, read_window)
    print(f"stats sample: {len(res['sample'])} windows | qualitative: "
          f"{len(res['qual'])}")
    print("N2 dev-sample stats:", res["n2_summary"])

    figs = comparison_figures(res)
    print(f"figures: {len(figs)}")

    loaded = [m for m in FORBIDDEN_IMPORTS if m in sys.modules]
    if loaded:
        raise AssertionError(f"forbidden modules loaded: {loaded}")

    def git(*a):
        return subprocess.run(["git", *a], cwd=REPO_ROOT, text=True,
                              capture_output=True).stdout.strip()

    repro = {
        "stage": "methodology_v2 Part 4B representation freeze study",
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
        "data_access": "Fold-1 TRAIN only "
                       "(part4b_signal_access_log.json)",
        "note": "N2 statistics here are DEV-SAMPLE estimates for the "
                "comparison only; final fold-specific normalizers are "
                "NOT fitted (requires approval)",
    }
    with open(PART4B_DIR / "part4b_reproducibility.json", "w") as f:
        json.dump(repro, f, indent=1)

    verify_frozen_hashes()
    verify_part3b_hashes()
    print("seals re-verified (post); no forbidden work executed")


if __name__ == "__main__":
    main()
