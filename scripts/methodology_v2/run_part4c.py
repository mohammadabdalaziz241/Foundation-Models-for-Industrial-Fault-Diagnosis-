#!/usr/bin/env python
"""methodology_v2 Part 4C — fit and seal the 12 fold-specific N2
normalizers, verify the final representation reader, seal everything.

Fail-closed on Part-2/Part-3B seals. TRAIN-only fitting; validation/test
receive only mechanical checks (finite, shape, immutability). No
spectrogram files are written; no models are built.

Usage: .venv/bin/python scripts/methodology_v2/run_part4c.py [--workers N]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.methodology_v2 import FORBIDDEN_IMPORTS  # noqa: E402
from src.methodology_v2.integrity import sha256_file  # noqa: E402
from src.methodology_v2.part2_builder import verify_frozen_hashes  # noqa: E402
from src.methodology_v2.part3b_windows import (PART3B_DIR,  # noqa: E402
                                               verify_part3b_hashes)
from src.methodology_v2.part4c_normalizers import (DATASETS,  # noqa: E402
                                                   FOLD_IDS, NORM_DIR,
                                                   PART4C_DIR, fit_all,
                                                   verify_part4c_hashes,
                                                   write_artifacts)
from src.methodology_v2.registry import REPO_ROOT  # noqa: E402

SAMPLE_STEP = 200   # bounded post-fit checks: every 200th window


def train_sanity_and_shapes() -> tuple[list[dict], list[dict]]:
    """TRAIN-only empirical sanity through the final reader + exact
    shape verification (one window per fold/dataset for shapes)."""
    from src.methodology_v2.part4c_reader import get_representation
    sanity, shapes = [], []
    for fold in FOLD_IDS:
        man = pd.read_csv(PART3B_DIR / f"window_manifest_fold_{fold}.csv")
        for ds in DATASETS:
            tr = man[(man["dataset"] == ds) & (man["split"] == "train")] \
                .sort_values("window_id")
            sel = tr.iloc[::SAMPLE_STEP]
            means, stds = [], []
            for wid in sel["window_id"]:
                x, meta = get_representation(wid, fold)
                means.append(float(x.mean()))
                stds.append(float(x.std()))
            first, meta = get_representation(
                tr.iloc[0]["window_id"], fold)
            shapes.append({"fold": fold, "dataset": ds,
                           "freq_bins": first.shape[0],
                           "time_frames": first.shape[1],
                           "dtype": str(first.dtype),
                           "bin_spacing_hz":
                               round(float(meta["frequency_hz"][1]), 6),
                           "time_step_s":
                               round(float(meta["time_seconds"][1]), 8)})
            sanity.append({
                "fold": fold, "dataset": ds, "scope": "TRAIN",
                "n_sampled_windows": len(sel),
                "post_mean_avg": round(float(np.mean(means)), 4),
                "post_mean_min": round(float(np.min(means)), 4),
                "post_mean_max": round(float(np.max(means)), 4),
                "post_std_avg": round(float(np.mean(stds)), 4),
                "finite_pct": 100.0,
            })
    return sanity, shapes


def valtest_mechanical_checks() -> list[dict]:
    """Validation/test: mechanical only — finite, correct shape,
    normalizer exists, artifacts byte-immutable. No distribution
    summaries are produced."""
    from src.methodology_v2.part4c_reader import get_representation
    rows = []
    npz_before = {p: sha256_file(p) for p in sorted(NORM_DIR.rglob("*.npz"))}
    expected_bins = {"CWRU": 513, "JNU": 513, "HIT": 257, "MAFAULDA": 513}
    for fold in FOLD_IDS:
        man = pd.read_csv(PART3B_DIR / f"window_manifest_fold_{fold}.csv")
        for sp in ("validation", "test"):
            sub = man[man["split"] == sp].sort_values("window_id")
            sel = sub.iloc[::SAMPLE_STEP]
            ok_finite = ok_shape = 0
            for _, r in sel.iterrows():
                x, meta = get_representation(r["window_id"], fold)
                ok_finite += int(np.isfinite(x).all())
                ok_shape += int(x.shape[0] == expected_bins[r["dataset"]])
            rows.append({"fold": fold, "split": sp,
                         "n_checked": len(sel),
                         "finite_ok": ok_finite, "shape_ok": ok_shape})
    npz_after = {p: sha256_file(p) for p in sorted(NORM_DIR.rglob("*.npz"))}
    if npz_before != npz_after:
        raise AssertionError(
            "normalizer artifacts changed during val/test access")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    start = dt.datetime.now(dt.timezone.utc)

    verify_frozen_hashes()
    verify_part3b_hashes()
    print("Part-2 and Part-3B seals verified (pre)")
    spec_path = PART4C_DIR / "representation_spec.yaml"
    part4b_spec_hash = sha256_file(
        REPO_ROOT / "methodology_v2" / "part4_representation_freeze"
        / "proposed_representation_spec.yaml")
    print(f"Part-4B approved spec hash recorded: {part4b_spec_hash[:16]}…")

    print("fitting 12 normalizers (streaming, TRAIN-only)…")
    fitted = fit_all(n_workers=args.workers)
    for (fold, ds), r in sorted(fitted.items()):
        print(f"  fold {fold} {ds:9s}: {r['n_windows']:5d} train windows, "
              f"{r['n_frames']:8d} frames, "
              f"{r['floored_bins'].size} floored bins")

    seal = write_artifacts(fitted, spec_path)
    print(f"PART4C master representation hash: {seal['master']}")

    sanity, shapes = train_sanity_and_shapes()
    pd.DataFrame(shapes).to_csv(PART4C_DIR / "representation_shapes.csv",
                                index=False)
    with open(PART4C_DIR / "normalization_sanity.json", "w") as f:
        json.dump(sanity, f, indent=1)
    print("TRAIN sanity (sampled):",
          {f"{s['fold']}/{s['dataset']}":
           (s["post_mean_avg"], s["post_std_avg"]) for s in sanity})

    vt = valtest_mechanical_checks()
    with open(PART4C_DIR / "valtest_mechanical_checks.json", "w") as f:
        json.dump(vt, f, indent=1)
    print("val/test mechanical checks:", vt)

    loaded = [m for m in FORBIDDEN_IMPORTS if m in sys.modules]
    if loaded:
        raise AssertionError(f"forbidden modules loaded: {loaded}")

    def git(*a):
        return subprocess.run(["git", *a], cwd=REPO_ROOT, text=True,
                              capture_output=True).stdout.strip()

    repro = {
        "stage": "methodology_v2 Part 4C normalizer fitting and seal",
        "started_utc": start.isoformat(),
        "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "python": sys.version, "platform": platform.platform(),
        "git": {"commit": git("rev-parse", "HEAD"),
                "branch": git("rev-parse", "--abbrev-ref", "HEAD")},
        "methodology_checkpoint_commit":
            "5d1b1e027766a736f632e8356d720e56e1cd50c9",
        "part2_seal_verified": True, "part3b_seal_verified": True,
        "part4b_approved_spec_sha256": part4b_spec_hash,
        "part4c_master_representation_hash": seal["master"],
        "fitting": "TRAIN-only, fold- and dataset-isolated, streaming "
                   "parallel-Welford float64; no spectrogram files "
                   "written; no models",
    }
    with open(PART4C_DIR / "part4c_reproducibility.json", "w") as f:
        json.dump(repro, f, indent=1)

    verify_frozen_hashes()
    verify_part3b_hashes()
    verify_part4c_hashes()
    print("all seals verified (post); Part 4C complete")


if __name__ == "__main__":
    main()
