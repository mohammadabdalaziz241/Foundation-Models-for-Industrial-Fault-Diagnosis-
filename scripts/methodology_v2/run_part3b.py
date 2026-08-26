#!/usr/bin/env python
"""methodology_v2 Part 3B — frozen raw-window manifest generation.

Verifies the Part-2 seal (fail closed) before and after; builds the three
window manifests + JNU guard artifact + HIT stream manifest; runs the
sampled signal-integrity audit through the lazy reader; seals everything
with a Part-3B master hash. No resampling, no STFT, no normalization, no
training.

Usage: .venv/bin/python scripts/methodology_v2/run_part3b.py
       [--integrity-step N] [--out-dir DIR]
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
from src.methodology_v2.part2_builder import PART2_DIR  # noqa: E402
from src.methodology_v2.part3b_windows import (build_all,  # noqa: E402
                                               verify_part3b_hashes)
from src.methodology_v2.part3b_reader import read_window  # noqa: E402
from src.methodology_v2.registry import REPO_ROOT  # noqa: E402


def sampled_integrity(manifests, step: int) -> dict:
    """Every `step`-th window per fold, read through the lazy reader:
    exact length, finite values, constant-window detection. Test-signal
    values are checked mechanically only (length/finite/constant) and
    never used to tune any parameter."""
    out = {"step": step, "checked": 0, "bad_length": 0, "nonfinite": 0,
           "constant_windows": []}
    for fold, df in manifests.items():
        sel = df.sort_values(["dataset", "recording_id",
                              "start_sample"]).iloc[::step]
        for _, row in sel.iterrows():
            x = read_window(row)
            out["checked"] += 1
            expect = int(round(row["window_duration_seconds"]
                               * row["native_sampling_rate_hz"]))
            if x.size != expect:
                out["bad_length"] += 1
            if not np.isfinite(x).all():
                out["nonfinite"] += 1
            if x.size and float(np.max(x)) == float(np.min(x)):
                out["constant_windows"].append(str(row["window_id"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--integrity-step", type=int, default=50)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    start = dt.datetime.now(dt.timezone.utc)

    verify_frozen_hashes()
    print("Part-2 seal verified (pre)")

    res = build_all(args.out_dir)
    out_dir = res["out_dir"]
    for fold, df in res["manifests"].items():
        by = df.groupby("split").size().to_dict()
        print(f"fold {fold}: {len(df)} windows {by}")
    print("estimate mismatches:",
          [c for c in res["stats"]["estimate_comparison"]
           if c["difference"] != 0] or "none")

    integ = sampled_integrity(res["manifests"], args.integrity_step)
    print(f"sampled integrity: {integ['checked']} windows checked, "
          f"bad_length={integ['bad_length']}, nonfinite={integ['nonfinite']}, "
          f"constant={len(integ['constant_windows'])}")
    if integ["bad_length"] or integ["nonfinite"]:
        raise AssertionError("signal integrity failure — see report")
    stats_path = out_dir / "window_statistics.json"
    stats = json.load(open(stats_path))
    stats["sampled_signal_integrity"] = integ
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=1, sort_keys=True)

    loaded = [m for m in FORBIDDEN_IMPORTS if m in sys.modules]
    if loaded:
        raise AssertionError(f"forbidden modules loaded: {loaded}")

    def git(*a):
        return subprocess.run(["git", *a], cwd=REPO_ROOT, text=True,
                              capture_output=True).stdout.strip()

    repro = {
        "stage": "methodology_v2 Part 3B frozen window extraction",
        "started_utc": start.isoformat(),
        "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "python": sys.version, "platform": platform.platform(),
        "git": {"commit": git("rev-parse", "HEAD"),
                "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
                "n_untracked_or_modified":
                    len(git("status", "--short").splitlines())},
        "part2_master_hash_verified": True,
        "part2_split_hashes_sha256":
            sha256_file(PART2_DIR / "split_hashes.csv"),
        "part3a_recommendations_sha256": sha256_file(
            REPO_ROOT / "methodology_v2" / "part3_input_design"
            / "part3a_recommendations.yaml"),
        "part3b_master_hash": res["master_hash"],
        "frozen_overrides": "native sampling rates kept (25 kHz common "
                            "rate rejected by approved decision)",
    }
    with open(out_dir / "part3b_reproducibility.json", "w") as f:
        json.dump(repro, f, indent=1)

    verify_frozen_hashes()
    verify_part3b_hashes(out_dir)
    print(f"Part-2 seal re-verified; Part-3B master hash "
          f"{res['master_hash']}")


if __name__ == "__main__":
    main()
