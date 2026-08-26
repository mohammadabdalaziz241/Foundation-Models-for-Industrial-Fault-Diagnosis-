#!/usr/bin/env python
"""methodology_v2 Part 2 — generate and seal the three frozen global folds.

Identity/region assignment only: no windows, no preprocessing, no training.

Usage:
    .venv/bin/python scripts/methodology_v2/run_part2_splits.py
    (optionally --out-dir DIR for a determinism check into a scratch dir)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.methodology_v2 import FORBIDDEN_IMPORTS  # noqa: E402
from src.methodology_v2.integrity import sha256_file  # noqa: E402
from src.methodology_v2.registry import OUTPUT_DIR as PART1_DIR  # noqa: E402
from src.methodology_v2.registry import REPO_ROOT  # noqa: E402
from src.methodology_v2.part2_builder import (PART2_DIR,  # noqa: E402
                                              verify_frozen_hashes,
                                              write_outputs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=None,
                    help="override output directory (determinism checks)")
    args = ap.parse_args()
    start = dt.datetime.now(dt.timezone.utc)

    res = write_outputs(args.out_dir)
    out_dir = res["out_dir"]

    loaded = [mod for mod in FORBIDDEN_IMPORTS if mod in sys.modules]
    if loaded:
        raise AssertionError(f"training/windowing modules loaded: {loaded}")

    def git(*a):
        return subprocess.run(["git", *a], cwd=REPO_ROOT, text=True,
                              capture_output=True).stdout.strip()

    repro = {
        "stage": "methodology_v2 Part 2 frozen split protocol",
        "started_utc": start.isoformat(),
        "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "python": sys.version,
        "platform": platform.platform(),
        "git": {"commit": git("rev-parse", "HEAD"),
                "branch": git("rev-parse", "--abbrev-ref", "HEAD")},
        "part1_artifact_hashes": {
            p.name: sha256_file(p) for p in [
                PART1_DIR / "recording_manifest.csv",
                PART1_DIR / "CWRU_GROUPING_RECHECK.md",
                PART1_DIR / "cwru_grouping_recheck_table.csv",
                PART1_DIR / "PART1_DATASET_AUDIT_REPORT.md",
            ]},
        "seeds_used": res["seeds_used"],
        "n_rejected_seeds": len(res["rejections"]),
        "master_protocol_hash": res["master_hash"],
    }
    with open(out_dir / "split_reproducibility.json", "w") as f:
        json.dump(repro, f, indent=1)

    verify_frozen_hashes(out_dir)
    print(f"folds written to {out_dir}")
    print(f"seeds used: {res['seeds_used']}")
    print(f"rejected seeds: {len(res['rejections'])}")
    print(f"master protocol hash: {res['master_hash']}")
    print("seal verified; no training/windowing modules touched")


if __name__ == "__main__":
    main()
