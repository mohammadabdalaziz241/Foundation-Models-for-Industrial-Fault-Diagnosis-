#!/usr/bin/env python
"""methodology_v2 Part 3A — input design study (audit only).

Verifies the Part-2 seal first (fail closed), then writes the metadata
study tables and the fold-1 TRAIN-only diagnostics. No resampling, no
windows, no STFT, no training.

Usage: .venv/bin/python scripts/methodology_v2/run_part3a.py
       [--skip-diagnostics] [--out-dir DIR]
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
from src.methodology_v2.part2_builder import (PART2_DIR,  # noqa: E402
                                              verify_frozen_hashes)
from src.methodology_v2.part3a_study import (PART3A_DIR,  # noqa: E402
                                             write_study_tables)
from src.methodology_v2.registry import REPO_ROOT  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-diagnostics", action="store_true")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    start = dt.datetime.now(dt.timezone.utc)

    verify_frozen_hashes()  # fail closed if Part-2 artefacts changed
    print("Part-2 seal verified")

    res = write_study_tables(args.out_dir)
    out_dir = res["out_dir"]
    print(f"study tables -> {out_dir}")

    if not args.skip_diagnostics:
        from src.methodology_v2.part3a_diagnostics import run_diagnostics
        diag = run_diagnostics(out_dir)
        for ds in ("CWRU", "JNU", "HIT", "MAFAULDA"):
            print(f"  {ds}: energy fractions {diag[ds]['energy_fraction']}")

    loaded = [m for m in FORBIDDEN_IMPORTS if m in sys.modules]
    if loaded:
        raise AssertionError(f"forbidden modules loaded: {loaded}")

    def git(*a):
        return subprocess.run(["git", *a], cwd=REPO_ROOT, text=True,
                              capture_output=True).stdout.strip()

    repro = {
        "stage": "methodology_v2 Part 3A input design study",
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
        "sealed_data_rule": "raw TEST signal content never read; "
                            "diagnostics restricted to fold-1 TRAIN",
    }
    with open(out_dir / "part3a_reproducibility.json", "w") as f:
        json.dump(repro, f, indent=1)
    verify_frozen_hashes()  # re-verify after all writes
    print("Part-2 seal re-verified after study; no forbidden code touched")


if __name__ == "__main__":
    main()
