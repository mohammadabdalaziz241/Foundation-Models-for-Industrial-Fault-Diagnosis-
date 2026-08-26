#!/usr/bin/env python
"""methodology_v2 Part 5A — architecture/novelty analysis artifacts.

Design study only: verifies all upstream seals (fail closed), writes the
deterministic analysis tables. No model layers, no training.
"""
from __future__ import annotations

import datetime as dt
import json
import platform
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.methodology_v2 import FORBIDDEN_IMPORTS  # noqa: E402
from src.methodology_v2.part2_builder import verify_frozen_hashes  # noqa: E402
from src.methodology_v2.part3b_windows import verify_part3b_hashes  # noqa: E402
from src.methodology_v2.part4c_normalizers import (  # noqa: E402
    verify_part4c_hashes)
from src.methodology_v2.part5a_analysis import (PART5A_DIR,  # noqa: E402
                                                write_all)
from src.methodology_v2.registry import REPO_ROOT  # noqa: E402


def main() -> None:
    start = dt.datetime.now(dt.timezone.utc)
    verify_frozen_hashes(); verify_part3b_hashes(); verify_part4c_hashes()
    print("Part-2, Part-3B, Part-4C seals verified (pre)")

    res = write_all()
    print(res["patch"][res["patch"].patch_freq_bins.eq(16)
                       & res["patch"].patch_time_frames.eq(8)]
          .to_string(index=False))
    print(res["budget"].to_string(index=False))

    loaded = [m for m in FORBIDDEN_IMPORTS if m in sys.modules]
    assert not loaded, f"forbidden modules loaded: {loaded}"
    assert "torch" not in sys.modules and "mamba_ssm" not in sys.modules

    def git(*a):
        return subprocess.run(["git", *a], cwd=REPO_ROOT, text=True,
                              capture_output=True).stdout.strip()
    repro = {
        "stage": "methodology_v2 Part 5A architecture and novelty audit",
        "started_utc": start.isoformat(),
        "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "python": sys.version, "platform": platform.platform(),
        "git": {"commit": git("rev-parse", "HEAD"),
                "branch": git("rev-parse", "--abbrev-ref", "HEAD")},
        "seals_verified": ["part2", "part3b", "part4c"],
        "literature_search_date": "2026-08-12",
        "note": "design study only; no model layers imported",
    }
    with open(PART5A_DIR / "part5a_reproducibility.json", "w") as f:
        json.dump(repro, f, indent=1)
    verify_frozen_hashes(); verify_part3b_hashes(); verify_part4c_hashes()
    print("seals re-verified (post); Part 5A artifacts complete")


if __name__ == "__main__":
    main()
