#!/usr/bin/env python
"""methodology_v2 Part 1 audit runner.

Builds, from raw data only and strictly read-only:
  - methodology_v2/part1_audit/recording_manifest.csv
  - methodology_v2/part1_audit/dataset_census.csv / dataset_census.md
  - methodology_v2/part1_audit/raw_file_hashes.csv
  - methodology_v2/part1_audit/integrity_details.json
  - methodology_v2/part1_audit/reproducibility.json

It never creates windows, splits, resampled signals or training artefacts.

Usage:
    .venv/bin/python scripts/methodology_v2/run_part1_audit.py \
        --datasets CWRU JNU HIT MAFAULDA
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.methodology_v2 import FORBIDDEN_IMPORTS  # noqa: E402
from src.methodology_v2.registry import (DATASETS, OUTPUT_DIR,  # noqa: E402
                                         REPO_ROOT, CWRU, JNU, HIT, MAFAULDA)
from src.methodology_v2.schema import (MANIFEST_COLUMNS,  # noqa: E402
                                       validate_manifest)
from src.methodology_v2.census import build_census, census_markdown  # noqa: E402
from src.methodology_v2.integrity import sha256_file  # noqa: E402


def _assert_no_training_modules() -> None:
    loaded = [m for m in FORBIDDEN_IMPORTS if m in sys.modules]
    if loaded:
        raise AssertionError(
            f"training/windowing modules loaded during audit: {loaded}")


def _git_state() -> dict:
    def run(*args):
        return subprocess.run(["git", *args], cwd=REPO_ROOT, text=True,
                              capture_output=True).stdout.strip()
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "n_untracked_or_modified": len(run("status", "--short").splitlines()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+",
                        default=["CWRU", "JNU", "HIT", "MAFAULDA"],
                        choices=list(DATASETS))
    parser.add_argument("--mafaulda-workers", type=int, default=4)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    start = dt.datetime.now(dt.timezone.utc)

    all_rows: list[dict] = []
    all_integrity: dict[str, list] = {}
    extra_reports: dict[str, dict] = {}

    if "CWRU" in args.datasets:
        from src.methodology_v2.audit_cwru import audit_cwru
        rows, integ = audit_cwru()
        all_rows += rows
        all_integrity["CWRU"] = integ
        print(f"CWRU: {len(rows)} recordings, {len(integ)} files audited")

    if "JNU" in args.datasets:
        from src.methodology_v2.audit_jnu import audit_jnu
        rows, integ = audit_jnu()
        all_rows += rows
        all_integrity["JNU"] = integ
        print(f"JNU: {len(rows)} recordings")

    if "HIT" in args.datasets:
        from src.methodology_v2.audit_hit import audit_hit_full
        rows, integ, gh_report = audit_hit_full()
        all_rows += rows
        all_integrity["HIT"] = integ
        extra_reports["HIT_github_release"] = gh_report
        print(f"HIT: {len(rows)} recordings "
              f"({gh_report['provenance']})")

    if "MAFAULDA" in args.datasets:
        from src.methodology_v2.audit_mafaulda import audit_mafaulda
        rows, integ = audit_mafaulda(n_workers=args.mafaulda_workers)
        all_rows += rows
        all_integrity["MAFAULDA"] = integ
        print(f"MAFAULDA: {len(rows)} recordings")

    _assert_no_training_modules()

    manifest = pd.DataFrame(all_rows, columns=MANIFEST_COLUMNS)
    valid_labels = {
        "CWRU": CWRU["valid_labels"],
        "JNU": JNU["valid_labels"],
        "HIT": HIT["valid_labels"],
        "MAFAULDA": {"normal", "imbalance", "horizontal-misalignment",
                     "vertical-misalignment",
                     "underhang/ball_fault", "underhang/cage_fault",
                     "underhang/outer_race",
                     "overhang/ball_fault", "overhang/cage_fault",
                     "overhang/outer_race"},
    }
    validate_manifest(manifest,
                      {k: v for k, v in valid_labels.items()
                       if k in set(manifest["dataset"])})
    manifest.to_csv(OUTPUT_DIR / "recording_manifest.csv", index=False)
    print(f"manifest: {len(manifest)} rows -> recording_manifest.csv")

    # ---- raw file hashes -------------------------------------------------
    hash_rows = []
    for ds, integ in all_integrity.items():
        for rec in integ:
            if rec.get("sha256") is None:
                continue
            f = rec["file"]
            base = {"CWRU": REPO_ROOT / "data",
                    "JNU": JNU["paths"]["root"],
                    "MAFAULDA": MAFAULDA["paths"]["root"]}.get(ds)
            p = (base / f) if base else None
            st = p.stat() if (p and p.exists()) else None
            hash_rows.append({
                "dataset": ds, "file": f, "sha256": rec["sha256"],
                "payload_sha256": rec.get("payload_sha256"),
                "bytes": st.st_size if st else None,
                "mtime_utc": (dt.datetime.fromtimestamp(
                    st.st_mtime, dt.timezone.utc).isoformat()
                    if st else None),
            })
    if "HIT" in all_integrity:
        gh = extra_reports["HIT_github_release"]
        for sname, s in gh["sessions"].items():
            p = REPO_ROOT / s["file"]
            hash_rows.append({
                "dataset": "HIT", "file": s["file"], "sha256": s["sha256"],
                "payload_sha256": None, "bytes": p.stat().st_size,
                "mtime_utc": dt.datetime.fromtimestamp(
                    p.stat().st_mtime, dt.timezone.utc).isoformat(),
            })
        for stem, f in gh["files"].items():
            p = HIT["paths"]["github_release"] / f"{stem}.mat"
            hash_rows.append({
                "dataset": "HIT", "file": f"github:{stem}.mat",
                "sha256": f["sha256"], "payload_sha256": None,
                "bytes": p.stat().st_size,
                "mtime_utc": dt.datetime.fromtimestamp(
                    p.stat().st_mtime, dt.timezone.utc).isoformat(),
            })
    pd.DataFrame(hash_rows).to_csv(OUTPUT_DIR / "raw_file_hashes.csv",
                                   index=False)
    print(f"hashes: {len(hash_rows)} files -> raw_file_hashes.csv")

    # ---- census ----------------------------------------------------------
    n_raw = {ds: len(v) for ds, v in all_integrity.items()}
    extra = {
        "CWRU": {
            "local_path": "data/raw (12k DE) + data/raw_cwru_48k (48k DE)",
            "file_format": "MATLAB .mat (v5)",
            "licence": "publicly distributed by CWRU Bearing Data Center; "
                       "no explicit licence text on site",
            "bearing_identity": "fault specimen inferable from fault "
                                "spec (type+size); per-specimen serials not "
                                "published; OR clock positions may share a "
                                "specimen (not documented)",
            "run_identity": "one .mat per acquisition; consecutive canonical "
                            "file numbers across the 4 loads of a specimen",
            "multi_file_note": "yes: same specimen recorded at 4 loads and "
                               "at both 12 kHz and 48 kHz",
            "official_split": "none",
        },
        "JNU": {
            "local_path": "data/raw_jnu/JNU-Bearing-Dataset "
                          f"(git {JNU['clone_commit'][:12]})",
            "file_format": "single-column CSV",
            "licence": "no licence file in repository",
            "bearing_identity": "one seeded specimen per condition "
                                "(inferred), reused across speeds",
            "run_identity": "one CSV per (condition, speed)",
            "multi_file_note": "same specimen across 3 speeds",
            "official_split": "none",
        },
        "HIT": {
            "local_path": "data/raw_hit/HIT-dataset (github, "
                          f"{HIT['clone_commit'][:12]}) + "
                          "data/raw_hit/gdrive_full/HIT-dataset",
            "file_format": ".npy (full) / .mat shards (github)",
            "licence": "paper CC BY 4.0; dataset licence not stated",
            "bearing_identity": "session -> physical bearing documented "
                                "(data1/2 healthy, data3/4 inner x2 "
                                "specimens, data5 outer x1)",
            "run_identity": "session (assembly) x speed-group; speed value "
                            "recorded per series in column 7",
            "multi_file_note": "one .npy per session holds all speed groups",
            "official_split": "YES (github xtrain/xtest) — but window-level "
                              "random; see integrity report",
            "rpm_note": "LP 1000-5000, HP 1200-6000 r/min, 28 planned "
                        "speed groups (paper Table V)",
        },
        "MAFAULDA": {
            "local_path": "data/raw_mafaulda/full (extracted from full.zip)",
            "file_format": "8-column CSV",
            "licence": "publicly distributed by UFRJ/SMT; no explicit "
                       "licence text on site",
            "bearing_identity": "3 defective bearings (ball/cage/outer), "
                                "each reused at underhang AND overhang "
                                "positions; single rig",
            "run_identity": "one 5 s CSV per (configuration, speed)",
            "multi_file_note": "same fault configuration recorded at ~49 "
                               "speeds",
            "official_split": "none",
        },
    }
    census = build_census(manifest, n_raw, extra)
    census.to_csv(OUTPUT_DIR / "dataset_census.csv", index=False)
    (OUTPUT_DIR / "dataset_census.md").write_text(census_markdown(census))
    print("census -> dataset_census.csv / dataset_census.md")

    # ---- integrity details ----------------------------------------------
    with open(OUTPUT_DIR / "integrity_details.json", "w") as f:
        json.dump({"per_file": all_integrity, "extra": extra_reports},
                  f, indent=1, default=str)

    # ---- reproducibility -------------------------------------------------
    import numpy, scipy  # noqa
    repro = {
        "stage": "methodology_v2 Part 1 dataset audit",
        "started_utc": start.isoformat(),
        "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "git": _git_state(),
        "dataset_sources": {
            "CWRU": {"url": CWRU["source_url"],
                     "local_version": "pre-existing repo copies; rates per "
                                      "docs/cwru_legacy_rate_impact_note.md"},
            "JNU": {"url": JNU["source_url"],
                    "clone_commit": JNU["clone_commit"]},
            "HIT": {"url": HIT["source_url"],
                    "clone_commit": HIT["clone_commit"],
                    "gdrive": "https://drive.google.com/drive/folders/"
                              "1Km1Go4ilB_bI033SBJ7eJ0uCzbqEqbgt"},
            "MAFAULDA": {"url": MAFAULDA["source_url"],
                         "archive": str(MAFAULDA["archive"]),
                         "archive_sha256": (
                             sha256_file(MAFAULDA["archive"])
                             if MAFAULDA["archive"].exists() and
                             "MAFAULDA" in args.datasets else None)},
        },
    }
    with open(OUTPUT_DIR / "reproducibility.json", "w") as f:
        json.dump(repro, f, indent=1)
    print("reproducibility.json written")
    _assert_no_training_modules()
    print("AUDIT COMPLETE — no training/windowing modules touched")


if __name__ == "__main__":
    main()
