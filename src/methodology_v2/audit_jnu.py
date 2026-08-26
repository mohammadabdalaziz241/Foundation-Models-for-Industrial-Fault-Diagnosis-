"""JNU audit: 12 long single-channel recordings, one per (condition, speed).

Filename convention (repo root): <cond><speed>_2.csv for faults and
n<speed>_3_2.csv for healthy, where cond in {ib, ob, tb} and speed in
{600, 800, 1000} rpm. The trailing suffixes are not documented by the
source repository; they are recorded verbatim and marked as such.

The healthy files are exactly 3x the length of the fault files (1,501,500
vs 500,500 samples). Whether they are single continuous acquisitions or
concatenations of three shorter ones is NOT documented; a boundary
discontinuity probe at the 1/3 and 2/3 points is run as evidence and the
result is reported, not assumed.
"""
from __future__ import annotations

import re

import numpy as np

from .registry import JNU
from .integrity import (sha256_file, sha256_array, signal_checks,
                        boundary_jump_probe)

_FNAME = re.compile(r"^(?P<cond>n|ib|ob|tb)(?P<speed>600|800|1000)"
                    r"(?P<suffix>(?:_\d+)+)\.csv$")

_COND_TO_FAULT = {"n": "healthy", "ib": "inner_race", "ob": "outer_race",
                  "tb": "rolling_element"}
_COND_TO_MAPPED = {"n": "Healthy", "ib": "InnerRace", "ob": "OuterRace",
                   "tb": "RollingElement"}


def audit_jnu() -> tuple[list[dict], list[dict]]:
    root = JNU["paths"]["root"]
    rate = JNU["sampling_rate_hz"]
    rows, integrity = [], []

    for path in sorted(root.glob("*.csv")):
        m = _FNAME.match(path.name)
        if not m:
            raise AssertionError(f"unrecognised JNU filename: {path.name}")
        cond, speed = m.group("cond"), int(m.group("speed"))

        x = np.loadtxt(path, dtype=np.float64)
        if x.ndim != 1:
            raise AssertionError(f"{path.name}: expected 1 column, "
                                 f"got shape {x.shape}")

        rec = {
            "dataset": "JNU",
            "file": path.name,
            "sha256": sha256_file(path),
            "payload_sha256": sha256_array(x),
            **{f"sig_{k}": v for k, v in
               signal_checks(x, expected_min_len=rate).items()},
        }
        # concatenation probe for the 3x-length healthy files
        if cond == "n":
            third = x.size // 3
            rec["boundary_probe"] = boundary_jump_probe(
                x, [third, 2 * third])
        integrity.append(rec)

        rows.append({
            "dataset": "JNU",
            "recording_id": f"jnu_{path.stem}",
            "group_id_candidate": f"jnu_{cond}_{speed}",
            # one physical specimen per condition, reused across speeds
            # (single seeded defect per element documented; per-speed
            # re-seeding not documented -> inferred)
            "physical_bearing_id": f"jnu_bearing_{cond}",
            "experiment_id": f"jnu_{cond}_{speed}",
            "session_id": None,
            "original_file": f"data/raw_jnu/JNU-Bearing-Dataset/{path.name}",
            "original_label": cond,
            "mapped_label_candidate": _COND_TO_MAPPED[cond],
            "sampling_rate_hz": rate,
            "duration_seconds": x.size / rate,
            "n_samples": int(x.size),
            "rpm": speed,
            "load": None,
            "sensor_channel": "acc_vertical",
            "fault_type": _COND_TO_FAULT[cond],
            "fault_severity": (None if cond == "n"
                               else "0.3mm x 0.05mm wire-cut dent"),
            "bearing_position": None,
            "source_url": JNU["source_url"],
            "metadata_confidence": "documented",
            "notes": (f"filename suffix '{m.group('suffix')}' undocumented; "
                      f"clone commit {JNU['clone_commit'][:12]}"),
        })

    if len(rows) != 12:
        raise AssertionError(f"expected 12 JNU recordings, found {len(rows)}")
    return rows, integrity
