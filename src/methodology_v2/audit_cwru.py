"""CWRU audit: recording-level manifest rows + integrity records.

One recording == one original CWRU acquisition (.mat file). The canonical
CWRU file number is recovered from the internal MATLAB variable names
(X105_DE_time -> file 105), which survives the local renaming and lets every
row be traced back to the official Bearing Data Center tables.

Known official quirks handled explicitly (they are evidence of authenticity,
not corruption — see Smith & Randall 2015 appendix):
  - 98.mat has no RPM variable;
  - 99.mat additionally contains leftover X098_* variables and an 'ans' var;
  - 175.mat (48k IR014, 1 hp) contains stray X217 variables;
  - 217.mat (48k IR021, 3 hp) contains leftover X215_* variables;
  - 173.mat (48k IR014, 0 hp) is unusually short (~1.3 s);
  - the four Normal_* files exist byte-identically in both local dirs.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import scipy.io as sio

from .registry import CWRU
from .integrity import sha256_file, sha256_array, signal_checks

_FNAME = re.compile(
    r"^(?P<spec>Normal|B\d{3}|IR\d{3}|OR\d{3}@\d+)_(?P<load>\d)"
    r"(?:HP)?(?:_baseline|_DE12k|_DE48k)?\.mat$"
)


def _canonical_ids(matvars: dict) -> list[str]:
    """All canonical CWRU file numbers present as X<nnn> variables."""
    ids = set()
    for k in matvars:
        m = re.match(r"^X(\d{3})", k)
        if m:
            ids.add(m.group(1))
    return sorted(ids)


def _payload_id(fname_spec: str, load: int, ids: list[str],
                matvars: dict) -> str:
    """Pick the canonical id that carries this file's payload.

    Files with leftover variables from a neighbouring acquisition contain
    several X<nnn> ids; the payload id is the one whose _DE_time variable is
    present and matches the largest complete channel set, resolved by the
    known quirk table (99->X099 primary, 175->X175, 217->X217).
    """
    if len(ids) == 1:
        return ids[0]
    known_primary = {("Normal", 2): "099",
                     ("IR014", 1): "175",
                     ("IR021", 3): "217"}
    key = (fname_spec, load)
    if key in known_primary and known_primary[key] in ids:
        return known_primary[key]
    raise AssertionError(
        f"multiple canonical ids {ids} in {fname_spec} load {load} "
        f"with no known-quirk resolution")


def iter_cwru_files():
    for rate_key, rate_hz_faults in (("12k_de", 12_000), ("48k_de", 48_000)):
        root = CWRU["paths"][rate_key]
        for f in sorted(root.glob("*/*.mat")):
            yield rate_key, rate_hz_faults, f


def audit_cwru() -> tuple[list[dict], list[dict]]:
    """Returns (manifest_rows, integrity_records)."""
    rows: list[dict] = []
    integrity: list[dict] = []
    seen_payload_hash: dict[str, str] = {}

    for rate_key, fault_rate, path in iter_cwru_files():
        m = _FNAME.match(path.name)
        if not m:
            raise AssertionError(f"unrecognised CWRU filename: {path}")
        spec, load = m.group("spec"), int(m.group("load"))

        mat = sio.loadmat(str(path))
        matvars = {k: v for k, v in mat.items() if not k.startswith("__")}
        ids = _canonical_ids(matvars)
        pid = _payload_id(spec, load, ids, matvars)

        de = matvars.get(f"X{pid}_DE_time")
        fe = matvars.get(f"X{pid}_FE_time")
        ba = matvars.get(f"X{pid}_BA_time")
        rpm_var = matvars.get(f"X{pid}RPM")
        channels = [c for c, v in (("DE", de), ("FE", fe), ("BA", ba))
                    if v is not None]
        if de is None:
            raise AssertionError(f"{path}: no DE payload for id {pid}")

        # Normal baseline is genuinely 48 kHz regardless of directory
        # (frozen finding, docs/cwru_legacy_rate_impact_note.md).
        is_normal = spec == "Normal"
        rate = 48_000 if is_normal else fault_rate

        file_hash = sha256_file(path)
        payload_hash = sha256_array(de)

        # byte-duplicate detection across the two directories
        dup_of = seen_payload_hash.get(payload_hash)
        seen_payload_hash.setdefault(payload_hash, f"cwru_X{pid}:{rate_key}")

        integrity.append({
            "dataset": "CWRU",
            "file": str(path.relative_to(CWRU["paths"][rate_key].parent)),
            "sha256": file_hash,
            "payload_sha256": payload_hash,
            "canonical_ids_present": ",".join(ids),
            "payload_id": pid,
            "channels": ",".join(channels),
            "duplicate_of": dup_of,
            **{f"de_{k}": v for k, v in
               signal_checks(de, expected_min_len=int(0.5 * rate)).items()},
        })

        if dup_of is not None:
            # e.g. Normal_* duplicated in raw_cwru_48k — never a second
            # manifest recording.
            continue

        if is_normal:
            fault_type, severity, position = "healthy", None, None
            label = "Normal"
            bearing = "cwru_DE_healthy_motor"
        else:
            label = spec
            if spec.startswith("IR"):
                fault_type, severity, position = "inner_race", int(spec[2:5]), None
            elif spec.startswith("B"):
                fault_type, severity, position = "ball", int(spec[1:4]), None
            else:  # OR007@6 etc.
                fault_type = "outer_race"
                severity = int(spec[2:5])
                position = spec.split("@")[1]
            pos_part = f"@{position}" if position else ""
            bearing = f"cwru_DE_{fault_type}_{severity:03d}{pos_part}"

        rpm_measured = (float(np.ravel(rpm_var)[0])
                        if rpm_var is not None else None)
        rpm_nominal = CWRU["nominal_rpm_by_load_hp"][load]

        rows.append({
            "dataset": "CWRU",
            "recording_id": f"cwru_X{pid}",
            # provisional grouping unit: fault specimen x load, merging the
            # 12k and 48k acquisitions of the same physical experiment
            "group_id_candidate": f"cwru_{spec}_load{load}",
            "physical_bearing_id": bearing,
            "experiment_id": f"cwru_{spec}_load{load}",
            "session_id": None,
            "original_file": str(path.relative_to(CWRU["paths"][rate_key].parent.parent)),
            "original_label": label,
            "mapped_label_candidate": {
                "healthy": "Healthy", "inner_race": "InnerRace",
                "outer_race": "OuterRace", "ball": "RollingElement",
            }[fault_type],
            "sampling_rate_hz": rate,
            "duration_seconds": de.shape[0] / rate,
            "n_samples": int(de.shape[0]),
            "rpm": rpm_measured if rpm_measured is not None else rpm_nominal,
            "load": f"{load}hp",
            "sensor_channel": "+".join(channels),
            "fault_type": fault_type,
            "fault_severity": (f"{severity} mil" if severity else None),
            "bearing_position": ("drive_end" if not is_normal else None),
            "source_url": CWRU["source_url"],
            "metadata_confidence": ("documented" if rpm_measured is not None
                                    else "derived"),
            "notes": (f"canonical file {pid}.mat; rpm "
                      + ("measured in-file" if rpm_measured is not None
                         else f"nominal for {load} hp")
                      + ("; normal baseline genuinely 48 kHz "
                         "(frozen audit)" if is_normal else "")),
        })

    return rows, integrity
