"""MaFaulDa audit: one manifest recording == one 5 s acquisition CSV.

Directory taxonomy of the official full.zip (operational ground truth for
labels; the website's sec-1.4.5 prose permutes fault-type names relative to
its own summary table — recorded in the integrity report):

    normal/<speed>.csv
    imbalance/<Wg>/<speed>.csv
    horizontal-misalignment/<Dmm>/<speed>.csv
    vertical-misalignment/<Dmm>/<speed>.csv
    underhang/<subfault>/<Wg>/<speed>.csv
    overhang/<subfault>/<Wg>/<speed>.csv

<speed> is the measured rotation frequency in Hz. Every CSV holds
250,000 rows x 8 columns at 50 kHz: tachometer; underhang accelerometer
(axial, radial, tangential); overhang accelerometer (axial, radial,
tangential); microphone.

The three defective bearings supplied by the manufacturer were moved
between the underhang and overhang positions, so underhang/X and
overhang/X share one physical defective bearing — physical_bearing_id
encodes this.
"""
from __future__ import annotations

from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from .registry import MAFAULDA
from .integrity import sha256_file, sha256_array

_N_EXPECTED = 250_000
_RATE = 50_000
_CHANNEL_NAMES = ["tachometer", "underhang_axial", "underhang_radial",
                  "underhang_tangential", "overhang_axial", "overhang_radial",
                  "overhang_tangential", "microphone"]


def _parse_path(rel: Path) -> dict:
    parts = rel.parts
    top = parts[0]
    speed_hz = float(rel.stem)
    if top == "normal":
        assert len(parts) == 2, rel
        return {"label": "normal", "fault_type": "healthy", "severity": None,
                "position": None, "config": "normal", "speed_hz": speed_hz,
                "bearing": "mafaulda_bearings_healthy"}
    if top == "imbalance":
        assert len(parts) == 3, rel
        return {"label": "imbalance", "fault_type": "imbalance",
                "severity": parts[1], "position": None,
                "config": f"imbalance_{parts[1]}", "speed_hz": speed_hz,
                "bearing": "mafaulda_bearings_healthy"}
    if top in ("horizontal-misalignment", "vertical-misalignment"):
        assert len(parts) == 3, rel
        kind = top.split("-")[0]
        return {"label": top, "fault_type": f"{kind}_misalignment",
                "severity": parts[1], "position": None,
                "config": f"{top}_{parts[1]}", "speed_hz": speed_hz,
                "bearing": "mafaulda_bearings_healthy"}
    if top in ("underhang", "overhang"):
        assert len(parts) == 4, rel
        subfault, mass = parts[1], parts[2]
        return {"label": f"{top}/{subfault}",
                "fault_type": f"bearing_{subfault}",
                "severity": f"added_mass_{mass}", "position": top,
                "config": f"{top}_{subfault}_{mass}", "speed_hz": speed_hz,
                "bearing": f"mafaulda_defective_{subfault}"}
    raise AssertionError(f"unrecognised MaFaulDa path: {rel}")


def _audit_one(args: tuple[str, str]) -> tuple[dict, dict]:
    root_s, rel_s = args
    root, rel = Path(root_s), Path(rel_s)
    path = root / rel
    meta = _parse_path(rel)

    df = pd.read_csv(path, header=None, dtype=np.float64)
    a = df.to_numpy()
    n_rows, n_cols = a.shape

    finite = np.isfinite(a)
    col_const = [bool((a[:, c] == a[0, c]).all()) for c in range(n_cols)]
    integrity = {
        "dataset": "MAFAULDA",
        "file": str(rel),
        "sha256": sha256_file(path),
        "payload_sha256": sha256_array(a),
        "n_rows": int(n_rows),
        "n_cols": int(n_cols),
        "shape_ok": (n_rows == _N_EXPECTED and n_cols == 8),
        "n_nan": int(np.isnan(a).sum()),
        "n_inf": int((~finite).sum() - np.isnan(a).sum()),
        "constant_channels": ",".join(
            _CHANNEL_NAMES[c] for c in range(min(n_cols, 8))
            if col_const[c]) or None,
        "ok": bool(n_rows == _N_EXPECTED and n_cols == 8
                   and finite.all() and not any(col_const)),
    }

    rpm = meta["speed_hz"] * 60.0
    row = {
        "dataset": "MAFAULDA",
        "recording_id": "mafaulda_" + str(rel.with_suffix("")).replace("/", "_"),
        "group_id_candidate": f"mafaulda_{meta['config']}",
        "physical_bearing_id": meta["bearing"],
        "experiment_id": f"mafaulda_{meta['config']}",
        "session_id": None,
        "original_file": str(Path("data/raw_mafaulda/full") / rel),
        "original_label": meta["label"],
        "mapped_label_candidate": {
            "healthy": "Healthy",
            "imbalance": "Imbalance_NON_BEARING",
            "horizontal_misalignment": "Misalignment_NON_BEARING",
            "vertical_misalignment": "Misalignment_NON_BEARING",
            "bearing_ball_fault": "RollingElement",
            "bearing_cage_fault": "Cage_NO_CROSS_DATASET_MATCH",
            "bearing_outer_race": "OuterRace",
        }[meta["fault_type"]],
        "sampling_rate_hz": _RATE,
        "duration_seconds": n_rows / _RATE,
        "n_samples": int(n_rows),
        "rpm": rpm,
        "load": (meta["severity"] if meta["fault_type"] == "imbalance"
                 or (meta["severity"] or "").startswith("added_mass")
                 else None),
        "sensor_channel": "8ch(tacho;uh_ax,rad,tan;oh_ax,rad,tan;mic)",
        "fault_type": meta["fault_type"],
        "fault_severity": meta["severity"],
        "bearing_position": meta["position"],
        "source_url": MAFAULDA["source_url"],
        "metadata_confidence": "documented",
        "notes": f"rotation {meta['speed_hz']:g} Hz from filename",
    }
    return row, integrity


def audit_mafaulda(n_workers: int = 4) -> tuple[list[dict], list[dict]]:
    root = MAFAULDA["paths"]["root"]
    rels = sorted(str(p.relative_to(root)) for p in root.rglob("*.csv"))
    if not rels:
        raise AssertionError(f"no MaFaulDa CSVs found under {root}")
    args = [(str(root), r) for r in rels]
    with Pool(n_workers) as pool:
        results = pool.map(_audit_one, args, chunksize=16)
    rows = [r for r, _ in results]
    integrity = [i for _, i in results]
    return rows, integrity
