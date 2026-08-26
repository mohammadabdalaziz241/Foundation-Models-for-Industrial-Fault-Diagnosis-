"""Dataset census assembly (Step 2 deliverables)."""
from __future__ import annotations

import pandas as pd

from .registry import DATASETS

CENSUS_COLUMNS = [
    "dataset", "original_source", "local_path", "n_raw_files",
    "n_recordings", "machinery_type", "sensor_type", "sensor_channels",
    "sampling_rates_hz", "recording_duration_s", "samples_per_recording",
    "rpm_speeds", "load_conditions", "fault_classes", "fault_severities",
    "fault_locations", "bearing_positions", "physical_bearing_identity",
    "experiment_run_identity", "multi_file_same_experiment",
    "official_train_test_split", "file_format", "licence_provenance",
]


def _rng(series) -> str:
    lo, hi = float(series.min()), float(series.max())
    return f"{lo:g}" if lo == hi else f"{lo:g}-{hi:g}"


def build_census(manifest: pd.DataFrame, n_raw_files: dict[str, int],
                 extra: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for name, reg in DATASETS.items():
        sub = manifest[manifest["dataset"] == name]
        if sub.empty:
            continue
        e = extra.get(name, {})
        rows.append({
            "dataset": name,
            "original_source": reg["source_url"],
            "local_path": e.get("local_path", ""),
            "n_raw_files": n_raw_files.get(name, len(sub)),
            "n_recordings": len(sub),
            "machinery_type": reg["machinery"],
            "sensor_type": reg["sensor"],
            "sensor_channels": "; ".join(sorted(sub["sensor_channel"]
                                                .unique())),
            "sampling_rates_hz": "; ".join(
                f"{int(r)}" for r in sorted(sub["sampling_rate_hz"]
                                            .unique())),
            "recording_duration_s": _rng(sub["duration_seconds"]),
            "samples_per_recording": _rng(sub["n_samples"]),
            "rpm_speeds": e.get("rpm_note", _rng(sub["rpm"].dropna())),
            "load_conditions": "; ".join(
                sorted(str(v) for v in sub["load"].dropna().unique())[:12])
                or "not applicable / not documented",
            "fault_classes": "; ".join(
                sorted(str(v) for v in sub["original_label"].unique())),
            "fault_severities": "; ".join(
                sorted(str(v) for v in sub["fault_severity"].dropna()
                       .unique())[:12]) or "none documented",
            "fault_locations": "; ".join(
                sorted(str(v) for v in sub["fault_type"].unique())),
            "bearing_positions": "; ".join(
                sorted(str(v) for v in sub["bearing_position"].dropna()
                       .unique())) or "not documented",
            "physical_bearing_identity": e.get("bearing_identity",
                                               "see grouping_policy.md"),
            "experiment_run_identity": e.get("run_identity",
                                             "see grouping_policy.md"),
            "multi_file_same_experiment": e.get("multi_file_note", "unknown"),
            "official_train_test_split": e.get("official_split", "none"),
            "file_format": e.get("file_format", "unknown"),
            "licence_provenance": e.get("licence", "not documented"),
        })
    return pd.DataFrame(rows, columns=CENSUS_COLUMNS)


def census_markdown(census: pd.DataFrame) -> str:
    lines = ["# Dataset census — methodology_v2 Part 1", ""]
    lines.append("One section per candidate dataset. Values marked "
                 "'not documented' were not stated by the original source "
                 "and are deliberately left unresolved rather than inferred.")
    lines.append("")
    for _, row in census.iterrows():
        lines.append(f"## {row['dataset']}")
        lines.append("")
        lines.append("| Field | Value |")
        lines.append("|---|---|")
        for col in CENSUS_COLUMNS[1:]:
            val = str(row[col]).replace("|", "\\|")
            lines.append(f"| {col} | {val} |")
        lines.append("")
    return "\n".join(lines)
