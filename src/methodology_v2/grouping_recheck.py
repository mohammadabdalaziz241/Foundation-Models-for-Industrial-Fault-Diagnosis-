"""CWRU grouping re-check (bounded Part-1 addendum, audit-only).

Builds, for the retained 48 kHz drive-end benchmark subset (Normal + IR/B/OR
at 0.007"/0.014"/0.021"; 0.028" and the 12 kHz fault family excluded), a
recording/specimen identity table and the group structure implied by three
candidate grouping units:

  A  original recording
  B  specimen x load
  C1 physical specimen across all loads, OR clock positions kept separate
     (installation-wise)
  C2 physical specimen across all loads, OR clock positions merged
     (specimen-wise, leakage-conservative)

Inputs are the frozen Part-1 manifest and the hash-frozen 48 kHz
enumeration (Amendment 4) that records each local file's official download
number, source URL and sha256. Read-only; no windows, no splits.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from .registry import REPO_ROOT, OUTPUT_DIR

ENUMERATION_PATH = (REPO_ROOT / "metadata" / "vibrationclip_v1"
                    / "cwru_48k_enumeration.json")

# Official 48k DE table (fetched 2026-08-11 from
# engineering.case.edu/bearingdatacenter/48k-drive-end-bearing-fault-data)
# spec -> {load_hp: official download file number}. Note the official
# numbering quirks: IR014 0hp is 174 (its internal variables are X173_*)
# and IR021 3hp is 217 (216 is skipped; 217 internally carries leftover
# X215_* variables besides X217_*).
OFFICIAL_48K_TABLE = {
    "IR007": {0: 109, 1: 110, 2: 111, 3: 112},
    "B007": {0: 122, 1: 123, 2: 124, 3: 125},
    "OR007@6": {0: 135, 1: 136, 2: 137, 3: 138},
    "OR007@3": {0: 148, 1: 149, 2: 150, 3: 151},
    "OR007@12": {0: 161, 1: 162, 2: 163, 3: 164},
    "IR014": {0: 174, 1: 175, 2: 176, 3: 177},
    "B014": {0: 189, 1: 190, 2: 191, 3: 192},
    "OR014@6": {0: 201, 1: 202, 2: 203, 3: 204},
    "IR021": {0: 213, 1: 214, 2: 215, 3: 217},
    "B021": {0: 226, 1: 227, 2: 228, 3: 229},
    "OR021@6": {0: 238, 1: 239, 2: 240, 3: 241},
    "OR021@3": {0: 250, 1: 251, 2: 252, 3: 253},
    "OR021@12": {0: 262, 1: 263, 2: 264, 3: 265},
    "Normal": {0: 97, 1: 98, 2: 99, 3: 100},
}

_CLASS_OF = {"IR": "InnerRace", "B": "Ball", "OR": "OuterRace",
             "Normal": "Healthy"}


def _parse_spec(label: str) -> tuple[str, int | None, str | None]:
    """label -> (class_key, diameter_mil, or_position)."""
    if label == "Normal":
        return "Normal", None, None
    m = re.match(r"^(IR|B|OR)(\d{3})(?:@(\d+))?$", label)
    if not m:
        raise AssertionError(f"unparseable CWRU label: {label}")
    return m.group(1), int(m.group(2)), m.group(3)


def build_recheck() -> tuple[pd.DataFrame, dict]:
    manifest = pd.read_csv(OUTPUT_DIR / "recording_manifest.csv")
    cwru = manifest[manifest["dataset"] == "CWRU"]

    enum = json.load(open(ENUMERATION_PATH))
    by_path = {e["file"]: e for e in enum["files"]}

    rows = []
    for _, r in cwru.iterrows():
        label = r["original_label"]
        is_normal = label == "Normal"
        in_48k = str(r["original_file"]).startswith("data/raw_cwru_48k/")
        if not (in_48k or is_normal):
            continue  # 12 kHz fault family: excluded from the new benchmark
        key, dia, pos = _parse_spec(label)
        if dia == 28:
            raise AssertionError("0.028\" must not appear in the 48k family")
        load = int(str(r["load"]).rstrip("hp"))

        # enumeration entry: faults by path; normals via their byte-identical
        # copy inside data/raw_cwru_48k/normal/
        epath = (r["original_file"] if not is_normal else
                 f"data/raw_cwru_48k/normal/Normal_{load}HP_baseline.mat")
        e = by_path.get(epath)
        if e is None:
            raise AssertionError(f"no enumeration entry for {epath}")
        official = OFFICIAL_48K_TABLE[label][load]
        if int(e["source_number"]) != official:
            raise AssertionError(
                f"{label} load {load}: enumeration source_number "
                f"{e['source_number']} != official table {official}")

        internal = r["recording_id"].removeprefix("cwru_")  # e.g. X173
        spec_c1 = label                                     # positions kept
        spec_c2 = label.split("@")[0]                       # positions merged
        rows.append({
            "recording_id": r["recording_id"],
            "internal_variable_id": internal,
            "official_download_number": official,
            "source_sha256_frozen": e["sha256"][:16],
            "class": _CLASS_OF[key],
            "fault_type": r["fault_type"],
            "diameter_mil": dia,
            "or_position_oclock": pos,
            "load_hp": load,
            "rpm_in_file": r["rpm"],
            "n_samples": int(r["n_samples"]),
            "duration_s": round(float(r["duration_seconds"]), 3),
            "group_A_recording": r["recording_id"],
            "group_B_specimen_x_load": f"cwru48k_{label}_load{load}",
            "group_C1_installation": f"cwru48k_{spec_c1}",
            "group_C2_specimen": f"cwru48k_{spec_c2}",
            "specimen_identity_evidence": (
                "single healthy baseline set 97-100; no second healthy "
                "bearing documented" if is_normal else
                "one seeded bearing per fault spec (welcome-page procedure "
                "text) + consecutive official numbering across the 4 loads"
                + ("; OR position identity across reinstallations NOT "
                   "documented" if pos else "")),
        })

    df = pd.DataFrame(rows).sort_values(
        ["class", "diameter_mil", "or_position_oclock", "load_hp"],
        na_position="first").reset_index(drop=True)
    if len(df) != 56:
        raise AssertionError(f"expected 56 retained recordings, got {len(df)}")

    stats = {}
    for opt, col in (("A_recording", "group_A_recording"),
                     ("B_specimen_x_load", "group_B_specimen_x_load"),
                     ("C1_installation", "group_C1_installation"),
                     ("C2_specimen", "group_C2_specimen")):
        g = df.groupby(col)
        per_class = df.groupby("class")[col].nunique().to_dict()
        per_sev = (df[df["diameter_mil"].notna()]
                   .groupby("diameter_mil")[col].nunique().to_dict())
        per_load = df.groupby("load_hp")[col].nunique().to_dict()
        # can one physical specimen (C2 unit) span several groups?
        spans = (df.groupby("group_C2_specimen")[col].nunique() > 1).any()
        stats[opt] = {
            "n_groups": int(g.ngroups),
            "groups_per_class": {k: int(v) for k, v in per_class.items()},
            "groups_per_severity_mil": {int(k): int(v)
                                        for k, v in per_sev.items()},
            "groups_touching_each_load": {int(k): int(v)
                                          for k, v in per_load.items()},
            "same_physical_specimen_can_cross_partitions": bool(spans),
            "recordings_per_group_min": int(g.size().min()),
            "recordings_per_group_max": int(g.size().max()),
        }
    return df, stats


def main() -> None:
    df, stats = build_recheck()
    out_csv = OUTPUT_DIR / "cwru_grouping_recheck_table.csv"
    out_json = OUTPUT_DIR / "cwru_grouping_recheck_stats.json"
    df.to_csv(out_csv, index=False)
    with open(out_json, "w") as f:
        json.dump(stats, f, indent=1)
    print(f"{len(df)} retained recordings -> {out_csv.name}")
    for opt, s in stats.items():
        print(f"{opt}: {s['n_groups']} groups | per class "
              f"{s['groups_per_class']} | specimen-crossing possible: "
              f"{s['same_physical_specimen_can_cross_partitions']}")


if __name__ == "__main__":
    main()
