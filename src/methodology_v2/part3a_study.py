"""Part 3A — signal-input design study (metadata/physics only).

Computes, WITHOUT reading any raw signal:
  - channel census (documented sensor facts),
  - common sampling-rate comparison (rational conversion ratios, Nyquist),
  - window-duration / shaft-rotation study from documented RPM,
  - future window-count feasibility per frozen fold/split (counts only —
    no windows are materialised),
  - JNU guard-width instantiation study (G >= window span, symbolic),
  - CWRU load-0 short-recording compatibility.

Nothing here modifies Part-1/Part-2 artefacts; the runner verifies the
Part-2 seal (fail closed) before writing anything.
"""
from __future__ import annotations

import math
from fractions import Fraction
from pathlib import Path

import pandas as pd

from .part2_builder import PART2_DIR
from .registry import OUTPUT_DIR as PART1_DIR, REPO_ROOT

PART3A_DIR = REPO_ROOT / "methodology_v2" / "part3_input_design"

DURATIONS_S = (0.25, 0.50, 1.00, 2.00)
NATIVE_RATE = {"CWRU": 48_000, "JNU": 50_000, "HIT": 25_000,
               "MAFAULDA": 50_000}
CANDIDATE_RATES = (20_000, 24_000, 25_000, 32_000)

# ---------------------------------------------------------------------------
# 3A.1 channel census — documented facts only (Part-1 census + source docs)
# ---------------------------------------------------------------------------
CHANNEL_CENSUS = [
    # dataset, channel, sensor, orientation, location, near_bearing,
    # synchronous, non_vibration, candidate, notes
    ("CWRU", "DE_time", "accelerometer (IMI, 12 o'clock housing)",
     "radial/vertical", "drive-end bearing housing (faulted bearing)",
     "yes", "yes (DE+FE same clock)", "no", "PRIMARY",
     "sensor directly on the housing of the seeded-fault bearing"),
    ("CWRU", "FE_time", "accelerometer (12 o'clock housing)",
     "radial/vertical", "fan-end bearing housing",
     "no (remote from DE fault)", "yes", "no", "alternative",
     "transfer path crosses the motor; weaker fault coupling"),
    ("CWRU", "RPM (scalar)", "torque transducer/encoder", "-", "shaft",
     "-", "-", "yes", "no", "single scalar per file, metadata only"),
    ("JNU", "acc_vertical", "PCB MA352A60 accelerometer", "vertical",
     "test-bearing housing (documented: vertical direction)",
     "yes (documented)", "single channel", "no", "PRIMARY (only channel)",
     "no alternative exists"),
    ("HIT", "ch1", "displacement (paper Table IV: KISTLER 8776A50M1)",
     "horizontal", "LP rotor", "no (rotor displacement)", "yes (6ch)",
     "yes (displacement, not acceleration)", "no",
     "different physical quantity; official GitHub windows came from ch1"),
    ("HIT", "ch2", "displacement", "vertical", "LP rotor", "no",
     "yes", "yes (displacement)", "no", ""),
    ("HIT", "ch3", "acceleration (K9000XL)", "normal-to-casing (radial)",
     "casing, measuring point 3", "casing (inter-shaft bearing internal)",
     "yes", "no", "PRIMARY",
     "first casing accelerometer in deterministic channel order; exact "
     "axial station vs inter-shaft bearing not documented"),
    ("HIT", "ch4", "acceleration (K9000XL)", "normal-to-casing (radial)",
     "casing, measuring point 4", "casing", "yes", "no", "alternative", ""),
    ("HIT", "ch5", "acceleration (K9000XL)", "normal-to-casing (radial)",
     "casing, measuring point 5", "casing", "yes", "no", "alternative", ""),
    ("HIT", "ch6", "acceleration (K9000XL)", "normal-to-casing (radial)",
     "casing, measuring point 6", "casing", "yes", "no", "alternative", ""),
    ("MAFAULDA", "col1 tachometer", "Monarch MT-190 analog tachometer",
     "-", "shaft", "-", "yes (8ch, 2x NI 9234)", "yes (speed signal)",
     "no", "usable later for speed verification only"),
    ("MAFAULDA", "col2 underhang axial", "IMI 601A01 accelerometer",
     "axial", "underhang bearing (between rotor and motor)", "yes",
     "yes", "no", "no", ""),
    ("MAFAULDA", "col3 underhang radial", "IMI 601A01 accelerometer",
     "radial", "underhang bearing", "yes", "yes", "no", "PRIMARY",
     "radial on a bearing housing = closest common denominator"),
    ("MAFAULDA", "col4 underhang tangential", "IMI 601A01 accelerometer",
     "tangential", "underhang bearing", "yes", "yes", "no", "no", ""),
    ("MAFAULDA", "col5 overhang axial", "IMI 604B31 triaxial", "axial",
     "overhang bearing (rotor between bearing and motor)", "yes", "yes",
     "no", "no", ""),
    ("MAFAULDA", "col6 overhang radial", "IMI 604B31 triaxial", "radial",
     "overhang bearing", "yes", "yes", "no", "alternative",
     "fault-position-ADAPTIVE channel choice is REJECTED: it would inject "
     "label information into preprocessing"),
    ("MAFAULDA", "col7 overhang tangential", "IMI 604B31 triaxial",
     "tangential", "overhang bearing", "yes", "yes", "no", "no", ""),
    ("MAFAULDA", "col8 microphone", "Shure SM81", "-", "ambient", "-",
     "yes", "yes (acoustic)", "no", ""),
]

CENSUS_COLS = ["dataset", "channel", "sensor_type", "orientation",
               "location", "near_faulted_bearing", "synchronous",
               "non_vibration", "candidate_main_encoder", "notes"]


def channel_census_df() -> pd.DataFrame:
    return pd.DataFrame(CHANNEL_CENSUS, columns=CENSUS_COLS)


# ---------------------------------------------------------------------------
# 3A.2 sampling-rate study
# ---------------------------------------------------------------------------

def rate_study_df() -> pd.DataFrame:
    rows = []
    for target in ("native",) + CANDIDATE_RATES:
        for ds, native in NATIVE_RATE.items():
            if target == "native":
                op, frac = "none (native)", Fraction(1, 1)
                nyq = native / 2
            else:
                frac = Fraction(target, native)
                nyq = target / 2
                op = ("none (native)" if target == native else
                      "downsample" if target < native else
                      "UPSAMPLE (creates no information)")
            rows.append({
                "target_rate_hz": target, "dataset": ds,
                "native_rate_hz": native, "operation": op,
                "ratio_up_L": frac.numerator, "ratio_down_M":
                    frac.denominator,
                "nyquist_hz": nyq,
                "anti_alias_needed": op == "downsample",
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3A.3 window-duration / rotation study
# ---------------------------------------------------------------------------

def rotations(rpm: float, w_s: float) -> float:
    return rpm / 60.0 * w_s


def duration_study_df(fold1: pd.DataFrame) -> pd.DataFrame:
    """Rotation statistics from documented/audited RPM metadata."""
    rows = []
    for ds in ("CWRU", "JNU", "HIT", "MAFAULDA"):
        sub = fold1[(fold1["dataset"] == ds)
                    & (fold1["is_usable"] == True)]  # noqa: E712
        rec = sub.drop_duplicates("recording_id")
        rpms = rec["rpm"].astype(float)
        speeds = {"basis": "shaft RPM (documented)", "min": rpms.min(),
                  "med": rpms.median(), "max": rpms.max()}
        speed_sets = [speeds]
        if ds == "HIT":
            hp = rec["experiment_id"].str.extract(r"hp(\d+)")[0].astype(float)
            rel = hp - rpms
            speed_sets = [
                {"basis": "LP shaft RPM", "min": rpms.min(),
                 "med": rpms.median(), "max": rpms.max()},
                {"basis": "relative HP-LP RPM (inter-shaft race speed)",
                 "min": rel.min(), "med": rel.median(), "max": rel.max()},
            ]
        for sp in speed_sets:
            for w in DURATIONS_S:
                rows.append({
                    "dataset": ds, "rpm_basis": sp["basis"],
                    "window_s": w,
                    "rpm_min": round(sp["min"], 1),
                    "rpm_median": round(sp["med"], 1),
                    "rpm_max": round(sp["max"], 1),
                    "rot_min": round(rotations(sp["min"], w), 2),
                    "rot_median": round(rotations(sp["med"], w), 2),
                    "rot_max": round(rotations(sp["max"], w), 2),
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3A.4 window-count feasibility (counts only, never windows)
# ---------------------------------------------------------------------------

def n_windows(length_s: float, w_s: float, overlap: float) -> int:
    """floor((L - W)/step) + 1 windows fit in a region of length L."""
    if length_s < w_s:
        return 0
    step = w_s * (1.0 - overlap)
    return int(math.floor((length_s - w_s) / step + 1e-9)) + 1


def jnu_usable_block_s(nominal_s: float, w_s: float,
                       guard_before: bool, guard_after: bool) -> float:
    """Usable seconds of a JNU macro-block once guards instantiate with
    G = window span W: each flagged edge loses G/2."""
    return nominal_s - (w_s / 2.0) * (int(guard_before) + int(guard_after))


def window_count_df(folds: dict[int, pd.DataFrame],
                    part1: pd.DataFrame) -> pd.DataFrame:
    dur = part1.set_index("recording_id")["duration_seconds"]
    rows = []
    for k, df in folds.items():
        usable = df[df["is_usable"] == True]  # noqa: E712
        for ds in ("CWRU", "JNU", "HIT", "MAFAULDA"):
            sub = usable[usable["dataset"] == ds]
            for sp in ("train", "validation", "test"):
                part = sub[sub["split"] == sp]
                for w in DURATIONS_S:
                    for ov in (0.0, 0.5):
                        if ds == "JNU":
                            lens = [
                                jnu_usable_block_s(
                                    (r["temporal_end_sample"]
                                     - r["temporal_start_sample"]) / 50_000,
                                    w, bool(r["guard_before"]),
                                    bool(r["guard_after"]))
                                for _, r in part.iterrows()]
                        else:
                            lens = dur.reindex(part["recording_id"]).tolist()
                        counts = [n_windows(x, w, ov) for x in lens]
                        rows.append({
                            "fold_id": k, "dataset": ds, "split": sp,
                            "window_s": w, "overlap_pct": int(ov * 100),
                            "counting_basis": "standard",
                            "n_regions": len(counts),
                            "n_regions_too_short":
                                sum(c == 0 for c in counts),
                            "n_windows": int(sum(counts)),
                        })
                        if ds == "HIT":
                            # conservative alternative: windows confined to
                            # single 20480-sample series (0.8192 s)
                            per_series = n_windows(20_480 / 25_000, w, ov)
                            rows.append({
                                "fold_id": k, "dataset": ds, "split": sp,
                                "window_s": w, "overlap_pct": int(ov * 100),
                                "counting_basis": "hit_series_constrained",
                                "n_regions": len(counts) * 18,
                                "n_regions_too_short":
                                    0 if per_series else len(counts) * 18,
                                "n_windows": per_series * len(counts) * 18,
                            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3A.6 JNU guard study
# ---------------------------------------------------------------------------

def jnu_guard_df() -> pd.DataFrame:
    rows = []
    for w in DURATIONS_S:
        g_s = w  # frozen rule: G >= effective window span; minimum G = W
        row = {"window_s": w, "guard_s": g_s,
               "guard_samples_native_50k": int(round(g_s * 50_000))}
        for r in CANDIDATE_RATES:
            row[f"guard_samples_{r//1000}k"] = int(round(g_s * r))
        # usable signal after guard removal (4 internal boundaries lose G
        # each, split G/2+G/2 across the two adjacent blocks)
        fault_usable = 10.01 - 4 * g_s
        healthy_usable = 30.03 - 4 * g_s
        row.update({
            "fault_recording_usable_s": round(fault_usable, 3),
            "fault_usable_fraction": round(fault_usable / 10.01, 3),
            "healthy_recording_usable_s": round(healthy_usable, 3),
            "healthy_usable_fraction": round(healthy_usable / 30.03, 3),
            "internal_fault_block_usable_s": round(2.002 - g_s, 3),
            "edge_fault_block_usable_s": round(2.002 - g_s / 2, 3),
            "internal_block_fits_one_window": (2.002 - g_s) >= w,
        })
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3A.7 CWRU load-0 analysis
# ---------------------------------------------------------------------------

def cwru_load0_df(fold1: pd.DataFrame, part1: pd.DataFrame) -> pd.DataFrame:
    dur = part1.set_index("recording_id")["duration_seconds"]
    cw = fold1[fold1["dataset"] == "CWRU"].drop_duplicates("recording_id")
    l0 = cw[cw["load"] == "0hp"]
    lens = dur.reindex(l0["recording_id"])
    rows = []
    for w in DURATIONS_S:
        counts0 = [n_windows(x, w, 0.0) for x in lens]
        counts50 = [n_windows(x, w, 0.5) for x in lens]
        rows.append({
            "window_s": w,
            "n_load0_recordings": len(l0),
            "shortest_load0_s": round(float(lens.min()), 3),
            "n_load0_zero_windows": sum(c == 0 for c in counts0),
            "min_windows_0pct": min(counts0),
            "total_windows_0pct": sum(counts0),
            "total_windows_50pct": sum(counts50),
            "eliminates_any_recording": any(c == 0 for c in counts0),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

def load_frozen_inputs():
    part1 = pd.read_csv(PART1_DIR / "recording_manifest.csv")
    folds = {k: pd.read_csv(PART2_DIR / f"global_fold_{k}.csv")
             for k in (1, 2, 3)}
    return part1, folds


def write_study_tables(out_dir: Path | None = None) -> dict:
    out_dir = Path(out_dir) if out_dir else PART3A_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    part1, folds = load_frozen_inputs()

    census = channel_census_df()
    census.to_csv(out_dir / "channel_census.csv", index=False)
    rates = rate_study_df()
    rates.to_csv(out_dir / "sampling_rate_study.csv", index=False)
    durs = duration_study_df(folds[1])
    durs.to_csv(out_dir / "window_duration_study.csv", index=False)
    counts = window_count_df(folds, part1)
    counts.to_csv(out_dir / "window_count_estimates.csv", index=False)
    guards = jnu_guard_df()
    guards.to_csv(out_dir / "jnu_guard_study.csv", index=False)
    load0 = cwru_load0_df(folds[1], part1)
    load0.to_csv(out_dir / "cwru_load0_study.csv", index=False)

    return {"out_dir": out_dir, "census": census, "rates": rates,
            "durations": durs, "counts": counts, "guards": guards,
            "load0": load0}
