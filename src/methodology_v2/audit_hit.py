"""HIT audit — two distinct releases of the same underlying test campaign.

(a) GitHub release (official benchmark files): xtrain_1..4 / xtest, each
    (2412, 2048) float32, channel 1 only, plus uint8 labels. These are
    pre-windowed and carry NO per-row speed/run metadata. They are audited
    (shapes, label counts, duplicate rows, provenance against the full
    release) but are NOT manifest recordings — a window is never a
    recording unit.

(b) Google Drive full release: data1..data5.npy, each (n_series, 8, 20480)
    — axis order (series, row, sample). Rows 1-2 displacement, 3-6
    acceleration; row 7 carries the LP speed at sample 0 and the HP speed
    at sample 1 (zeros elsewhere); row 8 carries the label at sample 0
    (paper Fig. 15, verified empirically). One manifest recording == one
    (session, speed-group) acquisition, i.e. the contiguous block of
    20480-sample series sharing an (LP, HP) speed pair within a session.
    Per the paper, each such acquisition was a 15 s continuous record at
    25 kHz, later segmented; invalid data were eliminated, so a recording
    may have fewer series than the nominal 18.
"""
from __future__ import annotations

import hashlib

import numpy as np
import scipy.io as sio

from .registry import HIT
from .integrity import sha256_file, sha256_array, signal_checks

SERIES_LEN = 20_480
WIN = 2_048


def _row_hash(a: np.ndarray) -> bytes:
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float32)
                          .tobytes()).digest()


def audit_hit_github() -> dict:
    """Audit the official windowed benchmark shards (not manifest rows)."""
    root = HIT["paths"]["github_release"]
    report: dict = {"files": {}, "label_counts": {}}
    xs, ys = {}, {}
    for stem in ("xtrain_1", "xtrain_2", "xtrain_3", "xtrain_4", "xtest",
                 "ytrain_1", "ytrain_2", "ytrain_3", "ytrain_4", "ytest"):
        path = root / f"{stem}.mat"
        mat = sio.loadmat(str(path))
        a = mat[stem]
        report["files"][stem] = {"shape": list(a.shape),
                                 "dtype": str(a.dtype),
                                 "sha256": sha256_file(path)}
        (xs if stem.startswith("x") else ys)[stem] = a

    for stem, a in ys.items():
        vals, counts = np.unique(a, return_counts=True)
        report["label_counts"][stem] = {int(v): int(c)
                                        for v, c in zip(vals, counts)}
        bad = set(vals.tolist()) - HIT["valid_labels"]
        if bad:
            raise AssertionError(f"HIT {stem}: labels outside registry: {bad}")

    # exact duplicate windows within/between shards
    seen: dict[bytes, str] = {}
    dup_within, dup_train_test = 0, 0
    for stem in ("xtrain_1", "xtrain_2", "xtrain_3", "xtrain_4", "xtest"):
        for i in range(xs[stem].shape[0]):
            h = _row_hash(xs[stem][i])
            if h in seen:
                if seen[h].startswith("xtrain") and stem == "xtest":
                    dup_train_test += 1
                else:
                    dup_within += 1
            else:
                seen[h] = stem
    report["exact_duplicate_windows"] = {
        "within_or_between_train_shards": dup_within,
        "between_train_and_test": dup_train_test,
        "total_windows": sum(x.shape[0] for x in xs.values()),
    }
    return report


def _provenance_check(report: dict, sessions: dict) -> None:
    """Match GitHub windows against aligned sub-windows of the full release.

    Builds hashes of all aligned 2048-sample sub-windows (10 offsets) of
    every series for each candidate channel, then counts GitHub-row matches.
    Establishes which channel the GitHub release used and whether train and
    test windows interleave within single 20480-sample series (the smoking
    gun for a window-level random split).
    """
    root = HIT["paths"]["github_release"]
    gh_rows = []
    for stem in ("xtrain_1", "xtrain_2", "xtrain_3", "xtrain_4", "xtest"):
        a = sio.loadmat(str(root / f"{stem}.mat"))[stem]
        gh_rows.append((stem, a))

    for ch in range(6):
        index: dict[bytes, tuple] = {}
        for sname, arr in sessions.items():
            sig = arr[:, ch, :]
            for si in range(sig.shape[0]):
                for off in range(0, SERIES_LEN, WIN):
                    h = _row_hash(sig[si, off:off + WIN])
                    index[h] = (sname, si, off)
        hits = {}
        per_series_split: dict[tuple, set] = {}
        n_hit = 0
        for stem, a in gh_rows:
            for i in range(a.shape[0]):
                loc = index.get(_row_hash(a[i]))
                if loc is not None:
                    n_hit += 1
                    per_series_split.setdefault(loc[:2], set()).add(
                        "test" if stem == "xtest" else "train")
            hits[stem] = n_hit
        if n_hit:
            mixed = sum(1 for v in per_series_split.values() if len(v) > 1)
            report["provenance"] = {
                "matched_channel_index0": ch,
                "n_github_windows_matched": n_hit,
                "n_github_windows_total": sum(a.shape[0] for _, a in gh_rows),
                "n_source_series_touched": len(per_series_split),
                "n_series_feeding_both_train_and_test": mixed,
            }
            return
    report["provenance"] = {"matched_channel_index0": None,
                            "n_github_windows_matched": 0}


def audit_hit_full() -> tuple[list[dict], list[dict], dict]:
    """Manifest rows from the Drive full release; one row per
    (session, speed-group) recording."""
    root = HIT["paths"]["gdrive_full"]
    rate = HIT["sampling_rate_hz"]
    rows, integrity = [], []
    session_stats: dict = {}
    sessions_arrays: dict[str, np.ndarray] = {}

    for sname, meta in HIT["sessions"].items():
        path = root / f"{sname}.npy"
        arr = np.load(path, mmap_mode="r")
        if arr.ndim != 3 or arr.shape[1] != 8 or arr.shape[2] != SERIES_LEN:
            raise AssertionError(f"{path}: unexpected shape {arr.shape}")
        sessions_arrays[sname] = arr

        # label row: label value at sample 0, zero-padded elsewhere
        series_labels = np.asarray(arr[:, 7, 0]).astype(int)
        bad = set(np.unique(series_labels).tolist()) - {meta["label"]}
        if bad:
            raise AssertionError(
                f"{sname}: labels {bad} contradict documented "
                f"state {meta['label']}")

        # speed row: LP speed at sample 0, HP speed at sample 1
        lp = np.asarray(arr[:, 6, 0], dtype=float)
        hp = np.asarray(arr[:, 6, 1], dtype=float)
        pairs = list(zip(lp.tolist(), hp.tolist()))

        # recordings = contiguous runs of a constant (LP, HP) pair
        rec_bounds = [0]
        for i in range(1, len(pairs)):
            if pairs[i] != pairs[i - 1]:
                rec_bounds.append(i)
        rec_bounds.append(len(pairs))

        session_stats[sname] = {
            "file": f"data/raw_hit/gdrive_full/HIT-dataset/{sname}.npy",
            "sha256": sha256_file(path),
            "dtype": str(arr.dtype),
            "n_series": int(arr.shape[0]),
            "n_unique_speed_pairs": len(set(pairs)),
            "n_contiguous_speed_blocks": len(rec_bounds) - 1,
            "speed_pairs": sorted(set(pairs)),
        }

        for k in range(len(rec_bounds) - 1):
            lo, hi = rec_bounds[k], rec_bounds[k + 1]
            block = np.asarray(arr[lo:hi, :6, :], dtype=np.float64)
            n_samples = int((hi - lo) * SERIES_LEN)
            lp_k, hp_k = pairs[lo]
            rec_id = f"hit_{sname}_rec{k:02d}"

            # first casing acceleration channel, series concatenated
            checks = signal_checks(block[:, 2, :].ravel(),
                                   expected_min_len=rate)
            integrity.append({
                "dataset": "HIT",
                "file": f"{sname}.npy[{lo}:{hi}]",
                "sha256": None,
                "payload_sha256": sha256_array(block),
                **{f"sig_{kk}": vv for kk, vv in checks.items()},
            })

            rows.append({
                "dataset": "HIT",
                "recording_id": rec_id,
                "group_id_candidate": rec_id,
                "physical_bearing_id": f"hit_bearing_{sname}",
                "experiment_id": f"hit_{sname}_lp{lp_k:g}_hp{hp_k:g}",
                "session_id": f"hit_{sname}",
                "original_file": f"data/raw_hit/gdrive_full/HIT-dataset/{sname}.npy",
                "original_label": meta["label"],
                "mapped_label_candidate": {0: "Healthy", 1: "InnerRace",
                                           2: "OuterRace"}[meta["label"]],
                "sampling_rate_hz": rate,
                "duration_seconds": n_samples / rate,
                "n_samples": n_samples,
                "rpm": lp_k,
                "load": None,
                "sensor_channel": "disp1+disp2+acc1+acc2+acc3+acc4",
                "fault_type": {0: "healthy", 1: "inner_race",
                               2: "outer_race"}[meta["label"]],
                "fault_severity": (None if meta["label"] == 0 else
                                   f"depth {meta['fault_depth_mm']} mm, "
                                   f"length {meta['fault_len_mm']} mm"),
                "bearing_position": "inter_shaft",
                "source_url": HIT["source_url"],
                "metadata_confidence": "derived",
                "notes": (f"{hi - lo} series of {SERIES_LEN}; LP {lp_k:g} "
                          f"/ HP {hp_k:g} r/min; rpm column = LP speed; "
                          "re-assembled from segmented series, contiguity "
                          "after invalid-data removal not guaranteed"),
            })

    gh_report = audit_hit_github()
    _provenance_check(gh_report, sessions_arrays)
    gh_report["sessions"] = session_stats
    return rows, integrity, gh_report
