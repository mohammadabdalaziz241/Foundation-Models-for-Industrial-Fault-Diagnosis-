"""Recording-manifest schema and loud validation for methodology_v2 Part 1.

One manifest row == one ORIGINAL recording / acquisition sequence, before any
windowing. Windows are never a manifest unit.
"""
from __future__ import annotations

import math

MANIFEST_COLUMNS = [
    "dataset",
    "recording_id",
    "group_id_candidate",
    "physical_bearing_id",
    "experiment_id",
    "session_id",
    "original_file",
    "original_label",
    "mapped_label_candidate",
    "sampling_rate_hz",
    "duration_seconds",
    "n_samples",
    "rpm",
    "load",
    "sensor_channel",
    "fault_type",
    "fault_severity",
    "bearing_position",
    "source_url",
    "metadata_confidence",
    "notes",
]

# Columns that must never be empty/NA.
REQUIRED_NON_NULL = [
    "dataset", "recording_id", "group_id_candidate", "original_file",
    "original_label", "sampling_rate_hz", "n_samples", "duration_seconds",
    "source_url", "metadata_confidence",
]

VALID_CONFIDENCE = {"documented", "derived", "inferred", "unknown"}


class ManifestValidationError(AssertionError):
    """Raised loudly when the recording manifest violates an invariant."""


def _fail(msg: str) -> None:
    raise ManifestValidationError(msg)


def validate_manifest(df, valid_labels_by_dataset: dict[str, set]) -> None:
    """Validate the recording manifest DataFrame. Raises on first violation."""
    missing_cols = [c for c in MANIFEST_COLUMNS if c not in df.columns]
    if missing_cols:
        _fail(f"manifest missing columns: {missing_cols}")

    for col in REQUIRED_NON_NULL:
        bad = df[col].isna()
        if bad.any():
            _fail(f"column '{col}' has {int(bad.sum())} null values, "
                  f"e.g. rows {df.index[bad][:5].tolist()}")

    # every row has a known dataset
    unknown = set(df["dataset"]) - set(valid_labels_by_dataset)
    if unknown:
        _fail(f"unknown dataset values: {unknown}")

    # unique recording identity within each dataset
    dup = df.duplicated(subset=["dataset", "recording_id"])
    if dup.any():
        _fail(f"duplicate (dataset, recording_id) rows: "
              f"{df.loc[dup, ['dataset', 'recording_id']].values[:5].tolist()}")

    # no raw file unexpectedly appears multiple times (per dataset+channel a
    # file may legitimately yield several recordings only for HIT sessions,
    # where one .npy holds many series; those rows share original_file but
    # must differ in recording_id, which the uniqueness check above enforces).
    dup_file = df.duplicated(subset=["dataset", "original_file",
                                     "recording_id"])
    if dup_file.any():
        _fail("duplicated (dataset, original_file, recording_id) rows")

    # numeric sanity
    for col, kind in (("sampling_rate_hz", "rate"),
                      ("n_samples", "length"),
                      ("duration_seconds", "duration")):
        for i, v in df[col].items():
            try:
                fv = float(v)
            except (TypeError, ValueError):
                _fail(f"row {i}: {col}={v!r} is not numeric")
            if not math.isfinite(fv) or fv <= 0:
                _fail(f"row {i}: {col}={v!r} must be finite and positive")

    # n_samples integral
    frac = [i for i, v in df["n_samples"].items()
            if float(v) != int(float(v))]
    if frac:
        _fail(f"n_samples not integral at rows {frac[:5]}")

    # labels valid according to each dataset's registry (string-canonical
    # comparison so CSV round-trips of numeric labels stay valid)
    for i, row in df.iterrows():
        valid = {str(v) for v in valid_labels_by_dataset[row["dataset"]]}
        if str(row["original_label"]) not in valid:
            _fail(f"row {i}: label {row['original_label']!r} not in "
                  f"{row['dataset']} registry")

    # confidence vocabulary
    bad_conf = set(df["metadata_confidence"]) - VALID_CONFIDENCE
    if bad_conf:
        _fail(f"invalid metadata_confidence values: {bad_conf}")

    # duration consistent with n_samples / rate (tolerance 1%)
    for i, row in df.iterrows():
        expect = float(row["n_samples"]) / float(row["sampling_rate_hz"])
        got = float(row["duration_seconds"])
        if abs(expect - got) > max(0.01 * expect, 1e-6):
            _fail(f"row {i}: duration {got} inconsistent with "
                  f"n_samples/rate = {expect}")
