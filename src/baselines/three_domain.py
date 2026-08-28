"""Fail-closed sampling, data access, losses, and metrics for the baseline."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)

from .inceptiontime import CLASS_ORDERS, DATASETS

EXPECTED_LENGTHS = {"CWRU": 48_000, "JNU": 50_000, "HIT": 25_000,
                    "MAFAULDA": 50_000}
LABEL_FIELD = {"CWRU": "fault_type", "JNU": "original_label",
               "HIT": "original_label", "MAFAULDA": "original_label"}


class FourDomainSampler:
    """TRAIN-only dataset -> class -> group -> window sampler."""

    def __init__(self, train: pd.DataFrame, seed: int) -> None:
        if set(train["split"].astype(str).str.lower()) != {"train"}:
            raise AssertionError("sampler input must contain TRAIN only")
        seen = set(train["dataset"])
        if seen != set(DATASETS):
            raise AssertionError(f"sampler datasets must be exactly {DATASETS}: {seen}")
        self._tree: dict[str, dict[str, dict[str, list[str]]]] = {}
        for ds in DATASETS:
            self._tree[ds] = {}
            dsub = train[train.dataset == ds]
            labels = dsub[LABEL_FIELD[ds]].astype(str)
            if set(labels) != set(CLASS_ORDERS[ds]):
                raise AssertionError(f"{ds} label set differs from frozen order")
            for label in CLASS_ORDERS[ds]:
                csub = dsub[labels == label]
                self._tree[ds][label] = {
                    str(g): sorted(grp.window_id.astype(str).tolist())
                    for g, grp in csub.groupby("group_id")
                }
                if not self._tree[ds][label]:
                    raise AssertionError(f"empty class {ds}/{label}")
        self._cycles = {ds: 0 for ds in DATASETS}
        self._step = 0
        self._rng = np.random.default_rng(seed)

    def next_batch(self) -> list[tuple[str, str, str]]:
        batch = []
        for ds in DATASETS:
            count = 16
            classes = CLASS_ORDERS[ds]
            for _ in range(count):
                label = classes[self._cycles[ds] % len(classes)]
                self._cycles[ds] += 1
                groups = self._tree[ds][label]
                gids = sorted(groups)
                gid = gids[int(self._rng.integers(len(gids)))]
                windows = groups[gid]
                wid = windows[int(self._rng.integers(len(windows)))]
                batch.append((ds, label, wid))
        self._step += 1
        return batch

    def state_dict(self) -> dict:
        return {"cycles": self._cycles.copy(), "step": self._step,
                "rng": self._rng.bit_generator.state}

    def load_state_dict(self, state: dict) -> None:
        self._cycles = {k: int(v) for k, v in state["cycles"].items()}
        self._step = int(state["step"])
        self._rng.bit_generator.state = state["rng"]


class GuardedWindowAccess:
    """Makes TEST waveform access structurally conditional on a seal file."""

    def __init__(self, manifest: pd.DataFrame, seal_path: Path,
                 reader: Callable, smoke: bool = False) -> None:
        self.manifest = manifest.set_index("window_id", drop=False)
        self.seal_path = Path(seal_path)
        self.reader = reader
        self.smoke = smoke

    def read(self, window_id: str, allowed_split: str) -> np.ndarray:
        row = self.manifest.loc[window_id]
        split = str(row["split"]).lower()
        if split != allowed_split.lower():
            raise AssertionError(f"{window_id}: requested {allowed_split}, is {split}")
        if row["dataset"] not in DATASETS:
            raise AssertionError("forbidden/unexpected dataset")
        if split == "test" and (self.smoke or not self.seal_path.is_file()):
            raise RuntimeError("TEST is sealed until test_seal.json exists")
        x = self.reader(row)
        expected = EXPECTED_LENGTHS[str(row["dataset"])]
        if len(x) != expected:
            raise AssertionError(f"{window_id}: native length {len(x)} != {expected}")
        return np.asarray(x, dtype=np.float32)


def validate_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    """Validate metadata without reading any waveform, then return baseline view."""
    required = {"dataset", "split", "window_id", "group_id", "original_label",
                "native_sampling_rate_hz", "window_duration_seconds",
                "start_sample", "end_sample"}
    if not required <= set(manifest.columns):
        raise AssertionError(f"manifest missing columns: {required-set(manifest.columns)}")
    view = manifest[manifest.dataset.isin(DATASETS)].copy()
    if set(view.dataset) != set(DATASETS):
        raise AssertionError("selected baseline view lacks an expected dataset")
    if not np.allclose(view.window_duration_seconds.astype(float), 1.0):
        raise AssertionError("all selected windows must be exactly one second")
    lengths = view.end_sample.astype(int) - view.start_sample.astype(int)
    expected = view.dataset.map(EXPECTED_LENGTHS).astype(int)
    if not (lengths == expected).all():
        raise AssertionError("manifest contains non-native/non-one-second lengths")
    split_sets = {s: set(view.loc[view.split.str.lower() == s, "window_id"])
                  for s in ("train", "validation", "test")}
    if any(split_sets[a] & split_sets[b] for a, b in
           (("train", "validation"), ("train", "test"),
            ("validation", "test"))):
        raise AssertionError("TRAIN/VALIDATION/TEST window IDs overlap")
    if set(view.split.str.lower()) != set(split_sets):
        raise AssertionError("unexpected split in baseline view")
    return view


def microbatch_objective(logit_chunks: Iterable[torch.Tensor],
                         target_chunks: Iterable[torch.Tensor],
                         dataset_size: int) -> torch.Tensor:
    """Exact dataset contribution: (1/4) * sum CE / N_d."""
    chunks = [torch.nn.functional.cross_entropy(z, y, reduction="sum")
              for z, y in zip(logit_chunks, target_chunks)]
    return sum(chunks) / (4.0 * dataset_size)


def classification_metrics(y_true: np.ndarray, probs: np.ndarray,
                           n_classes: int) -> dict[str, float]:
    labels = np.arange(n_classes)
    pred = probs.argmax(axis=1)
    onehot = np.eye(n_classes, dtype=np.int8)[y_true]
    try:
        auc = float(roc_auc_score(onehot, probs, average="macro",
                                  multi_class="ovr"))
    except ValueError:
        auc = float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "macro_precision": float(precision_score(y_true, pred, labels=labels,
                                                  average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, pred, labels=labels,
                                            average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, pred, labels=labels,
                                    average="macro", zero_division=0)),
        "macro_roc_auc_ovr": auc,
    }


def macro4(per_dataset: dict[str, dict], key: str) -> float:
    if set(per_dataset) != set(DATASETS):
        raise AssertionError(f"Macro-4 requires exactly {DATASETS}")
    return float(np.mean([per_dataset[ds][key] for ds in DATASETS]))


# Historical import compatibility only; corrected reports never use Macro-3.
ThreeDomainSampler = FourDomainSampler
macro3 = macro4
