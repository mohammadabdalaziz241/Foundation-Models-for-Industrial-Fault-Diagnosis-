#!/usr/bin/env python3
"""Recompute and verify the compact public InceptionTime evidence."""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
EVIDENCE = REPO / "results/baselines/inceptiontime_four_domain"
DATASETS = ("CWRU", "JNU", "HIT", "MAFAULDA")
EXPECTED_CELLS = {(fold, seed) for fold in (1, 2, 3)
                  for seed in (42, 1337, 2026)}


def exact_two_sided_sign_flip(differences: np.ndarray) -> float:
    """Enumerate all signs using absolute mean difference as the statistic."""
    observed = abs(float(np.mean(differences)))
    extreme = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(differences)):
        statistic = abs(float(np.mean(differences * np.asarray(signs))))
        extreme += statistic >= observed - 1e-15
    return extreme / (2 ** len(differences))


def main() -> None:
    cells = pd.read_csv(EVIDENCE / "per_cell_results.csv")
    paired = pd.read_csv(EVIDENCE / "paired_macro4.csv")
    cost = pd.read_csv(EVIDENCE / "training_cost.csv")
    keys = set(zip(cells.fold, cells.seed))
    if keys != EXPECTED_CELLS or len(cells) != 9:
        raise AssertionError("expected exactly the nine registered fold-seed cells")
    if set(zip(paired.fold, paired.seed)) != EXPECTED_CELLS:
        raise AssertionError("paired table does not contain the registered cells")

    dataset_columns = [f"{dataset}_macro_f1" for dataset in DATASETS]
    recomputed = cells[dataset_columns].mean(axis=1)
    np.testing.assert_allclose(recomputed, cells.macro4_f1, rtol=0, atol=1e-12)
    np.testing.assert_allclose(cells.macro4_f1, paired.inceptiontime_macro4,
                               rtol=0, atol=1e-12)

    for model in ("full_s1", "k1"):
        delta = paired[f"delta_{model}_minus_inceptiontime"].to_numpy()
        np.testing.assert_allclose(
            delta,
            paired[f"{model}_macro4"] - paired.inceptiontime_macro4,
            rtol=0, atol=1e-12,
        )
        print(f"{model}: mean={paired[f'{model}_macro4'].mean():.15f} "
              f"delta={delta.mean():.15f} "
              f"p={exact_two_sided_sign_flip(delta):.8f} "
              f"wins={int(np.sum(delta > 0))}/9")

    durations = cost.duration_hours.to_numpy()
    print(f"InceptionTime: mean={cells.macro4_f1.mean():.15f} "
          f"sample_sd={cells.macro4_f1.std(ddof=1):.15f}")
    print(f"cost: n={len(durations)} mean={durations.mean():.15f} "
          f"sample_sd={durations.std(ddof=1):.15f} "
          f"total={durations.sum():.15f}")


if __name__ == "__main__":
    main()
