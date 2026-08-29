# Reduced-label execution provenance

The dissertation evaluates exactly **1%, 5%, 10%, and 100%** labelled-data regimes under the same four-domain downstream protocol.

## Execution paths

| Fraction | Publication launcher | Output naming |
|---|---|---|
| 1% | `scripts/run_methodology_v2_1pct_extension.py` | `methodology_v2_1pct_extension/downstream/[s0|s1]_f{fold}_s{seed}_l001` |
| 5% | `scripts/run_methodology_v2_5pct.py` | `methodology_v2_5pct/downstream/[s0|s1]_f{fold}_s{seed}_l005` |
| 10% | `scripts/run_methodology_v2_10pct.py` | `methodology_v2_10pct/downstream/[s0|s1]_f{fold}_s{seed}_l010` |
| 100% | `scripts/methodology_v2/experiment_executor.py` | principal downstream run tree |

The reduced-label launchers preserve the same scientific protocol as the full-label experiment. Portability edits affect path resolution only; dataset definitions, model architecture, initialization, optimizer, scheduler, sampler, loss, validation, checkpoint selection, TEST handling, seeds, folds, and deterministic controls remain unchanged.

## Matched design

For each executed fraction:

- folds are 1, 2, and 3;
- seeds are 42, 1337, and 2026;
- arms are S0 and S1;
- there are 18 runs and 9 matched fold-seed cells;
- downstream training lasts 50 epochs;
- CWRU, JNU, HIT, and MaFaulDa contribute supervised loss and encoder gradients through four dataset-specific heads;
- checkpoints maximize four-domain validation MacroDomainF1 under the strict-improvement rule;
- TEST is evaluated only after checkpoint selection;
- final cross-dataset reporting uses **Macro-4**, the equal mean of CWRU, JNU, HIT, and MaFaulDa dataset-level Macro-F1.

## Public result and pairing evidence

The dissertation-facing reduced-label results are stored in:

- `results/tables/low_label/reduced_label_classification_summary.csv`
- `results/tables/statistical_analysis.csv`

The corresponding S0/S1 subset and batch-stream pairing proofs are stored under `methodology_v2/low_label_provenance/`. Across the 1%, 5%, and 10% regimes, these proofs record matching paired subset hashes and batch-stream hashes for each fold-seed cell.

The reduced-label aggregate values reported in the dissertation are:

| Labels | S0 Macro-4 F1 | S1 Macro-4 F1 | Δ S1−S0 | Exact p |
|---|---:|---:|---:|---:|
| 1% | 0.4803 ± 0.0655 | 0.5051 ± 0.0547 | +0.0249 | 0.3125 |
| 5% | 0.6519 ± 0.0248 | 0.6427 ± 0.0243 | −0.0092 | 0.4492 |
| 10% | 0.7135 ± 0.0425 | 0.7187 ± 0.0388 | +0.0052 | 0.6328 |

These artefacts provide the compact public evidence for the reduced-label experiments while the raw datasets, checkpoints, full prediction caches, and node-local logs remain outside the repository.
