# Curated result provenance

Files were copied without numerical recalculation from the research workspace.

- `tables/classification_*`, `paired_s0_s1.csv`, `per_class_*`, `roc_auc_*`, `statistical_analysis.csv`, `reconstruction_*`, `run_inventory.csv`, `class_mapping.csv`, `macro_f1_consistency.csv`: `results/100pct_final_analysis/tables/`.
- `tables/lightweight_macro3_*`: `results/lightweight_macro3_reaggregation/tables/`.
- `tables/cwru_*` and `four_dataset_reconstruction_summary.csv`: `results/cwru_reconstruction_analysis/tables/`.
- `tables/Q8_K1_COMPARISON.csv`, `PTQ_TABLE.csv`: `results/methodology_v2/part6_compression/ptq/`.
- `summaries/LATENCY_REPORT.md`, `latency_summary.csv`: `results/methodology_v2/part6_compression/latency/` (hostname redacted only; numerical content retained).
- `summaries/100pct_FINAL_REPORT.md`: `results/100pct_final_analysis/FINAL_REPORT.md`.
- `summaries/lightweight_macro3_README.md`: `results/lightweight_macro3_reaggregation/README.md`.
- `summaries/cwru_reconstruction_README.md`: `results/cwru_reconstruction_analysis/README.md`.
- aggregate confusion-matrix/ROC figures: `results/100pct_final_analysis/figures/`.
- `figures/cwru_reconstruction_example.png`: `results/cwru_reconstruction_analysis/figures/cwru_reconstruction/`.

The S0/S1/K1/Q8 models were trained/evaluated in the executed four-domain protocol. Macro-3 is a post-hoc three-dataset reporting aggregate.

## Reduced-label recovery

Exact public execution paths now exist for 1%, 5%, 10% and 100%. The recovered launchers are `scripts/run_methodology_v2_1pct_extension.py`, `scripts/run_methodology_v2_5pct.py`, and `scripts/run_methodology_v2_10pct.py`; 100% uses `scripts/methodology_v2/experiment_executor.py`. Each low-label path covers 18 runs (folds 1–3, seeds 42/1337/2026, S0/S1), trains CWRU, JNU, HIT and MaFaulDa, and selects by four-domain validation MacroDomainF1. Principal Macro-3 reporting averages JNU, HIT and MaFaulDa.

Original compact classification/AUC summaries are under `tables/low_label/`; original nine-cell pairing proofs are under `../methodology_v2/low_label_provenance/`. The 1% path is a reduced-label extension executed after the primary registered grid. Historical and publication hashes and all portability edits are recorded in `docs/low_label_provenance.md`. No values were recomputed and no heavy artifacts were included.
