# Curated result provenance

The publication repository preserves compact dissertation evidence without redistributing raw datasets, checkpoints or complete execution trees. Existing PC-STE result files listed below were copied without numerical recalculation from the research workspace; the controlled InceptionTime evidence was generated read-only from the already completed authoritative baseline outputs, without retraining or another TEST evaluation.

## PC-STE curated outputs

- `tables/classification_*`, `paired_s0_s1.csv`, `per_class_*`, `roc_auc_*`, `statistical_analysis.csv`, `reconstruction_*`, `run_inventory.csv`, `class_mapping.csv`, `macro_f1_consistency.csv`: `results/100pct_final_analysis/tables/` in the research workspace.
- `tables/lightweight_macro3_*`: `results/lightweight_macro3_reaggregation/tables/`.
- `tables/cwru_*` and `four_dataset_reconstruction_summary.csv`: `results/cwru_reconstruction_analysis/tables/`.
- `tables/Q8_K1_COMPARISON.csv`, `PTQ_TABLE.csv`: `results/methodology_v2/part6_compression/ptq/`.
- `summaries/LATENCY_REPORT.md`, `latency_summary.csv`: final four-domain dissertation-facing summaries derived from `results/methodology_v2/part6_compression/latency_four_domain/`.
- `summaries/LATENCY_REPORT_THREE_DOMAIN_HISTORICAL.md`, `latency_summary_three_domain_historical.csv`: preserved historical three-domain latency summaries.
- `summaries/100pct_FINAL_REPORT.md`: `results/100pct_final_analysis/FINAL_REPORT.md`.
- `summaries/lightweight_macro3_README.md`: `results/lightweight_macro3_reaggregation/README.md`.
- `summaries/cwru_reconstruction_README.md`: `results/cwru_reconstruction_analysis/README.md`.
- aggregate confusion-matrix/ROC figures: `results/100pct_final_analysis/figures/`.
- `figures/cwru_reconstruction_example.png`: `results/cwru_reconstruction_analysis/figures/cwru_reconstruction/`.

The S0/S1/K1/Q8 models were trained and evaluated under the executed four-domain protocol. The dissertation reports Macro-4 over CWRU, JNU, HIT and MaFaulDa. Macro-3, a three-dataset aggregate excluding CWRU, is retained as a historical secondary reporting view derived from the same per-dataset four-domain results.

## Four-domain efficiency evidence

The **primary dissertation efficiency evidence** is `methodology_v2/part6_compression/latency_four_domain/`, with the independent audit at `methodology_v2/verification/four_domain_verification.md`. These are authoritative for dissertation Tables 5.10 and 5.11 and their discussion.

The models were already trained under the four-domain protocol. The earlier `methodology_v2/part6_compression/latency/` benchmark aggregated runtime over JNU, HIT and MaFaulDa and excluded CWRU; it is retained, unchanged, as **historical three-domain latency provenance**. The final dissertation benchmark remeasured all four datasets in one controlled session rather than splicing CWRU into the earlier timings.

## Controlled InceptionTime baseline

The public baseline evidence is under `baselines/inceptiontime_four_domain/` and was derived read-only from the completed nine-cell baseline run outputs. No model was retrained and the sealed TEST partition was not evaluated again for publication packaging.

- `baseline_spec.json` records the portable frozen architecture/training/evaluation protocol.
- `per_cell_results.csv` contains the nine fold–seed TEST Macro-F1 results for CWRU, JNU, HIT and MaFaulDa together with Macro-4.
- `paired_macro4.csv` records the strict fold–seed pairing against Full S1 and K1.
- `dataset_summary.csv` contains dataset-level descriptive summaries.
- `aggregate_summary.csv` contains aggregate means, sample standard deviations, paired differences, exact two-sided sign-flip p-values and win counts.
- `training_cost.csv` contains explicit baseline run-state start/completion timing provenance.
- `README.md` states the interpretation and publication boundary.

The frozen baseline evidence reproduces the dissertation values: InceptionTime `0.2816 ± 0.0442` Macro-4; Full S1 `0.7708 ± 0.0344`, paired difference `+0.4891`, exact two-sided `p=0.00390625`, 9/9 wins; and K1 `0.7931 ± 0.0288`, paired difference `+0.5115`, exact two-sided `p=0.00390625`, 9/9 wins. InceptionTime dataset means are CWRU `0.1440 ± 0.0823`, JNU `0.2835 ± 0.1083`, HIT `0.4762 ± 0.2233`, and MaFaulDa `0.2228 ± 0.0817`.

Baseline timing evidence gives nine successful runs, mean `10.17` hours/run and `91.49` summed GPU-hours, with one NVIDIA RTX 4000 Ada Generation GPU per run. Combined with the previously audited programme, the dissertation reports **99 successful training runs** and approximately **661.01 GPU-hours**.

The baseline is a complete-method comparison rather than an architecture-only ablation: its model consumes native-length raw one-second waveforms while Full S1 uses the log-STFT/N2 PC-STE representation after SSL pretraining.

## Reduced-label recovery

Exact public execution paths exist for 1%, 5%, 10% and 100%. The recovered launchers are `scripts/run_methodology_v2_1pct_extension.py`, `scripts/run_methodology_v2_5pct.py`, and `scripts/run_methodology_v2_10pct.py`; 100% uses `scripts/methodology_v2/experiment_executor.py`. Each low-label path covers 18 runs (folds 1–3, seeds 42/1337/2026, S0/S1), trains CWRU, JNU, HIT and MaFaulDa, and selects by four-domain validation MacroDomainF1. Principal dissertation reporting uses Macro-4, averaging CWRU, JNU, HIT and MaFaulDa. Historical Macro-3 summaries average JNU, HIT and MaFaulDa only.

Original compact classification/AUC summaries are under `tables/low_label/`; original nine-cell pairing proofs are under `../methodology_v2/low_label_provenance/`. The 1% path was executed after the other three label fractions and uses the identical protocol. Its historical launcher and directory names are retained unchanged so that recorded paths and hashes remain valid. Historical and publication hashes and all portability edits are recorded in `docs/low_label_provenance.md`. No low-label numerical result was recomputed for the publication repository.

## Publication exclusions

Raw vibration recordings, trained checkpoints, full predictions, probability caches, teacher caches, complete experiment run trees, SLURM/node-local logs, temporary files and secrets are intentionally excluded. These exclusions preserve dataset-distribution terms and keep the public release compact without changing the scientific results or frozen partition evidence.
