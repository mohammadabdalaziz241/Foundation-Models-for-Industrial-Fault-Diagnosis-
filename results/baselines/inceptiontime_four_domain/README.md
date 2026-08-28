# Four-domain InceptionTime baseline

This folder contains compact public evidence for the completed supervised
InceptionTime-style waveform baseline used in the final dissertation. The
experiment comprises nine matched fold-seed cells: folds 1--3 crossed with
seeds 42, 1337, and 2026. Every cell used the full labelled TRAIN subset for
CWRU, JNU, HIT, and MaFaulDa under the same frozen leakage-safe partitions as
the PC-STE downstream study.

The selected checkpoint in every cell was the strict maximum four-domain
validation MacroDomainF1 checkpoint. TEST remained sealed until selection and
was evaluated once. Macro-4 is the equal mean of the four dataset-level TEST
Macro-F1 values.

This is a complete-method comparison, not an architecture-only ablation.
InceptionTime consumes native-length raw one-second waveforms; Full S1 uses its
log-STFT/N2 spectrogram representation and sealed SSL initialization.

## Final results

- InceptionTime: **0.2816 +/- 0.0442** Macro-4 (mean +/- sample SD).
- Full S1: **0.7708 +/- 0.0344**; paired difference **+0.4891**;
  exact two-sided sign-flip **p = 0.00390625**; Full S1 wins 9/9 cells.
- K1 (secondary): **0.7931 +/- 0.0288**; paired difference **+0.5115**;
  exact two-sided sign-flip **p = 0.00390625**; K1 wins 9/9 cells.

Training cost was 91.4850857413889 summed GPU-hours across nine successful
runs (mean 10.165009526821 hours/run), using one NVIDIA RTX 4000 Ada Generation
GPU per run. Adding these runs to the previously audited programme gives 99
successful training runs and 661.011477839445 summed GPU-hours.

## Evidence files

- `baseline_spec.json`: portable frozen protocol summary.
- `per_cell_results.csv`: four dataset Macro-F1 values and recomputed Macro-4.
- `paired_macro4.csv`: strict fold-seed pairing with Full S1 and K1.
- `dataset_summary.csv`: dataset-level descriptive comparison with Full S1.
- `aggregate_summary.csv`: aggregate and exact paired-test results.
- `training_cost.csv`: explicit state start/completion timing provenance.

All values were derived read-only from the completed authoritative run outputs.
Checkpoints, prediction files, full run trees, caches, node-local logs, and raw
datasets are intentionally not redistributed.
