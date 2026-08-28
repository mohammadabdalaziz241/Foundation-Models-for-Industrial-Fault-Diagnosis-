# methodology_v2 — dissertation experiment

This directory contains the complete Methodology V2 provenance and supporting artefacts for the dissertation experiment built around the **PC-STE (Physical-Coordinate Spectro-Temporal Encoder)**.

## Dissertation scope

The executed study uses four rotating-machinery datasets:

- **CWRU**
- **JNU**
- **HIT**
- **MaFaulDa**

Masked-reconstruction self-supervised learning (SSL) uses all four datasets. The downstream S0/S1 experiment also uses all four datasets with four dataset-specific classification heads and four-domain validation `MacroDomainF1` for checkpoint selection.

The dissertation reports **Macro-4**, the equal mean of the CWRU, JNU, HIT and MaFaulDa dataset-level Macro-F1 scores. **Macro-3**, the equal mean of JNU, HIT and MaFaulDa only, is retained in some historical artefacts as a secondary reporting view from an earlier stage; it does not represent a separate three-domain training protocol.

The final downstream label regimes are exactly:

- **1%**
- **5%**
- **10%**
- **100%**

Each label regime uses two arms, three folds and three seeds (`42`, `1337`, `2026`), giving **18 runs** and **9 paired fold–seed cells** per regime:

- **S0** — supervised training from random initialisation
- **S1** — SSL initialisation followed by full fine-tuning

The dissertation-facing protocol is defined in `configs/dissertation/experiment_protocol.yaml`.

## Main pipeline

The implemented pipeline is:

`recordings -> leakage-aware splits -> 1 s windows -> native-rate STFT -> log magnitude -> N2 normalisation -> spectrogram patches -> physical coordinate encoding -> 4 x BiMamba -> valid-time pooling -> Hz-aware cross-band mixing -> valid-band pooling -> 192-D embedding`

The same encoder is used for masked-reconstruction SSL and downstream classification.

## Directory layout

- `part1_audit/` — dataset census, recording-level manifests, grouping policy, integrity checks and reproducibility metadata.
- `part2_splits/` — sealed leakage-aware global folds, test identities, split statistics and hashes.
- `part3_input_design/` — channel, sampling-rate and window-duration studies.
- `part3_windows/` — frozen one-second window manifests and dataset-specific guards.
- `part4_stft_design/` — STFT design study and physical-resolution checks.
- `part4_representation_freeze/` — representation and normalisation comparison study.
- `part4_representation_final/` — frozen dissertation representation, N2 normalisers and verification artefacts.
- `part5_architecture_design/` — PC-STE architecture-design and novelty analysis.
- `part5_encoder/` — frozen encoder specification, parameter/compute audits and selective-scan parity checks.
- `part5_ssl_design/` — masked-reconstruction SSL design and supporting studies.
- `part5_experiment_registry/` — sealed historical planning registry and provenance for the downstream study.
- `low_label_provenance/` — pairing proofs for the 1%, 5% and 10% reduced-label experiments.
- `part6_compression/` — lightweight PC-STE study, including the K1 student, knowledge distillation, PTQ/Q8 specifications, statistics and execution provenance.

PC-STE implementation code lives under `src/methodology_v2/`, experiment and analysis entry points under `scripts/`, and verification tests under `tests/methodology_v2/`.

The final dissertation additionally contains a controlled full-label four-domain **InceptionTime** comparison. That baseline is intentionally kept outside this historical Methodology V2 provenance directory: its implementation is under `src/baselines/`, execution/analysis tools under `scripts/baselines/`, tests under `tests/baselines/`, and compact result/protocol evidence under `results/baselines/inceptiontime_four_domain/`. It uses the same frozen downstream partitions and is reported as a complete-method comparison, not an architecture-only ablation.

Curated dissertation-facing result tables, summaries and figures are under `results/`.

## Historical sealed artefacts

Some sealed planning files intentionally preserve superseded candidate conditions or historical names so that their hashes and provenance remain verifiable. In particular, the original Part 5D planning registry contains candidate **25%** and **50%** label fractions that were **not executed and are not reported as dissertation experiments**. They are retained only as historical provenance.

Similarly, files whose names contain `macro3`, `posthoc`, `extension`, or other historical terminology may be retained unchanged when renaming them would break recorded paths, imports or hashes. The dissertation-facing configuration and documentation define the authoritative interpretation.

## Status

The main Methodology V2 pipeline, four-domain S0/S1 experiments, reduced-label evaluations, lightweight K1 study, Q8 evaluation, latency analysis and curated result export are complete and represented in this repository. The final controlled InceptionTime baseline is also complete and represented in the separate baseline paths identified above. Historical development artefacts are preserved where necessary for reproducibility rather than rewritten after the fact.

Raw datasets, large checkpoints and machine-local execution directories are not committed to the repository; see the root documentation and `results/PROVENANCE.md` for reproducibility and result provenance details.
