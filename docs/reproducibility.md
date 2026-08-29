# Reproducibility

## Fixed PC-STE experimental design

- Python target: 3.12; frozen packages: `requirements_frozen.txt`.
- Datasets: CWRU, JNU, HIT, MaFaulDa.
- Folds: 1, 2, 3. Seeds: 42, 1337, 2026.
- Window: 1.000 s at native rate; TRAIN stride 0.5 s; validation/TEST stride 1.0 s; no cross-dataset resampling.
- Channels: CWRU DE, JNU vertical accelerometer, HIT ch3, MaFaulDa underhang radial column 3.
- STFT: periodic Hann, `center=False`, no padding, one-sided real FFT. CWRU/JNU/MaFaulDa use `n_fft=1024`, hop 256; HIT uses `n_fft=512`, hop 128.
- Transform: `log1p(abs(STFT(x)))`.
- N2: independently for each fold, dataset and frequency bin, fitted only on TRAIN frames in float64; `(X-mu)/std`, with denominator exactly 1 where raw std is below `1e-6`; validation and TEST reuse frozen statistics.
- Patches: 16 frequency bins × 8 time frames.
- PC-STE: stem channels 8; `d_model=192`; four Mamba-1 temporal blocks; two directions fused by mean; absolute Fourier Hz/seconds coordinates; Hz-gated cross-band mixer; validity-masked mean pooling. Encoder parameters: 2,382,033.
- SSL: 60 epochs, 60% random patch masking, reconstruction in normalized log-STFT space.
- Downstream: 50 epochs, no early stopping. S0 random initialization; S1 matched best SSL initialization; encoder and heads fully trained.
- Dissertation label regimes: **1%, 5%, 10%, and 100% only**.
- AdamW: lr `3e-4`, betas `(0.9,0.95)`, eps `1e-8`, weight decay `0.05`, global gradient clip 1.0, 5 warm-up epochs, cosine decay to `1e-6`.
- Sampling: effective batch 64 = 16 windows per dataset, with replacement; dataset→class→group→window supervised hierarchy. Fold-specific steps/epoch: 202, 205, 201.
- Downstream selector: maximum four-domain validation MacroDomainF1, strict improvement, earlier epoch on exact ties.
- TEST: checkpoint sealed first; one TEST evaluation; no TEST-driven selection.
- Dissertation aggregate: **Macro-4**, the equal mean of CWRU, JNU, HIT and MaFaulDa dataset-level Macro-F1.
- K1: 1,375,953-parameter unidirectional encoder, initialized by retaining the forward direction of all four S1 blocks; CE + KL plus relational mixer-attention KL.

## Controlled InceptionTime baseline

The dissertation's external baseline is a supervised InceptionTime-style network evaluated at **100% labels only**. It is a complete-method comparison rather than an architecture-only ablation.

- Input: the same frozen native-length one-second waveform windows used by the downstream study; no STFT and no N2 transformation.
- Architecture: six Inception modules with residual shortcuts after modules 3 and 6, bottleneck dimension 32, 32 filters per convolution branch, kernel sizes 40/20/10, global average pooling, and dataset-specific heads for CWRU/JNU/HIT/MaFaulDa with 3/4/3/10 classes.
- Design: folds 1–3 × seeds 42/1337/2026 = nine matched cells.
- Training: 50 epochs, effective batch 64 with 16 examples per dataset, fold-specific step counts 202/205/201, and the same AdamW learning-rate schedule used for PC-STE downstream training.
- Selection: strict maximum of equally weighted four-domain validation MacroDomainF1.
- TEST: one sealed evaluation after checkpoint selection.
- Primary comparison: Full S1 vs InceptionTime by matched fold and seed using Macro-4 and the exact two-sided paired sign-flip test over all `2^9 = 512` sign assignments. K1 vs InceptionTime is the secondary comparison.

The portable baseline protocol is `results/baselines/inceptiontime_four_domain/baseline_spec.json`. Implementation and public entry points are in `src/baselines/` and `scripts/baselines/`.

The compact public evidence is under `results/baselines/inceptiontime_four_domain/`. It records InceptionTime Macro-4 `0.2816 ± 0.0442`; Full S1 `0.7708 ± 0.0344` with paired difference `+0.4891`, 9/9 wins and exact two-sided `p=0.00390625`; and K1 `0.7931 ± 0.0288` with paired difference `+0.5115`, 9/9 wins and the same exact p-value.

## Training-cost audit

The final dissertation programme contains **99 successful training executions** and approximately **661.01 summed GPU-hours**. Every reported training execution used one GPU, so summed successful-run wall-clock duration is used as the GPU-hour approximation.

| Stage | Runs | Mean h/run | Total h |
|---|---:|---:|---:|
| SSL pretraining | 9 | 8.51 | 76.56 |
| S0 full-label downstream | 9 | 6.48 | 58.34 |
| S1 full-label downstream | 9 | 6.48 | 58.34 |
| Reduced-label S0/S1 downstream | 54 | 6.58 | 355.42 |
| K1 distillation | 9 | 2.32 | 20.86 |
| InceptionTime baseline | 9 | 10.17 | 91.49 |
| **Total** | **99** | — | **661.01** |

## Roots

`PCSTE_DATA_ROOT` defaults to `./data`; `PCSTE_RESULTS_ROOT` defaults to `./results`. Optional distributed scheduling uses neutral worker names from `PCSTE_WORKER_HOSTS`.

## Frozen artifacts

`methodology_v2/part1_audit` through `part6_compression` include the dataset audit, splits, window manifests, N2 normalizers, architecture/SSL/experiment/compression specifications, registries and hashes. The final dissertation protocol is `configs/dissertation/experiment_protocol.yaml` and lists 1%, 5%, 10%, and 100% labelled conditions.

## Label-efficiency execution paths

The publication release contains the execution paths used for all dissertation-facing label fractions:

- 1%: `scripts/run_methodology_v2_1pct_extension.py`
- 5%: `scripts/run_methodology_v2_5pct.py`
- 10%: `scripts/run_methodology_v2_10pct.py`
- 100%: `scripts/methodology_v2/experiment_executor.py`

All four public execution paths preserve the four-domain supervised protocol: CWRU, JNU, HIT and MaFaulDa contribute supervised loss and encoder gradients through four dataset-specific heads; checkpoints are selected by four-domain validation MacroDomainF1; and final TEST aggregation uses Macro-4 over all four datasets.

The reduced-label dissertation-facing summary is `results/tables/low_label/reduced_label_classification_summary.csv`. S0/S1 subset and stream pairing proofs are under `methodology_v2/low_label_provenance/`.

Raw datasets, model checkpoints, full predictions, probability caches, teacher caches, full experiment run trees and node-local logs are deliberately not included. Artifact-dependent integration tests therefore require locally regenerated or separately preserved artifacts, but the publication repository contains the code paths, final specifications, manifests, compact result summaries and provenance needed to inspect and rerun the reported study after obtaining the third-party datasets.
