# Reproducibility

## Fixed experimental design

- Python target: 3.12; frozen packages: `requirements_frozen.txt`.
- Folds: 1, 2, 3. Seeds: 42, 1337, 2026.
- Window: 1.000 s at native rate; TRAIN stride 0.5 s; validation/TEST stride 1.0 s; no resampling.
- Channels: CWRU DE, JNU vertical accelerometer, HIT ch3, MaFaulDa underhang radial column 3.
- STFT: periodic Hann, `center=False`, no padding, one-sided real FFT. CWRU/JNU/MaFaulDa use `n_fft=1024`, hop 256; HIT uses `n_fft=512`, hop 128.
- Transform: `log1p(abs(STFT(x)))`.
- N2: independently for each fold, dataset and frequency bin, fitted only on TRAIN frames in float64; `(X-mu)/std`, with denominator exactly 1 where raw std is below `1e-6`; validation and TEST reuse frozen statistics.
- Patches: 16 frequency bins × 8 time frames.
- PC-STE: stem channels 8; `d_model=192`; four Mamba-1 temporal blocks; two directions fused by mean; absolute Fourier Hz/seconds coordinates; Hz-gated cross-band mixer; validity-masked mean pooling. Encoder parameters: 2,382,033.
- SSL: 60 epochs, 60% random patch masking, reconstruction in normalized log-STFT space.
- Downstream: 50 epochs, no early stopping. S0 random initialization; S1 matched best SSL initialization; encoder and heads fully trained.
- AdamW: lr `3e-4`, betas `(0.9,0.95)`, eps `1e-8`, weight decay `0.05`, global gradient clip 1.0, 5 warm-up epochs, cosine decay to `1e-6`.
- Sampling: effective batch 64 = 16 windows per dataset, with replacement; dataset→class→group→window supervised hierarchy. Fold-specific steps/epoch: 202, 205, 201.
- Downstream selector: maximum four-domain validation MacroDomainF1, strict improvement, earlier epoch on exact ties.
- TEST: checkpoint sealed first; one TEST evaluation; no TEST-driven selection.
- Primary dissertation aggregate: Macro-4, the equal mean of CWRU, JNU, HIT and MaFaulDa Macro-F1 from the executed four-domain outputs. Historical Macro-3 summaries, averaging JNU, HIT and MaFaulDa only, are retained as secondary results.
- K1: 1,375,953-parameter unidirectional encoder, initialized by retaining the forward direction of all four S1 blocks; CE + KL plus relational mixer-attention KL.

## Roots

`PCSTE_DATA_ROOT` defaults to `./data`; `PCSTE_RESULTS_ROOT` defaults to `./results`. Optional distributed scheduling uses three neutral names from `PCSTE_WORKER_HOSTS`.

## Frozen artifacts

`methodology_v2/part1_audit` through `part6_compression` include the dataset audit, splits, window manifests, N2 normalizers, architecture/SSL/experiment/compression specifications, registries and hashes. The largest manifest is under 9 MB and is included directly.

## Label-efficiency execution paths

The publication release contains the execution paths used for all dissertation-facing label fractions:

- 1%: `scripts/run_methodology_v2_1pct_extension.py`
- 5%: `scripts/run_methodology_v2_5pct.py`
- 10%: `scripts/run_methodology_v2_10pct.py`
- 100%: `scripts/methodology_v2/experiment_executor.py`

The 1% experiment is retained with its historical provenance as a reduced-label extension executed after the primary registered grid. The 5% and 10% launchers belong to the registered reduced-label grid. All four public execution paths preserve the executed four-domain supervised protocol: CWRU, JNU, HIT and MaFaulDa contribute supervised loss and encoder gradients through four dataset-specific heads; checkpoints are selected by four-domain validation MacroDomainF1; the principal reported Macro-4 statistic averages CWRU, JNU, HIT and MaFaulDa from the saved four-domain outputs. Historical Macro-3 summaries average JNU, HIT and MaFaulDa only and are retained as secondary aggregates.

The original 1%, 5% and 10% launchers were recovered from their authoritative execution copies. Their historical and publication SHA-256 hashes, publication-only portability edits, output naming and pairing evidence are documented in `docs/low_label_provenance.md`. Original compact result summaries are under `results/tables/low_label/`, and original S0/S1 subset/stream pairing proofs are under `methodology_v2/low_label_provenance/`.

Raw datasets, model checkpoints, full predictions, continuous-score caches and full experiment run trees are deliberately not included. Artifact-dependent integration tests therefore require locally regenerated or separately preserved artifacts, but the publication repository contains the code paths, frozen scientific specifications, manifests, compact result summaries and provenance needed to inspect and rerun the study after obtaining the third-party datasets.
