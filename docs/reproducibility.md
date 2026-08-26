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
- Macro-3 reports: post-hoc mean of JNU, HIT and MaFaulDa from saved four-domain outputs.
- K1: 1,375,953-parameter unidirectional encoder, initialized by retaining the forward direction of all four S1 blocks; CE + KL plus relational mixer-attention KL.

## Roots

`PCSTE_DATA_ROOT` defaults to `./data`; `PCSTE_RESULTS_ROOT` defaults to `./results`. Optional distributed scheduling uses three neutral names from `PCSTE_WORKER_HOSTS`.

## Frozen artifacts

`methodology_v2/part1_audit` through `part6_compression` include the dataset audit, splits, window manifests, N2 normalizers, architecture/SSL/experiment/compression specifications, registries and hashes. The largest manifest is under 9 MB and is included directly.

## Low-label reproducibility gaps

The frozen subset/registry supports 5% and 10%, but the surviving executor is explicitly authorized only for 100%. No exact surviving 1% subset definition or launcher exists. The exact additional launch paths and run artifacts that produced dissertation-facing 1%, 5% and 10% results are therefore not reproducible from this release. No replacement was fabricated.
