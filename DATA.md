# Data and dataset reproducibility

This repository does **not** redistribute the raw CWRU, JNU, HIT or MaFaulDa recordings. They are third-party datasets and should be obtained from their original sources under the applicable terms. The code expects the datasets below `PCSTE_DATA_ROOT` (default: repository-relative `data/`).

| Dataset | Role in executed study | Raw data distributed here? | Expected local path below `PCSTE_DATA_ROOT` | Channel | Native rate |
|---|---|---|---|---|---:|
| CWRU | SSL + supervised classification | No | `raw/` and `raw_cwru_48k/` | drive-end acceleration | 48 kHz primary |
| JNU | SSL + supervised classification | No | `raw_jnu/JNU-Bearing-Dataset/` | vertical acceleration | 50 kHz |
| HIT | SSL + supervised classification | No | `raw_hit/gdrive_full/HIT-dataset/` | casing accelerometer ch3 | 25 kHz |
| MaFaulDa | SSL + supervised classification | No | `raw_mafaulda/full/` | underhang radial acceleration, column 3 | 50 kHz |

## What is included instead of the raw recordings

The repository contains the scientific metadata needed to inspect and reproduce the data preparation protocol after obtaining the datasets:

- authoritative dataset registry and acquisition references in `src/methodology_v2/registry.py`;
- dataset census and raw-file integrity hashes in `methodology_v2/part1_audit/`;
- frozen fold definitions and TEST identities in `methodology_v2/part2_splits/`;
- 1-second window manifests in `methodology_v2/part3_windows/`;
- JNU guard definitions and HIT logical-stream metadata where applicable;
- frozen STFT and representation specifications in `methodology_v2/part4_*`;
- TRAIN-only fold/dataset N2 normalizers;
- low-label subset and pairing provenance for the reduced-label experiments;
- configuration mirrors under `configs/dissertation/`.

The largest included reproducibility manifests are below GitHub's normal file-size limits. Raw data, model checkpoints, teacher caches, full prediction caches and experiment run trees are intentionally excluded.

## Preparation summary

The executed input pipeline is:

```text
raw vibration recording
→ leakage-aware fold/group assignment
→ 1.000 s windows
→ native-rate periodic-Hann STFT
→ abs(STFT)
→ log1p magnitude
→ fold/dataset/frequency-bin TRAIN-only N2 normalization
→ 16 × 8 spectrogram patches
→ PC-STE
```

Training windows use a 0.5 s stride; validation and TEST windows use a 1.0 s stride. No cross-dataset resampling is performed.

STFT settings are:

| Dataset | Sampling rate | `n_fft` | Hop |
|---|---:|---:|---:|
| CWRU | 48 kHz | 1024 | 256 |
| JNU | 50 kHz | 1024 | 256 |
| HIT | 25 kHz | 512 | 128 |
| MaFaulDa | 50 kHz | 1024 | 256 |

## Verification workflow

After downloading the datasets, set the root explicitly if desired:

```bash
export PCSTE_DATA_ROOT=/path/to/local/pcste-data
```

Then use the frozen audit/preparation scripts and metadata described in `docs/reproducibility.md`. Dataset integrity can be checked against `methodology_v2/part1_audit/raw_file_hashes.csv`.

This structure is intentional: the public repository shares the code, manifests, hashes, splits and preprocessing protocol while avoiding redistribution of third-party raw data.
