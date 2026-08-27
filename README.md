# Foundation Models for Industrial Fault Diagnosis: PC-STE

This repository is the clean reproducibility release of an MSc dissertation study of **PC-STE** (Physical-Coordinate Spectro-Temporal Encoder) for rotating-machinery fault diagnosis. It represents the experiment that was actually executed.

## Quick review

If you are reviewing the dissertation implementation, the main entry points are:

- **Model and training implementation:** `src/methodology_v2/`
- **Experiment entry points:** `scripts/`
- **Dataset acquisition, layout and integrity:** `DATA.md`
- **Frozen experimental specifications, splits and manifests:** `methodology_v2/`
- **Reproducibility instructions:** `docs/reproducibility.md`
- **Final compact results:** `results/`
- **Result provenance:** `results/PROVENANCE.md`
- **Reduced-label recovery provenance:** `docs/low_label_provenance.md`
- **Repository map:** `docs/repository_structure.md`

Raw third-party vibration recordings and trained checkpoints are intentionally not redistributed. The repository includes the dataset hashes, manifests, split definitions, preprocessing specifications, compact results and execution paths required to inspect and reproduce the study after obtaining the datasets from their original sources.

## Scientific scope and provenance

Masked-reconstruction SSL used CWRU, JNU, HIT and MaFaulDa. The original downstream S0/S1 experiment also used all four datasets, four dataset-specific heads and four-domain validation `MacroDomainF1`. The dissertation reports **Macro-4**, the equal mean of CWRU, JNU, HIT and MaFaulDa Macro-F1, matching the executed four-domain protocol. Macro-3, the equal mean of JNU, HIT and MaFaulDa only, is retained here as a historical secondary aggregate from an earlier reporting stage. Neither is a natively three-domain experiment.

S0 is a randomly initialized PC-STE trained end-to-end with supervised loss. S1 uses the matched SSL checkpoint and then trains the same encoder and heads end-to-end. K1 is the retained four-block, one-direction student initialized by surgery from Full S1 and trained with `(1-alpha) CE + alpha T^2 KL + L_rel`, where `alpha=0.5`, `T=4`, and `L_rel` is mixer-attention relational KL with weight 1.0. Q8 is the recorded weight-only per-output-channel INT8 representation; packed dynamic INT8 is separately used for CPU latency.

## Data and preprocessing

The four third-party datasets are not distributed here. See [`DATA.md`](DATA.md) for acquisition, local layout, channels, sampling rates and the included integrity/reproducibility metadata.

| Dataset | Expected local path below `PCSTE_DATA_ROOT` | Channel | Native rate |
|---|---|---|---:|
| CWRU | `raw/` and `raw_cwru_48k/` | drive-end acceleration | 48 kHz primary |
| JNU | `raw_jnu/JNU-Bearing-Dataset/` | vertical acceleration | 50 kHz |
| HIT | `raw_hit/gdrive_full/HIT-dataset/` | casing accelerometer ch3 | 25 kHz |
| MaFaulDa | `raw_mafaulda/full/` | underhang radial acceleration, column 3 | 50 kHz |

Acquire each dataset from the authoritative URLs in `src/methodology_v2/registry.py`; follow its terms and verify against `methodology_v2/part1_audit/raw_file_hashes.csv`. Set `PCSTE_DATA_ROOT`, or place data under repository-relative `data/`.

Pipeline: waveform → native-rate 1 s windows → periodic-Hann STFT → `abs(STFT)` → `log1p` → fold/dataset/frequency-bin TRAIN-only N2 → 16×8 patches → PC-STE. Training stride is 0.5 s; validation/TEST stride is 1 s. No resampling is used.

## Architecture

PC-STE uses a convolutional patch stem, absolute Fourier frequency/time coordinates, four bidirectional Mamba-1 blocks at `d_model=192`, a Hz-aware gated cross-band mixer, validity-masked pooling and four linear dataset heads. The frozen specification is `configs/dissertation/pcste_encoder.yaml`.

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements_frozen.txt
pip install -e . --no-deps
cp .env.example .env
```

Python 3.12 is the preserved environment target. CUDA is optional for inspection/tests but required for the original GPU execution profile.

## Reproduction workflow

Commands below build artifacts and can be computationally expensive. They are not executed by repository tests.

```bash
python scripts/methodology_v2/run_part1_audit.py
python scripts/methodology_v2/run_part2.py
python scripts/methodology_v2/run_part3b.py
python scripts/methodology_v2/run_part4c.py
python scripts/methodology_v2/experiment_executor.py --help
python scripts/methodology_v2/part6_compression.py --help
```

Exact public label-efficiency execution paths are included:

- 1%: `scripts/run_methodology_v2_1pct_extension.py`
- 5%: `scripts/run_methodology_v2_5pct.py`
- 10%: `scripts/run_methodology_v2_10pct.py`
- 100%: `scripts/methodology_v2/experiment_executor.py`

The 1% path is a reduced-label extension executed after the primary registered grid. All paths train CWRU, JNU, HIT and MaFaulDa with four dataset heads and select checkpoints by four-domain validation `MacroDomainF1`. Principal Macro-3 reporting is the later mean over JNU, HIT and MaFaulDa from the saved four-domain outputs; these are not native three-domain experiments. See `docs/reproducibility.md` and `docs/low_label_provenance.md`.

To reproduce the existing compact analyses after generating checkpoints/results:

```bash
python scripts/extract_100pct_final_analysis.py
python scripts/posthoc_100pct_frozen_evaluation.py --help
python scripts/methodology_v2/benchmark_part6_latency.py --help
```

## Results summary

Compact, unchanged result tables are under `results/tables/`; selected figures are under `results/figures/`. Principal reported means are Macro-4: full-label S0 0.7711, S1 0.7708, K1 0.793125 and Q8(K1) 0.793382. The corresponding historical Macro-3 values, excluding CWRU, are 0.9214, 0.9199, 0.936913 and 0.937011. Both are computed from the same per-dataset results of the executed four-domain protocol. These are Macro-3 summaries of models trained under the executed four-domain protocol. See `results/PROVENANCE.md` before reuse.

## Repository layout

- `src/methodology_v2/`: complete current implementation
- `scripts/methodology_v2/`: frozen builders, executor and compression tools
- `methodology_v2/`: frozen specifications, manifests, registries and hashes
- `configs/dissertation/`: convenient copies of authoritative frozen specifications
- `tests/methodology_v2/`: current PC-STE tests
- `results/`: curated summaries only
- `docs/`: methodology, reproducibility and layout documentation

No raw data, checkpoints, teacher caches, probability caches or training logs are included. No software licence has yet been selected.

## Citation

Dissertation citation to be added after final bibliographic approval.
