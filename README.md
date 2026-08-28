# Foundation Models for Industrial Fault Diagnosis: PC-STE

This repository is the clean reproducibility release of the MSc dissertation **A Foundation-Model-Based Approach for Fault Diagnosis in Industrial Rotating Machinery**. It implements **PC-STE** (Physical-Coordinate Spectro-Temporal Encoder), the matched S0/S1 experiments, the lightweight K1/Q8 study, and the controlled four-domain InceptionTime comparison reported in the dissertation.

## Quick review

If you are reviewing the dissertation implementation, the main entry points are:

- **PC-STE model and training implementation:** `src/methodology_v2/`
- **PC-STE experiment entry points:** `scripts/methodology_v2/` and the public reduced-label launchers under `scripts/`
- **Controlled InceptionTime baseline:** `src/baselines/` and `scripts/baselines/`
- **Baseline compact evidence:** `results/baselines/inceptiontime_four_domain/`
- **Dataset acquisition, layout and integrity:** `DATA.md`
- **Final dissertation protocol:** `configs/dissertation/experiment_protocol.yaml`
- **Historical frozen specifications, splits and manifests:** `methodology_v2/`
- **Reproducibility instructions:** `docs/reproducibility.md`
- **Final compact results:** `results/`
- **Result provenance:** `results/PROVENANCE.md`
- **Reduced-label recovery provenance:** `docs/low_label_provenance.md`
- **Repository map:** `docs/repository_structure.md`

Raw third-party vibration recordings and trained checkpoints are intentionally not redistributed. The repository includes dataset hashes, frozen manifests and partitions, training-only normalisation statistics, model/training code, compact results, and public execution paths required to inspect and reproduce the reported study after obtaining the datasets from their original providers.

## Scientific scope and provenance

Masked-reconstruction SSL used CWRU, JNU, HIT and MaFaulDa. The downstream S0/S1 experiment also used all four datasets, four dataset-specific heads and four-domain validation `MacroDomainF1`. The dissertation reports **Macro-4**, the equal mean of CWRU, JNU, HIT and MaFaulDa Macro-F1, matching the executed four-domain protocol. **Macro-3**, the equal mean of JNU, HIT and MaFaulDa only, is retained as a historical secondary aggregate from an earlier reporting stage. It does not represent separate three-domain training.

The downstream label-efficiency study used exactly **1%, 5%, 10%, and 100% labels**. No 25% or 50% experiment was executed or reported. The sealed pre-execution Part 5D planning registry contains candidate 25% and 50% rows; those rows are preserved only as historical planning provenance so that its original hash chain remains verifiable. The authoritative dissertation-facing fraction list is `configs/dissertation/experiment_protocol.yaml`.

S0 is a randomly initialised PC-STE trained end-to-end with supervised loss. S1 uses the matched SSL checkpoint and then trains the same encoder and heads end-to-end. K1 is the retained four-block, one-direction student initialised by surgery from Full S1 and trained with `(1-alpha) CE + alpha T^2 KL + L_rel`, where `alpha=0.5`, `T=4`, and `L_rel` is mixer-attention relational KL with weight 1.0. Q8 is the recorded weight-only per-output-channel INT8 representation; packed dynamic INT8 is separately used for CPU latency.

The dissertation also evaluates a supervised **InceptionTime-style waveform baseline** under the same frozen downstream partitions, full-label availability, three folds, seeds 42/1337/2026, 50-epoch budget, fold-specific step counts, validation-only four-domain checkpoint selector, and sealed TEST boundary. It uses six Inception modules with residual connections, bottleneck dimension 32, 32 filters per convolution branch, kernels 40/20/10, global average pooling, and four dataset-specific heads. This is a **complete-method comparison**, not an architecture-only ablation: InceptionTime consumes native-length one-second waveforms, whereas Full S1 uses the frozen log-STFT/N2 PC-STE pipeline after masked-reconstruction pretraining.

## Data and preprocessing

The four third-party datasets are not distributed here. See [`DATA.md`](DATA.md) for acquisition, local layout, channels, sampling rates and the included integrity/reproducibility metadata.

| Dataset | Expected local path below `PCSTE_DATA_ROOT` | Channel | Native rate |
|---|---|---|---:|
| CWRU | `raw/` and `raw_cwru_48k/` | drive-end acceleration | 48 kHz primary |
| JNU | `raw_jnu/JNU-Bearing-Dataset/` | vertical acceleration | 50 kHz |
| HIT | `raw_hit/gdrive_full/HIT-dataset/` | casing accelerometer ch3 | 25 kHz |
| MaFaulDa | `raw_mafaulda/full/` | underhang radial acceleration, column 3 | 50 kHz |

Acquire each dataset from the authoritative URLs in `src/methodology_v2/registry.py`; follow its terms and verify against `methodology_v2/part1_audit/raw_file_hashes.csv`. Set `PCSTE_DATA_ROOT`, or place data under repository-relative `data/`.

PC-STE pipeline: waveform → native-rate 1 s windows → periodic-Hann STFT → `abs(STFT)` → `log1p` → fold/dataset/frequency-bin TRAIN-only N2 → 16×8 patches → PC-STE. Training stride is 0.5 s; validation/TEST stride is 1 s. No cross-dataset resampling is used. The controlled InceptionTime baseline instead consumes the same native-length one-second waveform windows directly.

## Architecture

PC-STE uses a convolutional patch stem, absolute Fourier frequency/time coordinates, four bidirectional Mamba-1 blocks at `d_model=192`, a Hz-aware gated cross-band mixer, validity-masked pooling and four linear dataset heads. The frozen specification is `configs/dissertation/pcste_encoder.yaml`.

The controlled baseline implementation is in `src/baselines/inceptiontime.py`; its portable protocol summary is `results/baselines/inceptiontime_four_domain/baseline_spec.json`.

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

These are the only four label regimes used in the dissertation. The 1% path was executed after the other three executed label fractions and uses the identical protocol. Its historical launcher and directory names are retained unchanged so that recorded paths and hashes remain valid. All paths train CWRU, JNU, HIT and MaFaulDa with four dataset heads and select checkpoints by four-domain validation `MacroDomainF1`. See `docs/reproducibility.md` and `docs/low_label_provenance.md`.

The controlled baseline can be inspected or rerun through:

```bash
python scripts/baselines/run_inceptiontime.py --help
python scripts/baselines/aggregate_inceptiontime_four_domain.py --help
python scripts/baselines/analyze_inceptiontime_public.py --help
```

To reproduce the existing compact PC-STE analyses after generating checkpoints/results:

```bash
python scripts/extract_100pct_final_analysis.py
python scripts/posthoc_100pct_frozen_evaluation.py --help
python scripts/methodology_v2/benchmark_part6_latency_four_domain.py --help
```

The old `scripts/methodology_v2/benchmark_part6_latency.py` entry point is retained only as historical three-domain latency provenance.

## Results summary

Principal reported means use Macro-4, the equal mean of CWRU, JNU, HIT and MaFaulDa Macro-F1:

| Model | Macro-4 mean ± sample SD |
|---|---:|
| Full S0 | 0.7711 ± 0.0330 |
| Full S1 | 0.7708 ± 0.0344 |
| K1 | 0.7931 ± 0.0288 |
| Q8(K1) | 0.793382 ± 0.028887 |
| InceptionTime baseline | 0.2816 ± 0.0442 |

For the controlled external comparison, Full S1 exceeded InceptionTime in all **9/9** matched fold-seed cells, with mean paired difference **+0.4891** and exact two-sided sign-flip **p = 0.00390625**. K1 also exceeded the baseline in all 9/9 cells, with mean paired difference **+0.5115** and the same exact p-value. These results support superiority over the evaluated supervised Inception-style baseline **under this dissertation's common protocol**; they are not a claim of universal superiority over InceptionTime or published fault-diagnosis methods in general.

The completed training programme comprises **99 successful training runs** and approximately **661.01 summed GPU-hours**. The nine InceptionTime runs contributed **91.49 GPU-hours**, averaging **10.17 h/run**, each on one NVIDIA RTX 4000 Ada Generation GPU.

### Four-domain efficiency

The final dissertation efficiency benchmark uses nine matched fold-seed cells, all four datasets with equal domain weight, batch size 1, 9,000 warm-up forwards and 90,000 timed forwards in one controlled session.

| Model | Encoder parameters | Serialized size | GFLOP/window | CPU ms/window | GPU ms/window |
|---|---:|---:|---:|---:|---:|
| Full S1 FP32 | 2,382,033 | 9.579 MB | 3.058 | 22.2394 ± 0.6728 | 7.6638 ± 0.0295 |
| K1 FP32 | 1,375,953 | 5.543 MB | 1.680 | 11.7294 ± 0.2985 | 4.1529 ± 0.0131 |
| Packed Q8(K1) CPU | 1,375,953 logical | 1.600 MB | 1.680 logical | 11.3912 ± 0.0728 | N/A |

Relative to Full S1, K1 reduces encoder parameters by 42.24%, FLOPs by 45.08%, sequential scan steps by 50%, CPU latency by 47.26%, and GPU latency by 45.81%. Its CPU/GPU speed-ups are 1.8960×/1.8454×. Packed Q8(K1) gives a 1.9523× CPU speed-up and 48.78% CPU latency reduction versus Full S1.

There is no genuine packed INT8 CUDA latency result: the CUDA-compatible Q8 simulation dequantizes to FP32. Packed-versus-simulated Q8 showed **high class-level agreement**, 571/576 validation windows (99.13%), but not exact numerical parity and not universally identical predictions.

Authoritative evidence is under `results/methodology_v2/part6_compression/latency_four_domain/`; the independent read-only audit is `results/methodology_v2/verification/four_domain_verification.md`.

Compact PC-STE result tables are under `results/tables/`; selected figures are under `results/figures/`; the baseline evidence is under `results/baselines/inceptiontime_four_domain/`. Historical Macro-3 values remain in some preserved summaries as secondary aggregates derived from the same four-domain model outputs. See `results/PROVENANCE.md` before reuse.

## Repository layout

- `src/methodology_v2/`: complete PC-STE implementation
- `scripts/methodology_v2/`: frozen builders, executor and compression tools
- `src/baselines/`, `scripts/baselines/`: controlled InceptionTime baseline implementation and analysis
- `tests/methodology_v2/`, `tests/baselines/`: current PC-STE and baseline tests
- `methodology_v2/`: historical frozen specifications, manifests, registries and hashes
- `configs/dissertation/`: dissertation-facing PC-STE specifications aligned with the final report
- `results/baselines/inceptiontime_four_domain/`: compact controlled-baseline evidence and protocol summary
- `results/`: curated dissertation summaries, tables and figures
- `docs/`: methodology, reproducibility and repository-layout documentation

No raw data, checkpoints, teacher caches, prediction/probability caches, full run trees or training logs are included. This repository is released under the MIT License. See [`LICENSE`](LICENSE) for details.

## Citation

Abdalaziz, M. (2026). *A Foundation-Model-Based Approach for Fault Diagnosis in Industrial Rotating Machinery.* MSc dissertation, University of Surrey.
