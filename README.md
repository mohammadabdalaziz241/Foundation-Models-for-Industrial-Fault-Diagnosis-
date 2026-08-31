# A Foundation-Model-Based Approach for Fault Diagnosis in Industrial Rotating Machinery

This repository contains the implementation, frozen methodology artefacts, and dissertation-facing results for the MSc dissertation **A Foundation-Model-Based Approach for Fault Diagnosis in Industrial Rotating Machinery**.

## Scope

The project develops the Physical-Coordinate Spectro-Temporal Encoder (PC-STE) for vibration fault diagnosis across four heterogeneous datasets: **CWRU, JNU, HIT, and MaFaulDa**. The shared encoder processes native-rate log-STFT spectrograms using physical time-frequency coordinates, four bidirectional Mamba blocks, and an Hz-aware cross-band mixer.

Masked-reconstruction self-supervised learning pretrains the encoder on the training partitions of all four datasets. Downstream classification uses dataset-specific heads and compares:

- **S0:** supervised training from random initialisation.
- **S1:** masked-reconstruction pretraining followed by full fine-tuning.

The matched evaluation uses three frozen folds, seeds 42/1337/2026, and labelled-data fractions of 1%, 5%, 10%, and 100%. Final cross-dataset reporting uses **Macro-4**, the equal mean of CWRU, JNU, HIT, and MaFaulDa dataset-level Macro-F1 values.

## Principal full-label result

Across the nine fold-seed cells, full-label Macro-4 F1 is:

| Arm | Macro-4 F1 |
|---|---:|
| S0 | 0.7711 ± 0.0330 |
| S1 | 0.7708 ± 0.0344 |

The exact paired two-sided sign-flip test gives **p = 0.9805**, indicating no systematic full-label difference between S0 and S1. The clearest mean S1 advantage occurs at 1% labels, where Macro-4 F1 increases from 0.4803 to 0.5051.

CWRU is the only domain on which the model remains approximately at the uniform-random chance reference under specimen-disjoint evaluation. At 100% labels, its Macro-F1 is 0.3200 ± 0.0902 for S0 and 0.3234 ± 0.1286 for S1, against a fold-averaged uniform-random chance reference of 0.3174. Per-class AUC shows the rolling-element fault is ranked well above chance, while the two race faults are ranked at or below it.

## Lightweight model

The structurally reduced K1 model retains four temporal stages but uses one Mamba direction per block. Relative to Full S1, K1 reduces encoder parameters by 42.24%, forward-pass computation by 45.08%, and selective-scan processing by 50%. Mean Macro-4 rises from 0.7708 for Full S1 to 0.7931 for K1. Q8(K1) retains essentially the same predictive performance while reducing serialized model-state size to 1.600 MB.

## Controlled baseline diagnostics

The InceptionTime baseline was trained under the same optimiser, learning rate,
schedule, 50-epoch budget, effective batch, four-dataset balancing, gradient
clipping and validation checkpoint selector as the PC-STE downstream runs.
Retained training-loss histories fall from approximately 1.26 at epoch 1 to
below 1.3e-5 by epoch 50, so the near-chance TEST performance reflects failure
to generalise rather than failure to fit.

The architecture uses six Inception modules with a longest kernel of 40, stride
one and dilation one, giving a maximum receptive field of 235 input samples
against one-second windows of 25,000-50,000 samples. It therefore observes
under 1% of each window before global temporal average pooling.

Chance-level references for the frozen TEST partitions, computed as the
uniform-random macro-F1 for each class prior, are 0.3174 for CWRU (fold-averaged
over supports [35,35,35], [35,99,35] and [28,99,35]), 0.2143 for JNU, 0.3258 for
HIT and 0.0961 for MaFaulDa, giving a Macro-4 uniform-random chance reference of 0.2384. Chance macro-AUC
is 0.5.

## Data

The four datasets are **not redistributed in this repository**. They are obtained from their original
sources and placed on disk under a single data root. Raw data is excluded from version control
(`data/` is listed in `.gitignore`).

### Sources

| Dataset | Source | Reference |
|---|---|---|
| CWRU | https://engineering.case.edu/bearingdatacenter | CWRU Bearing Data Center |
| JNU | https://github.com/ClarkGableWang/JNU-Bearing-Dataset | Jiangnan University bearing dataset (pinned commit `75b33611b51649d1da8ff5999397899420753e5b`) |
| HIT | https://github.com/HouLeiHIT/HIT-dataset | Hou et al. 2023, doi:10.37965/jdmd.2023.314 (pinned commit `ef17655977519eda9463a3d060da9fb8b47fab6f`) |
| MaFaulDa | https://www02.smt.ufrj.br/~offshore/mfs/page_01.html | UFRJ/SMT Machinery Fault Database |

Licence and provenance for each dataset, as documented by the original distributor, are recorded per
dataset in `methodology_v2/part1_audit/dataset_census.csv` (`licence_provenance` column). Where the
distributor states no explicit licence, that is recorded as such rather than assumed.

### Data root

All dataset paths resolve from a single root, set via environment variable (see `.env.example`):

```bash
export PCSTE_DATA_ROOT=/path/to/local/pcste-data
```

If unset, the root defaults to `data/` inside the repository. The resolution is in
`src/methodology_v2/registry.py`, which is the single source of truth for dataset paths, valid
labels, and documented acquisition facts.

### Expected layout

```
$PCSTE_DATA_ROOT/
├── raw/                                  # CWRU, 12 kHz drive-end
│   ├── ball/                             #  16 files
│   ├── inner_race/                       #  16 files
│   ├── outer_race/                       #  24 files
│   └── normal/                           #   4 files
├── raw_cwru_48k/                         # CWRU, 48 kHz drive-end
│   ├── ball/                             #  12 files
│   ├── inner_race/                       #  12 files
│   ├── outer_race/                       #  28 files
│   └── normal/                           #   4 files
├── raw_jnu/
│   └── JNU-Bearing-Dataset/              # 12 single-column CSVs
├── raw_hit/
│   ├── HIT-dataset/                      # GitHub release (.mat shards)
│   └── gdrive_full/
│       └── HIT-dataset/                  # data1.npy ... data5.npy
└── raw_mafaulda/
    ├── full.zip                          # downloaded archive
    └── full/                             # extracted, 1,951 CSVs
        ├── normal/                       #    49
        ├── horizontal-misalignment/      #   197
        ├── vertical-misalignment/        #   301
        ├── imbalance/                    #   333
        ├── underhang/                    #   558
        └── overhang/                     #   513
```

### Per-dataset notes

**CWRU.** Local files are the official CWRU `.mat` files renamed to a readable convention
(`B007_0_DE12k.mat`, `IR014_2_DE12k.mat`, and so on). The internal MATLAB variable names
(`X097`, `X105`, …) preserve the canonical CWRU file numbers, so the mapping back to the official
distribution is recoverable from the file contents. The 12 kHz and 48 kHz trees are kept separate
because sampling rate is not reliably inferable from the official filenames; the four `Normal_*`
baseline files are byte-identical in both trees and are genuinely 48 kHz.

**JNU.** Twelve CSVs, one per (condition, speed) pair, with file prefixes `n`, `ib`, `ob`, `tb` for
healthy, inner-race, outer-race, and rolling-element respectively.

**HIT.** Two copies are used for different purposes. The pipeline consumes the five session-level
`.npy` arrays under `gdrive_full/`, which retain the session and speed-group structure needed for
leakage-safe grouping. The GitHub release shards (`xtrain_*.mat`, `xtest.mat`, …) are retained for
auditing only: that release ships an official train/test split, but it is window-level random and is
therefore **rejected as a split authority** in this work (`src/methodology_v2/part2_builder.py`).

**MaFaulDa.** Extract `full.zip` in place. The archive's own folder taxonomy is treated as
operational ground truth; the website prose in section 1.4.5 permutes fault-type names relative to
its own summary table, and this discrepancy is flagged in the integrity report rather than silently
resolved.

### Verifying a local copy

Part 1 of the methodology is a strictly read-only audit that rebuilds the manifest, census, and
per-file hashes from raw data alone. It creates no windows, splits, or training artefacts, and is the
fastest way to confirm a local copy matches the one used for the reported results:

```bash
python scripts/methodology_v2/run_part1_audit.py --datasets CWRU JNU HIT MAFAULDA
```

Compare the regenerated output against the committed
`methodology_v2/part1_audit/raw_file_hashes.csv`, which records SHA-256 digests for 2,094 raw files.
Both whole-file and payload-level digests are stored, so a copy that differs only in container
metadata can be distinguished from one that differs in signal content.

## Repository layout

- `src/` — model, preprocessing, experiment, and compression implementation.
- `scripts/` — experiment, evaluation, baseline, and benchmarking entry points.
- `configs/dissertation/` — dissertation configuration files.
- `methodology_v2/` — frozen audit, split, representation, architecture, and experiment-registry artefacts.
- `results/` — dissertation-facing result tables, summaries, figures, baseline outputs, and efficiency measurements.
- `tests/` — automated checks for the frozen pipeline and implementation.

## Key result files

- `results/tables/classification_summary_s0_vs_s1.csv` — full-label four-domain classification summary corresponding to Dissertation Table 5.2.
- `results/tables/paired_s0_s1.csv` — nine paired full-label Macro-4 cells corresponding to Table 5.3.
- `results/tables/statistical_analysis.csv` — paired S0-S1 statistical results across label fractions corresponding to Table 5.5.
- `results/tables/per_class_cwru_summary.csv` — CWRU full-label per-class precision, recall, F1, and AUC summary corresponding to the 100% rows of Table A.1.
- `results/baselines/inceptiontime_four_domain/` — controlled four-domain InceptionTime comparison.
- `results/methodology_v2/part6_compression/latency_four_domain/` — four-domain latency benchmark artefacts.

## Reproducibility principles

The experimental protocol partitions recordings or physical groups before one-second window generation, estimates N2 normalisation statistics from training data only, restricts SSL pretraining to training-partition groups, and keeps TEST data isolated until the selected checkpoint is evaluated. S0 and S1 use matched folds, seeds, labelled subsets, architecture, optimisation, validation rule, and TEST partitions; encoder initialisation is the controlled difference between the two arms.

## License

See `LICENSE`.
