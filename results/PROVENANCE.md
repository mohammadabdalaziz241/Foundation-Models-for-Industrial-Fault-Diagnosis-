# Results Provenance

The public result tables in this repository follow the four-domain evaluation protocol reported in the dissertation. The downstream domains are **CWRU, JNU, HIT, and MaFaulDa**, and aggregate classification performance is reported as their equal-weight **Macro-4** mean.

## Principal S0-S1 results

The main result artefacts correspond directly to the dissertation:

| Repository artefact | Dissertation evidence |
|---|---|
| `tables/classification_summary_s0_vs_s1.csv` | Table 5.2, full-label downstream classification performance |
| `tables/paired_s0_s1.csv` | Table 5.3, nine paired full-label Macro-4 cells |
| `tables/statistical_analysis.csv` | Table 5.5, exact paired comparison across label fractions |
| `tables/per_class_cwru_summary.csv` | Table A.1, CWRU 100% per-class precision, recall, F1, and AUC |
| `tables/per_class_jnu_summary.csv` | Table A.2, JNU per-class analysis |
| `tables/per_class_hit_summary.csv` | Table A.3, HIT per-class analysis |
| `tables/per_class_mafaulda_summary.csv` | Table A.4, MaFaulDa per-class analysis |

At 100% labels, the dissertation reports S0 Macro-4 F1 of **0.7711 ± 0.0330** and S1 Macro-4 F1 of **0.7708 ± 0.0344**. The exact paired two-sided sign-flip test gives **p = 0.9805**.

At reduced label fractions, the Macro-4 F1 results are:

| Labels | S0 | S1 | S1-S0 |
|---|---:|---:|---:|
| 1% | 0.4803 ± 0.0655 | 0.5051 ± 0.0547 | +0.0249 |
| 5% | 0.6519 ± 0.0248 | 0.6427 ± 0.0243 | -0.0092 |
| 10% | 0.7135 ± 0.0425 | 0.7187 ± 0.0388 | +0.0052 |
| 100% | 0.7711 ± 0.0330 | 0.7708 ± 0.0344 | -0.0003 |

The corresponding exact paired p-values are 0.3125, 0.4492, 0.6328, and 0.9805.

## CWRU evidence

CWRU uses the specimen-grouped downstream protocol and is the most challenging domain in the reported results. Full-label Macro-F1 is **0.3200 ± 0.0902** for S0 and **0.3234 ± 0.1286** for S1. The 100% per-class S1 results in `tables/per_class_cwru_summary.csv` are:

- inner-race fault: F1 0.1764 ± 0.2835; AUC 0.5100 ± 0.3819;
- outer-race fault: F1 0.2486 ± 0.3627; AUC 0.3939 ± 0.2456;
- rolling-element fault: F1 0.5452 ± 0.3016; AUC 0.8143 ± 0.1862.

These values support the dissertation discussion that persistent race-fault confusion limits CWRU performance.

## Lightweight models

The dissertation compares Full S1, K1, and Q8(K1) over the same four downstream domains and nine fold-seed cells. Mean Macro-4 is **0.770791 ± 0.034432** for Full S1, **0.793125 ± 0.028787** for K1, and **0.793382 ± 0.028887** for Q8(K1). The one-sided non-inferiority tests give **p = 0.001953** for K1 versus Full S1 at the -0.02 margin and **p = 0.001953** for Q8(K1) versus K1 at the -0.01 margin.

Four-domain latency and efficiency evidence is stored under `methodology_v2/part6_compression/latency_four_domain/`. The dissertation reports Full S1 versus K1 equal-domain latency of 22.2394 versus 11.7294 ms/window on CPU and 7.6638 versus 4.1529 ms/window on GPU.

## Controlled baseline

`baselines/inceptiontime_four_domain/` contains the controlled InceptionTime comparison. Under the same four-domain downstream protocol, InceptionTime achieves **0.2816 ± 0.0442 Macro-4**, Full S1 **0.7708 ± 0.0344**, and K1 **0.7931 ± 0.0288**. Full S1 and K1 each outperform the matched baseline in all nine cells, with exact two-sided **p = 0.0039**.

## Evaluation integrity

The repository preserves the leakage controls described in the dissertation: partitions are defined before one-second window generation; N2 normalisation statistics are estimated from training data only; self-supervised pretraining is restricted to training-partition groups; and TEST data are excluded from optimisation and checkpoint selection. S0 and S1 use matched folds, seeds, labelled subsets, optimisation, validation, and TEST partitions.
