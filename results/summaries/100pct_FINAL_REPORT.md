# 100% Label S0–S1 Results

This summary corresponds to the dissertation's four-domain full-label evaluation of CWRU, JNU, HIT, and MaFaulDa across three folds and seeds 42, 1337, and 2026.

## Dataset-level Macro-F1

| Dataset | S0 | S1 |
|---|---:|---:|
| CWRU | 0.3200 ± 0.0902 | 0.3234 ± 0.1286 |
| JNU | 0.9905 ± 0.0286 | 0.9810 ± 0.0378 |
| HIT | 0.9744 ± 0.0519 | 0.9755 ± 0.0227 |
| MaFaulDa | 0.7994 ± 0.1015 | 0.8033 ± 0.0626 |
| **Macro-4** | **0.7711 ± 0.0330** | **0.7708 ± 0.0344** |

The four-domain Macro AUC is 0.8850 ± 0.0359 for S0 and 0.8890 ± 0.0374 for S1.

## Paired full-label Macro-4 cells

| Fold | Seed | S0 | S1 | Δ S1−S0 |
|---:|---:|---:|---:|---:|
| 1 | 42 | 0.7510 | 0.7235 | −0.0275 |
| 1 | 1337 | 0.7488 | 0.7412 | −0.0076 |
| 1 | 2026 | 0.7517 | 0.8117 | +0.0599 |
| 2 | 42 | 0.7973 | 0.8271 | +0.0298 |
| 2 | 1337 | 0.7422 | 0.7389 | −0.0033 |
| 2 | 2026 | 0.7315 | 0.7606 | +0.0291 |
| 3 | 42 | 0.8276 | 0.7680 | −0.0596 |
| 3 | 1337 | 0.7911 | 0.7762 | −0.0148 |
| 3 | 2026 | 0.7985 | 0.7901 | −0.0085 |

The mean paired difference is −0.0003. The exact two-sided paired sign-flip test over all 512 sign assignments gives **p = 0.9805**, so the dissertation reports no systematic full-label difference between S0 and S1.

## CWRU per-class interpretation

CWRU is the most challenging downstream dataset. At full labels, S1 class-level F1 is 0.1764 for inner-race fault, 0.2486 for outer-race fault, and 0.5452 for rolling-element fault. Corresponding AUC values are 0.5100, 0.3939, and 0.8143. The result therefore reflects persistent difficulty separating the two race-fault categories rather than uniformly poor recognition.

Primary machine-readable files:

- `../tables/classification_summary_s0_vs_s1.csv`
- `../tables/paired_s0_s1.csv`
- `../tables/statistical_analysis.csv`
- `../tables/per_class_cwru_summary.csv`
