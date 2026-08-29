# Reduced-Label S0–S1 Results

The dissertation evaluates S0 and S1 at 1%, 5%, and 10% labelled training data while retaining the same four downstream datasets and frozen validation/TEST partitions.

| Labels | S0 Macro-4 F1 | S1 Macro-4 F1 | ΔF1 | S0 Macro-4 AUC | S1 Macro-4 AUC | ΔAUC | Exact p |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1% | 0.4803 ± 0.0655 | 0.5051 ± 0.0547 | +0.0249 | 0.7282 ± 0.0677 | 0.7851 ± 0.0589 | +0.0569 | 0.3125 |
| 5% | 0.6519 ± 0.0248 | 0.6427 ± 0.0243 | −0.0092 | 0.8429 ± 0.0383 | 0.8628 ± 0.0330 | +0.0199 | 0.4492 |
| 10% | 0.7135 ± 0.0425 | 0.7187 ± 0.0388 | +0.0052 | 0.8771 ± 0.0366 | 0.8578 ± 0.0246 | −0.0193 | 0.6328 |

The clearest mean S1 advantage occurs at 1% labels. At 5% and 10% the aggregate differences are smaller, with dataset-level responses varying by label availability.

The paired test summary across all label fractions is stored in `../../tables/statistical_analysis.csv`.
