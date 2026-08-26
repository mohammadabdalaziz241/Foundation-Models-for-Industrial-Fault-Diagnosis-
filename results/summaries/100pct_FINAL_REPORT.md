# 100% Label Final Experimental Results

## 1. Executive Summary

This report covers 18/18 completed downstream runs: S0 (supervised from scratch) and S1 (SSL-pretrained, full fine-tuning), evaluated over three folds and three seeds on JNU, HIT and MaFaulDa at 100% labels. The equal-dataset Macro-3 F1 was **0.9214 ± 0.0273 for S0** and **0.9199 ± 0.0179 for S1** (paired Δ = **−0.0015 ± 0.0311**). The exact 512-pattern paired sign-flip test was not significant. All frozen-checkpoint replay checks passed.

ROC/AUC and sample-level reconstruction measures are explicitly post-hoc supplemental evaluations from frozen checkpoints. No training, checkpoint selection, tuning or model modification occurred.

## 2. Experimental Scope

Datasets: JNU, HIT and MaFaulDa. Arms: S0 and S1. Folds: 1–3. Seeds: 42, 1337 and 2026. This report includes only the 100%-label condition. Each arm has nine paired fold×seed runs and each run evaluates all three datasets.

## 3. Evaluation Metrics

Classification metrics are calculated on the frozen TEST split. Balanced accuracy is macro recall; macro metrics weight classes equally; weighted metrics use class support. The principal AUC measure is multiclass one-vs-rest macro ROC-AUC.

Reconstruction metrics operate on masked valid cells of the fold-normalized log1p-magnitude STFT patches. MSE, MAE, R², Pearson r and Spearman ρ are calculated per window and then averaged. Supplemental NMSE is `Σ(target−prediction)² / Σtarget²` per window, then averaged. Undefined constant/zero-energy cases are logged rather than replaced.

Waveform-domain reconstruction metrics were not reported because the SSL objective reconstructs normalized log-magnitude STFT patches rather than a complete complex-valued waveform representation.

## 4. Reconstruction Results

**Table 1. Masked normalized log-STFT reconstruction metrics (mean ± sample SD over nine fold×seed cells).**

| Dataset | MSE | NMSE | MAE | R² | Pearson r | Spearman ρ |
|---|---:|---:|---:|---:|---:|---:|
| JNU | 0.6597 ± 0.0158 | 0.6646 ± 0.0107 | 0.6096 ± 0.0085 | 0.0974 ± 0.0133 | 0.3220 ± 0.0116 | 0.2738 ± 0.0121 |
| HIT | 0.6153 ± 0.0286 | 0.6069 ± 0.0260 | 0.6160 ± 0.0166 | 0.3430 ± 0.0175 | 0.5807 ± 0.0127 | 0.5390 ± 0.0053 |
| MaFaulDa | 0.4558 ± 0.0211 | 0.5807 ± 0.0050 | 0.5188 ± 0.0078 | 0.2423 ± 0.0098 | 0.4831 ± 0.0082 | 0.4286 ± 0.0080 |
| Macro-3 | 0.5769 ± 0.0197 | 0.6174 ± 0.0077 | 0.5814 ± 0.0102 | 0.2275 ± 0.0076 | 0.4619 ± 0.0076 | 0.4138 ± 0.0067 |

![Figure 1. JNU deterministic masked log-STFT reconstruction example.](figures/reconstruction/reconstruction_jnu_masked_logstft.png)

![Figure 2. HIT deterministic masked log-STFT reconstruction example.](figures/reconstruction/reconstruction_hit_masked_logstft.png)

![Figure 3. MaFaulDa deterministic masked log-STFT reconstruction example.](figures/reconstruction/reconstruction_mafaulda_masked_logstft.png)

The displayed example is the lexicographically sorted middle validation window for fold 1, seed 42; selection is independent of reconstruction quality.

## 5. Overall Classification Results

**Table 2. Complete dataset-level classification results (mean ± sample SD).**

| Dataset | S0 Accuracy | S0 Bal.Acc | S0 Precision | S0 Recall | S0 F1 | S0 AUC | S1 Accuracy | S1 Bal.Acc | S1 Precision | S1 Recall | S1 F1 | S1 AUC | ΔF1 | ΔAUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| JNU | 0.9954 ± 0.0139 | 0.9907 ± 0.0278 | 0.9931 ± 0.0208 | 0.9907 ± 0.0278 | 0.9905 ± 0.0286 | 1.0000 ± 0.0000 | 0.9907 ± 0.0184 | 0.9815 ± 0.0367 | 0.9861 ± 0.0276 | 0.9815 ± 0.0367 | 0.9810 ± 0.0378 | 1.0000 ± 0.0000 | -0.0095 | +0.0000 |
| HIT | 0.9786 ± 0.0366 | 0.9722 ± 0.0573 | 0.9837 ± 0.0270 | 0.9722 ± 0.0573 | 0.9744 ± 0.0519 | 0.9991 ± 0.0025 | 0.9746 ± 0.0240 | 0.9762 ± 0.0201 | 0.9777 ± 0.0233 | 0.9762 ± 0.0201 | 0.9755 ± 0.0227 | 0.9957 ± 0.0118 | +0.0011 | -0.0034 |
| MaFaulDa | 0.8221 ± 0.0859 | 0.8330 ± 0.0830 | 0.8209 ± 0.1038 | 0.8330 ± 0.0830 | 0.7994 ± 0.1015 | 0.9804 ± 0.0148 | 0.8352 ± 0.0540 | 0.8503 ± 0.0448 | 0.8301 ± 0.0700 | 0.8503 ± 0.0448 | 0.8033 ± 0.0626 | 0.9874 ± 0.0074 | +0.0039 | +0.0070 |
| Macro-3 | 0.9320 ± 0.0236 | 0.9320 ± 0.0216 | 0.9325 ± 0.0321 | 0.9320 ± 0.0216 | 0.9214 ± 0.0273 | 0.9932 ± 0.0046 | 0.9335 ± 0.0126 | 0.9360 ± 0.0134 | 0.9313 ± 0.0171 | 0.9360 ± 0.0134 | 0.9199 ± 0.0179 | 0.9944 ± 0.0059 | -0.0015 | +0.0012 |

Weighted precision, recall and F1 are retained in `tables/classification_summary_s0_vs_s1.csv`.

## 6. Paired Fold × Seed Results

**Table 3. Paired equal-dataset Macro-3 F1.**

| Fold | Seed | S0 Macro-3 F1 | S1 Macro-3 F1 | Δ S1−S0 |
|---|---:|---:|---:|---:|
| 1 | 42 | 0.9250 | 0.8906 | -0.0345 |
| 1 | 1337 | 0.9277 | 0.9538 | 0.0261 |
| 1 | 2026 | 0.9010 | 0.9220 | 0.0210 |
| 2 | 42 | 0.9006 | 0.9332 | 0.0325 |
| 2 | 1337 | 0.9051 | 0.9069 | 0.0018 |
| 2 | 2026 | 0.8784 | 0.9100 | 0.0317 |
| 3 | 42 | 0.9620 | 0.9153 | -0.0467 |
| 3 | 1337 | 0.9429 | 0.9291 | -0.0138 |
| 3 | 2026 | 0.9501 | 0.9185 | -0.0317 |

S1 wins 5 cells, S0 wins 4, with no ties.

## 7. Statistical Analysis

The frozen exact two-sided paired sign-flip procedure enumerated all 512 sign assignments. Observed mean paired difference = **-0.001503**, median = +0.001794, SD = 0.031069, Cohen's dz = -0.0484, exact p = **0.882812**. At α = 0.05, this is **not statistically significant**. This inference is conditional on the stated caveat that fold×seed cells are paired but not fully independent.

## 8. JNU Per-Class Results

**Table 4. JNU per-class results (mean ± sample SD over nine cells).**

| Class | S0 Precision | S0 Recall | S0 F1 | S0 AUC | S1 Precision | S1 Recall | S1 F1 | S1 AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| healthy | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| inner race fault | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.9722 ± 0.0833 | 1.0000 ± 0.0000 | 0.9841 ± 0.0476 | 1.0000 ± 0.0000 |
| outer race fault | 0.9722 ± 0.0833 | 1.0000 ± 0.0000 | 0.9841 ± 0.0476 | 1.0000 ± 0.0000 | 0.9722 ± 0.0833 | 1.0000 ± 0.0000 | 0.9841 ± 0.0476 | 1.0000 ± 0.0000 |
| rolling element fault | 1.0000 ± 0.0000 | 0.9630 ± 0.1111 | 0.9778 ± 0.0667 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.9259 ± 0.1470 | 0.9556 ± 0.0882 | 1.0000 ± 0.0000 |

## 9. HIT Per-Class Results

**Table 5. HIT per-class results (mean ± sample SD over nine cells).**

| Class | S0 Precision | S0 Recall | S0 F1 | S0 AUC | S1 Precision | S1 Recall | S1 F1 | S1 AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| healthy | 0.9752 ± 0.0480 | 0.9901 ± 0.0234 | 0.9818 ± 0.0258 | 0.9999 ± 0.0004 | 0.9600 ± 0.0538 | 0.9851 ± 0.0414 | 0.9710 ± 0.0304 | 0.9936 ± 0.0163 |
| inner race fault | 0.9759 ± 0.0401 | 0.9861 ± 0.0249 | 0.9805 ± 0.0259 | 0.9988 ± 0.0034 | 0.9953 ± 0.0142 | 0.9593 ± 0.0591 | 0.9760 ± 0.0304 | 0.9936 ± 0.0191 |
| outer race fault | 1.0000 ± 0.0000 | 0.9405 ± 0.1656 | 0.9609 ± 0.1105 | 0.9987 ± 0.0040 | 0.9778 ± 0.0667 | 0.9841 ± 0.0244 | 0.9795 ± 0.0361 | 1.0000 ± 0.0000 |

## 10. MaFaulDa Per-Class Results

**Table 6. MaFaulDa per-class results (mean ± sample SD over nine cells).**

| Class | S0 Precision | S0 Recall | S0 F1 | S0 AUC | S1 Precision | S1 Recall | S1 F1 | S1 AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| normal | 0.6187 ± 0.3325 | 0.9079 ± 0.1602 | 0.7019 ± 0.2791 | 0.9815 ± 0.0393 | 0.5512 ± 0.2645 | 0.9810 ± 0.0474 | 0.6748 ± 0.1951 | 0.9974 ± 0.0040 |
| imbalance | 0.7693 ± 0.1064 | 0.8948 ± 0.1142 | 0.8194 ± 0.0751 | 0.9845 ± 0.0175 | 0.7466 ± 0.1542 | 0.9240 ± 0.0844 | 0.8117 ± 0.0766 | 0.9910 ± 0.0076 |
| horizontal-misalignment | 0.8424 ± 0.3263 | 0.5710 ± 0.3890 | 0.6318 ± 0.3921 | 0.9588 ± 0.0750 | 0.7171 ± 0.3771 | 0.3950 ± 0.4190 | 0.4350 ± 0.4125 | 0.9705 ± 0.0376 |
| vertical-misalignment | 0.8782 ± 0.1493 | 0.9662 ± 0.0417 | 0.9119 ± 0.0801 | 0.9962 ± 0.0056 | 0.8397 ± 0.1790 | 0.9667 ± 0.0565 | 0.8856 ± 0.1008 | 0.9971 ± 0.0049 |
| underhang/ball fault | 0.9188 ± 0.0848 | 0.8717 ± 0.2416 | 0.8782 ± 0.1893 | 0.9964 ± 0.0064 | 0.9434 ± 0.0562 | 0.9446 ± 0.1277 | 0.9395 ± 0.0851 | 0.9979 ± 0.0045 |
| underhang/cage fault | 0.7976 ± 0.2309 | 0.6293 ± 0.3301 | 0.6553 ± 0.2920 | 0.9465 ± 0.0635 | 0.8204 ± 0.1625 | 0.6791 ± 0.3844 | 0.6757 ± 0.3103 | 0.9444 ± 0.0808 |
| underhang/outer race | 0.7524 ± 0.2587 | 0.8283 ± 0.1855 | 0.7750 ± 0.2057 | 0.9768 ± 0.0339 | 0.8451 ± 0.1688 | 0.8178 ± 0.2285 | 0.8067 ± 0.1655 | 0.9828 ± 0.0234 |
| overhang/ball fault | 0.9350 ± 0.0762 | 0.9764 ± 0.0533 | 0.9524 ± 0.0401 | 0.9999 ± 0.0002 | 0.9898 ± 0.0241 | 0.9580 ± 0.0632 | 0.9726 ± 0.0376 | 0.9999 ± 0.0002 |
| overhang/cage fault | 0.8794 ± 0.1774 | 0.8789 ± 0.0856 | 0.8675 ± 0.1104 | 0.9910 ± 0.0198 | 0.9216 ± 0.0935 | 0.9215 ± 0.1077 | 0.9156 ± 0.0746 | 0.9967 ± 0.0053 |
| overhang/outer race | 0.8170 ± 0.2489 | 0.8054 ± 0.3054 | 0.8003 ± 0.2777 | 0.9727 ± 0.0697 | 0.9264 ± 0.0843 | 0.9156 ± 0.1196 | 0.9159 ± 0.0809 | 0.9965 ± 0.0065 |

## 11. ROC/AUC Results

**Table 7. Macro OvR ROC-AUC (mean ± sample SD).**

| Dataset | S0 Macro AUC | S1 Macro AUC | Δ S1−S0 |
|---|---:|---:|---:|
| JNU | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | +0.0000 |
| HIT | 0.9991 ± 0.0025 | 0.9957 ± 0.0118 | -0.0034 |
| MaFaulDa | 0.9804 ± 0.0148 | 0.9874 ± 0.0074 | +0.0070 |
| Macro-3 | 0.9932 ± 0.0046 | 0.9944 ± 0.0059 | +0.0012 |

![Figure 4. JNU S0 mean macro OvR ROC.](figures/roc_curves/aggregate/roc_jnu_s0_aggregate_mean.png)

![Figure 7. JNU S1 mean macro OvR ROC.](figures/roc_curves/aggregate/roc_jnu_s1_aggregate_mean.png)

![Figure 5. HIT S0 mean macro OvR ROC.](figures/roc_curves/aggregate/roc_hit_s0_aggregate_mean.png)

![Figure 8. HIT S1 mean macro OvR ROC.](figures/roc_curves/aggregate/roc_hit_s1_aggregate_mean.png)

![Figure 6. MAFAULDA S0 mean macro OvR ROC.](figures/roc_curves/aggregate/roc_mafaulda_s0_aggregate_mean.png)

![Figure 9. MAFAULDA S1 mean macro OvR ROC.](figures/roc_curves/aggregate/roc_mafaulda_s1_aggregate_mean.png)

Aggregate ROC curves interpolate each run's macro OvR curve onto a fixed 1001-point FPR grid, then average the nine fold×seed curves. Shading is ±1 run SD. Predictions are not concatenated, so repeated seeds do not inflate independent sample size.

## 12. Confusion Matrix Analysis

The following are mean run-normalized matrices: each run is row-normalized first, then the nine matrices are averaged.

### JNU

![Figure 10. JNU S0 mean run-normalized confusion matrix.](figures/confusion_matrices/aggregate/cm_jnu_s0_aggregate_mean_normalized.png)

![Figure 11. JNU S1 mean run-normalized confusion matrix.](figures/confusion_matrices/aggregate/cm_jnu_s1_aggregate_mean_normalized.png)

### HIT

![Figure 12. HIT S0 mean run-normalized confusion matrix.](figures/confusion_matrices/aggregate/cm_hit_s0_aggregate_mean_normalized.png)

![Figure 13. HIT S1 mean run-normalized confusion matrix.](figures/confusion_matrices/aggregate/cm_hit_s1_aggregate_mean_normalized.png)

### MaFaulDa

![Figure 14. MaFaulDa S0 mean run-normalized confusion matrix.](figures/confusion_matrices/aggregate/cm_mafaulda_s0_aggregate_mean_normalized.png)

![Figure 15. MaFaulDa S1 mean run-normalized confusion matrix.](figures/confusion_matrices/aggregate/cm_mafaulda_s1_aggregate_mean_normalized.png)

The complete 126-figure confusion-matrix collection remains under `figures/confusion_matrices/`.

## 13. Best and Worst Runs

**Table 8. Verified best, worst and largest paired differences.**

| Case | Fold | Seed | Macro-3 F1 / Δ |
|---|---:|---:|---:|
| Best S0 | 3 | 42 | 0.9620 |
| Worst S0 | 2 | 2026 | 0.8784 |
| Best S1 | 1 | 1337 | 0.9538 |
| Worst S1 | 1 | 42 | 0.8906 |
| Largest positive S1−S0 | 2 | 42 | +0.0325 |
| Largest negative S1−S0 | 3 | 42 | −0.0467 |

## 14. Integrity and Reproducibility

All 18 original runs were complete. Frozen TEST membership, taxonomy ordering, supports, confusion matrices, accuracy and Macro-F1 checks passed. Post-hoc inference used the exact stored checkpoint and fold manifest for every run; all regenerated hard predictions matched the original predictions (zero mismatches), probabilities were finite and summed to one within floating-point tolerance, and ROC used continuous softmax probabilities. All nine SSL replays reproduced stored validation MSE within 1×10⁻⁶. Reconstruction targets and predictions had matching masks/shapes; no constant-target, undefined-correlation or zero-energy windows occurred.

Checkpoint hashes, manifest paths and replay checks are in `posthoc_metrics/classification_replay_audit.csv` and `posthoc_metrics/reconstruction_replay_audit.csv`.

## 15. Post-hoc Supplemental Metrics

> Precision, recall, F1, accuracy and confusion-matrix results were extracted from the original sealed TEST evaluation. ROC/AUC required continuous prediction scores that were not retained in the original TEST CSV files. Consequently, ROC/AUC was calculated post hoc by read-only inference using the exact previously selected frozen checkpoints and frozen TEST manifests. No training, checkpoint selection, hyperparameter tuning, or model modification was performed. These metrics are therefore reported as supplemental evaluation measures and were not used for model selection.

> Sample-level masked log-STFT reconstruction outputs were regenerated post hoc from the exact frozen SSL checkpoints, validation manifests, preprocessing and deterministic validation masks. No training or checkpoint selection occurred. Regenerated MSE was required to match the stored validation MSE before supplemental NMSE, MAE, R², Pearson and Spearman metrics were accepted.

## 16. Conclusions From the 100% Label Experiment

S1 did not improve the overall equal-dataset Macro-3 F1: its mean was 0.0015 lower than S0, and the paired sign-flip result was not significant. Dataset-level mean F1 was lower on JNU, slightly higher on HIT, and slightly higher on MaFaulDa. Variability across fold×seed cells was lower for S1 on HIT and MaFaulDa but not JNU. The reconstruction objective produced finite, reproducible masked log-STFT outputs; these results demonstrate reconstruction in the trained objective space but are not waveform-reconstruction evidence. Per-class error patterns are reported descriptively in the tables and six seed-balanced confusion matrices.
