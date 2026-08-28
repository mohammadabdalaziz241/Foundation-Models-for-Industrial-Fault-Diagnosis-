# Executed methodology

The authoritative PC-STE scientific record is the frozen material in `methodology_v2/` together with the implementation in `src/methodology_v2/`. The final controlled InceptionTime comparison is implemented separately in `src/baselines/` and `scripts/baselines/`, with its compact frozen protocol and result evidence under `results/baselines/inceptiontime_four_domain/`.

SSL sampled CWRU, JNU, HIT and MaFaulDa equally and selected checkpoints by four-dataset MacroDomain reconstruction MSE. Downstream S0/S1 sampled the same four datasets, constructed four heads, optimized the equal mean of per-dataset CE losses, selected maximum four-domain validation MacroDomainF1, then touched TEST once after sealing. The dissertation's principal downstream aggregate is Macro-4, the equal mean of CWRU, JNU, HIT and MaFaulDa Macro-F1. Macro-3 is retained only as a historical secondary aggregate over JNU, HIT and MaFaulDa.

The downstream label-efficiency comparison uses exactly 1%, 5%, 10%, and 100% labels, with folds 1–3 and seeds 42, 1337, and 2026. S0 is trained from random initialization; S1 starts from the matched SSL checkpoint and is then fully fine-tuned. Apart from initial encoder weights, the paired S0/S1 protocol holds architecture, partitions, labelled subsets, heads, sampling, optimization and checkpoint selection constant.

K1 is the four-block unidirectional `half_4x1` student initialized from same-cell S1. It used same-fold three-seed S1 teacher ensembles, hard CE, temperature-scaled forward KL (`T=4`, `alpha=0.5`) and mixer-attention relational KL (weight 1.0). Q8 and packed CPU INT8 are distinct representations: the former is the predictive weight-only quantized model state reported for classification/storage, whereas packed dynamic INT8 is used for the CPU runtime measurement.

## Controlled external baseline

The dissertation evaluates a supervised InceptionTime-style waveform network under the same frozen downstream evaluation conditions as the full-label PC-STE comparison. The baseline contains six Inception modules with residual connections, bottleneck dimension 32, 32 filters per convolution branch, kernel sizes 40/20/10, global average pooling, and four dataset-specific classification heads.

It operates directly on native-length one-second waveforms, whereas Full S1 operates on the log-STFT/N2 spectrogram representation after masked-reconstruction pretraining. The experiment is therefore a **complete-method comparison, not an architecture-only ablation**.

The baseline uses the same four datasets, frozen partitions, full labelled training subsets, folds 1–3, seeds 42/1337/2026, 50-epoch budget, effective batch 64, fold-specific step counts, AdamW learning-rate schedule, strict maximum four-domain validation MacroDomainF1 selector, and sealed TEST boundary. The primary paired comparison is Full S1 minus InceptionTime Macro-4 across the nine matched fold–seed cells, tested using the exact two-sided sign-flip procedure over all 512 sign assignments. K1 versus InceptionTime is reported as a secondary comparison.

The final controlled results are InceptionTime `0.2816 ± 0.0442`, Full S1 `0.7708 ± 0.0344` (paired difference `+0.4891`, 9/9 wins, `p=0.00390625`), and K1 `0.7931 ± 0.0288` (paired difference `+0.5115`, 9/9 wins, `p=0.00390625`). These results support superiority over the evaluated supervised Inception-style baseline under the common protocol adopted in the dissertation; they do not establish universal superiority over InceptionTime or over published fault-diagnosis methods evaluated under different protocols.
