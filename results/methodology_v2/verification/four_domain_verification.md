# Four-domain efficiency and diagnostic verification

## Items that could not be completed

1. **A5 window-level disagreement diagnostics:** the parity implementation aggregates each 16-window block immediately and retains only count-level agreement, maximum absolute logit difference, and mean absolute logit difference. It does not retain window IDs, true labels, either predicted class, or individual logits. Therefore the true class, both predicted classes, and top-two logit margins for disagreeing windows cannot be recovered without rerunning inference, which was prohibited.
2. **B macro-AUC:** S0/S1 post-hoc continuous-score AUC is retained only for JNU, HIT, and MaFaulDa. No retained S0/S1 CWRU AUC and no K1/Q8(K1) continuous-score AUC were found. Hard-label prediction CSVs cannot validly reconstruct ROC-AUC, so those entries are reported as unavailable.
3. **C3 training-set classification metrics:** the InceptionTime epoch histories record mean training loss and validation metrics, but no training-set macro-F1 or accuracy. Only two of the nine complete 50-epoch histories are present in the local public result tree (`f1/s2026` and `f2/s42`); the compact nine-cell TEST summaries remain present. No inference was rerun to fill these gaps.

All other requested checks were completed from existing code and artifacts. No training or benchmark was run.

## A. Four-domain benchmark verification

### A1. Hardware identity

Sources: the sealed three-domain `latency/latency_metadata.json` and the new four-domain `latency_four_domain/latency_metadata.json`.

| Field | Sealed three-domain run | New four-domain run | Identical? |
|---|---|---|---|
| Hostname | `[redacted]` | `[redacted]` | Yes before publication redaction |
| CPU | `Intel(R) Core(TM) i9-14900` | `Intel(R) Core(TM) i9-14900` | Yes |
| CPU physical/logical cores | 24 / 32 | 24 / 32 | Yes |
| GPU | `NVIDIA RTX 4000 Ada Generation` | `NVIDIA RTX 4000 Ada Generation` | Yes |
| PyTorch intra-op threads | 4 | 4 | Yes |
| PyTorch inter-op threads | 1 | 1 | Yes |
| PyTorch | `2.12.0+cu130` | `2.12.0+cu130` | Yes |
| CUDA | `13.0` | `13.0` | Yes |

**Answer:** every requested hardware/runtime identity field was recorded, and the two runs used identical recorded hardware, hostname, thread settings, PyTorch, and CUDA versions. The institutional hostname is redacted in this public copy only.

### A2. Pass counts

`benchmark_part6_latency_four_domain.py` imports the sealed implementation, sets four datasets, and invokes its unchanged `main()`. The actual parser defaults are 50 warm-ups and 500 timed iterations. The loop is 3 folds × 3 seeds = 9 cells, then the selected device/model configurations, then 4 datasets.

| Quantity | Exact value |
|---|---|
| Warm-up forwards per cell/dataset/configuration | 50 |
| Timed forwards per cell/dataset/configuration | 500 |
| Cells | 9: folds 1, 2, 3 × seeds 42, 1337, 2026 |
| Datasets | 4: CWRU, JNU, HIT, MaFaulDa |
| CPU configurations | Full S1 FP32; K1 FP32; packed Q8(K1) INT8 |
| GPU configurations | Full S1 FP32; K1 FP32 |
| Q8 GPU | Not timed; no genuine packed INT8 CUDA implementation |
| `warm-up_total` | 50 × 9 × 4 × 5 = **9000** |
| `timed_total` | 500 × 9 × 4 × 5 = **90000** |

### A3. Per-dataset efficiency breakdown

Latency is the mean ± sample SD across the nine fold/seed cell means, calculated from `latency_by_cell.csv`. FLOPs and scan steps are per one-second window. Temporal patch positions are read from the recorded validation input geometry.

| Dataset | Spectrogram shape | Temporal patch positions | Full S1 scan steps | K1 scan steps | Full S1 GFLOP/window | K1 GFLOP/window | Full S1 CPU ms | K1 CPU ms | Packed Q8 CPU ms | Full S1 GPU ms | K1 GPU ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CWRU | 513 × 184 | 23 | 184 | 92 | 3.375500928 | 1.854049920 | 23.968042992444445 ± 0.7234124567924715 | 12.619483440222222 ± 0.37688538538115895 | 12.170366542666667 ± 0.07910518063579644 | 7.461362009684245 ± 0.035716799747613465 | 4.0637667299906415 ± 0.011142809983990176 |
| JNU | 513 × 192 | 24 | 192 | 96 | 3.521685504 | 1.934137344 | 24.79760159688889 ± 0.805074048814402 | 13.057838558 ± 0.36252484370472804 | 12.615295098888888 ± 0.11861471222405263 | 7.734294946564568 ± 0.02984670224885769 | 4.1836027821434865 ± 0.015896812375551534 |
| HIT | 257 × 192 | 24 | 192 | 96 | 1.814201984 | 0.996374144 | 15.334296671333334 ± 0.32189040215072184 | 8.143243260444443 ± 0.1583704404205818 | 8.172971847777777 ± 0.043333066985789384 | 7.71905268383026 ± 0.03676371599765754 | 4.179123271518283 ± 0.01909431010993341 |
| MaFaulDa | 513 × 192 | 24 | 192 | 96 | 3.521687808 | 1.934139648 | 24.857536982 ± 0.8900879141937909 | 13.097152605333333 ± 0.3109117971935335 | 12.606080739333333 ± 0.11938611182690738 | 7.740680398411221 ± 0.02895113867639666 | 4.185274929470486 ± 0.01709026953892609 |

### A4. Direction of the aggregate change

On CPU, CWRU ranks third-highest (second-lowest) for Full S1, K1, and packed Q8: it is slower than JNU and MaFaulDa but faster than HIT. Its CPU latency is above the old three-domain mean because the very low HIT value pulls that old mean down; adding CWRU therefore raises the equal-domain CPU aggregate.

On GPU, CWRU is the fastest dataset for both Full S1 and K1. Adding it therefore lowers the equal-domain GPU aggregate. HIT is not the fastest on GPU despite having roughly half the frequency positions, because it retains the same 192/96 sequential scan-step counts as JNU and MaFaulDa.

Pearson correlations across the four dataset means are:

| Model | GPU latency vs scan steps | CPU latency vs GFLOP/window |
|---|---:|---:|
| Full S1 | 0.9977489293992992 | 0.9999740152199843 |
| K1 | 0.999047148661088 | 0.9999504724049434 |

**Hypothesis:** confirmed for these four geometries. GPU latency tracks sequential scan steps very closely, while CPU latency tracks the dense arithmetic/FLOP burden over frequency bands very closely. This is a four-point descriptive correlation, not causal proof.

### A5. Packed-Q8 parity breakdown

Five of 576 windows disagreed: 0.8680555555555556% overall.

| Dataset | Disagreeing / checked | Disagreement percentage |
|---|---:|---:|
| CWRU | 4 / 144 | 2.7777777777777777% |
| JNU | 0 / 144 | 0.0% |
| HIT | 0 / 144 | 0.0% |
| MaFaulDa | 1 / 144 | 0.6944444444444444% |

Only three fold/seed/dataset blocks had disagreements:

| Fold | Seed | Dataset | Disagreeing / 16 | Cell disagreement percentage | Cell maximum absolute logit deviation |
|---:|---:|---|---:|---:|---:|
| 1 | 42 | CWRU | 3 / 16 | 18.75% | 0.6158218383789062 |
| 2 | 2026 | CWRU | 1 / 16 | 6.25% | 3.3920068740844727 |
| 3 | 2026 | MaFaulDa | 1 / 16 | 6.25% | 1.0160880088806152 |

The other 33 fold/seed/dataset blocks had 0/16 disagreements. The global maximum absolute logit deviation, 3.987194061279297, occurred in **CWRU, fold 2, seed 42**, but that 16-window block had 16/16 predicted-class agreement. The retained aggregate cannot identify its true class or predicted classes. It also cannot provide top-two margins for the five disagreeing windows. Those details were never persisted.

**Concentration answer:** disagreements are concentrated on CWRU (4/5, 80.0%); MaFaulDa accounts for 1/5 (20.0%), and JNU/HIT account for none.

### A6. Contamination check

The new aggregate latency and speed-up values do not equal the sealed three-domain aggregate values. The raw four-domain CSV contains 90000 newly measured timing rows, and the compact JSON's new aggregate is recomputable from its four per-dataset entries. Thus no three-domain aggregate latency result was copied into the new aggregate.

However, it would be false to say the JSON has no identical value whatsoever. The following figures legitimately appear unchanged because the protocol/hardware or input geometry was deliberately preserved:

- folds 1, 2, 3 and seeds 42, 1337, 2026;
- batch size 1, 50 warm-ups, and 500 timed iterations;
- 4 intra-op threads and 1 inter-op thread;
- CPU/GPU identities, PyTorch `2.12.0+cu130`, and CUDA `13.0`;
- JNU and MaFaulDa geometry 513 × 192 and HIT geometry 257 × 192;
- HIT maximum allocated memory: Full S1 25.58642578125 MiB and K1 21.15087890625 MiB, identical under the same model/input geometry.

These are invariant protocol/geometry measurements, not carried-over three-domain aggregate results.

## B. Chance-level reference values

For class prior pᵢ and K classes, the uniform-random population reference uses prediction probability qᵢ=1/K and class F1 `2pᵢqᵢ/(pᵢ+qᵢ)`, averaged over classes. The majority reference predicts only the largest-support class and averages its `2p/(1+p)` F1 with K−1 zeros. Supports and priors are identical across seeds within each fold. Reported model results are mean ± sample SD over the nine cells. Chance macro-AUC is 0.5.

| Dataset | Fold supports; priors | Uniform-random macro-F1 by fold | Majority macro-F1 by fold | Chance macro-AUC | S0 macro-F1; macro-AUC | S1 macro-F1; macro-AUC | K1 macro-F1; macro-AUC | Q8(K1) macro-F1; macro-AUC |
|---|---|---|---|---:|---|---|---|---|
| CWRU | F1 [35,35,35]; [0.3333333333333333,0.3333333333333333,0.3333333333333333]; F2 [35,99,35]; [0.20710059171597633,0.5857988165680473,0.20710059171597633]; F3 [28,99,35]; [0.1728395061728395,0.6111111111111112,0.21604938271604937] | F1 0.3333333333333333; F2 0.31194720299071665; F3 0.30706237002885556 | F1 0.16666666666666666; F2 0.24626865671641787; F3 0.25287356321839083 | 0.5 | 0.3200104984976347 ± 0.09022975346307531; unavailable | 0.32340077806834333 ± 0.1286321228422148; unavailable | 0.361762184774925 ± 0.11616967413569655; unavailable | 0.3624945555637929 ± 0.11635516769229437; unavailable |
| JNU | F1/F2/F3 [15,3,3,3]; [0.625,0.125,0.125,0.125] | F1/F2/F3 0.21428571428571427 | F1/F2/F3 0.19230769230769232 | 0.5 | 0.9904761904761905 ± 0.028571428571428543; 1.0 ± 0.0 | 0.980952380952381 ± 0.037796447300922686; 1.0 ± 0.0 | 1.0 ± 0.0; unavailable | 1.0 ± 0.0; unavailable |
| HIT | F1/F2/F3 [112,112,56]; [0.4,0.4,0.2] | F1/F2/F3 0.32575757575757575 | F1/F2/F3 0.1904761904761905 | 0.5 | 0.974428794812082 ± 0.05186331987776982; 0.9991053713151928 ± 0.0024584014513742624 | 0.9755005263976548 ± 0.022741172349011896; 0.9957256629503655 ± 0.011772788586546954 | 0.9825088469373208 ± 0.023264550746292915; unavailable | 0.9828378073426968 ± 0.022654745883188786; unavailable |
| MaFaulDa | F1 [35,235,245,250,245,245,245,245,245,245]; [0.015659955257270694,0.10514541387024609,0.10961968680089486,0.11185682326621924,0.10961968680089486,0.10961968680089486,0.10961968680089486,0.10961968680089486,0.10961968680089486,0.10961968680089486]; F2 [35,245,245,250,190,240,185,100,245,245]; [0.017676767676767676,0.12373737373737374,0.12373737373737374,0.12626262626262627,0.09595959595959595,0.12121212121212122,0.09343434343434344,0.050505050505050504,0.12373737373737374,0.12373737373737374]; F3 [35,245,245,250,245,210,185,125,245,245]; [0.017241379310344827,0.1206896551724138,0.1206896551724138,0.12315270935960591,0.1206896551724138,0.10344827586206896,0.09113300492610837,0.06157635467980296,0.1206896551724138,0.1206896551724138] | F1 0.09673079027868944; F2 0.09553350080734882; F3 0.09599372928356995 | F1 0.02012072434607646; F2 0.022421524663677132; F3 0.02192982456140351 | 0.5 | 0.799367532212296 ± 0.10146851821572693; 0.9804138121245529 ± 0.014849133969814058 | 0.8033104128068107 ± 0.06262399187878731; 0.9874195915875088 ± 0.007446658028077015 | 0.8282287685255804 ± 0.05910797132711021; unavailable | 0.8281953475539802 ± 0.05902273616520959; unavailable |

## C. InceptionTime diagnostics

### C1. Architecture and receptive field

The model has six sequential Inception modules. Every module has a 1×1 bottleneck to 32 channels; three convolution branches with kernel sizes 40, 20, and 10 and 32 filters each; and a same-length max-pool-3 plus 1×1-convolution branch. Batch normalization and ReLU follow concatenation. Residual shortcuts are applied after modules 3 and 6. The final pooling is global temporal average pooling (`x.mean(dim=-1)`), producing 128 features.

With stride one, dilation one, and the longest kernel 40 in every module, the maximum theoretical receptive field is `1 + 6 × (40−1) = 235` input samples. Same-length asymmetric padding does not alter this span.

| Dataset | One-second input length | Maximum receptive field | Fraction of input covered |
|---|---:|---:|---:|
| CWRU | 48000 | 235 | 0.004895833333333334 |
| JNU | 50000 | 235 | 0.0047 |
| HIT | 25000 | 235 | 0.0094 |
| MaFaulDa | 50000 | 235 | 0.0047 |

### C2. Optimisation configuration

| Setting | InceptionTime actual configuration |
|---|---|
| Optimizer | AdamW |
| Learning rate | 0.0003 |
| Betas | (0.9, 0.95) |
| Epsilon | 1e-8 |
| Weight decay | 0.05 |
| Schedule | 5-epoch linear warm-up, then cosine decay to 0.000001 |
| Epochs | 50 |
| Effective batch | 64 = 16 examples from each of four datasets |
| Micro-batch | 2 |
| Gradient clipping | global norm 1.0 |
| Early stopping | None; all runs execute 50 epochs |
| Checkpoint selection | strict maximum validation four-domain MacroDomainF1 |

The PC-STE downstream recipe uses the same optimizer, learning rate, betas, epsilon, weight decay, warm-up/cosine schedule, minimum learning rate, 50 epochs, effective batch 64, equal four-dataset contribution, gradient clipping, and no early stopping. The necessary implementation difference is micro-batching: the InceptionTime raw-waveform run used micro-batch 2, whereas the PC-STE feasibility record used micro-batch 32. The input representation and model architecture also necessarily differ; the optimization recipe itself otherwise matches.

### C3. Fit diagnostics

No training-set F1/accuracy was recorded. The available training loss histories are:

| Cell | Epoch 1 loss | Epoch 5 | Epoch 10 | Epoch 20 | Epoch 30 | Epoch 40 | Epoch 50 | Minimum recorded loss | Behaviour |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| fold 1, seed 2026 | 1.2597910764046234 | 0.13968706534123526 | 0.031086558971955836 | 0.006453072159085108 | 0.0029717189309087897 | 0.000026310620814216 | 0.000012209870099960277 | 0.00000917465370167763 | Converged toward zero; no divergence; near-zero plateau late in training |
| fold 2, seed 42 | 1.239057071134448 | 0.13085210292865224 | 0.0179292951356805 | 0.0051683034371290725 | 0.00024169455655500765 | 0.000009977205175920736 | 0.000008755435638274091 | 0.000004882472621625125 | Converged toward zero; no divergence; near-zero plateau late in training |

The corresponding sealed TEST Macro-F1 values for these retained-history cells are:

| Cell | CWRU | JNU | HIT | MaFaulDa |
|---|---:|---:|---:|---:|
| fold 1, seed 2026 | 0.044444444444444446 | 0.2882352941176471 | 0.3308641975308642 | 0.20303484642242475 |
| fold 2, seed 42 | 0.11860940695296524 | 0.244047619047619 | 0.7131652661064426 | 0.1402102540447929 |

Across all nine cells, the retained TEST summaries are:

| Dataset | InceptionTime TEST Macro-F1 mean ± sample SD |
|---|---:|
| CWRU | 0.14399512900152767 ± 0.08227603592565919 |
| JNU | 0.28349326251695534 ± 0.10834869268566412 |
| HIT | 0.4762297674970147 ± 0.2232872103399377 |
| MaFaulDa | 0.22284749799295067 ± 0.0817404154242269 |

**Explicit answer:** InceptionTime fit the sampled training data to near-zero cross-entropy loss but failed to generalise; the retained evidence does not support the claim that it never fit at all.

## Verification status

The four-domain efficiency aggregate is internally consistent with the raw nine-cell measurements and was collected under hardware/thread settings identical to the sealed three-domain run. The limitations above concern evidence that was never retained and cannot be filled without prohibited inference; they do not invalidate the timing aggregate.
