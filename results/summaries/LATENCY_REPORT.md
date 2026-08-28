# Final four-domain Part-6 efficiency report

This is the dissertation-facing summary for Tables 5.10 and 5.11. It replaces
the earlier three-domain aggregation as the primary summary; that earlier
summary remains beside this file with `THREE_DOMAIN_HISTORICAL` in its name.

## Protocol

- Datasets: CWRU, JNU, HIT, MaFaulDa, equally weighted.
- Matched cells: folds 1–3 × seeds 42, 1337, 2026 (nine cells).
- Batch size: 1.
- Per cell/dataset/configuration: 50 warm-ups and 500 timed forwards.
- Total: 9,000 warm-up and 90,000 timed forwards.
- CPU: Full S1 FP32, K1 FP32, packed Q8(K1) INT8.
- GPU: Full S1 FP32 and K1 FP32.
- Validation representations only; no TEST data were opened.
- Latency SD is the sample SD (`ddof=1`) across nine within-cell four-domain means.

## FLOPs and scan steps

| Model | Four-domain GFLOP/window | Reduction vs Full S1 |
|---|---:|---:|
| Full S1 | 3.058269056 | — |
| K1 | 1.679675264 | 45.077584% |

| Dataset | Full S1 scan steps | K1 scan steps | Reduction |
|---|---:|---:|---:|
| CWRU | 184 | 92 | 50% |
| JNU | 192 | 96 | 50% |
| HIT | 192 | 96 | 50% |
| MaFaulDa | 192 | 96 | 50% |

## Per-dataset latency

Values are means ± sample SD across the nine fold-seed cell means.

| Dataset | Full S1 CPU ms | K1 CPU ms | Packed Q8 CPU ms | Full S1 GPU ms | K1 GPU ms |
|---|---:|---:|---:|---:|---:|
| CWRU | 23.968042992444445 ± 0.7234124567924715 | 12.619483440222222 ± 0.37688538538115895 | 12.170366542666667 ± 0.07910518063579644 | 7.461362009684245 ± 0.035716799747613465 | 4.0637667299906415 ± 0.011142809983990176 |
| JNU | 24.79760159688889 ± 0.805074048814402 | 13.057838558 ± 0.36252484370472804 | 12.615295098888888 ± 0.11861471222405263 | 7.734294946564568 ± 0.02984670224885769 | 4.1836027821434865 ± 0.015896812375551534 |
| HIT | 15.334296671333334 ± 0.32189040215072184 | 8.143243260444443 ± 0.1583704404205818 | 8.172971847777777 ± 0.043333066985789384 | 7.71905268383026 ± 0.03676371599765754 | 4.179123271518283 ± 0.01909431010993341 |
| MaFaulDa | 24.857536982 ± 0.8900879141937909 | 13.097152605333333 ± 0.3109117971935335 | 12.606080739333333 ± 0.11938611182690738 | 7.740680398411221 ± 0.02895113867639666 | 4.185274929470486 ± 0.01709026953892609 |

## Four-domain latency

Each cell is first averaged equally over the four datasets; the table then
reports mean ± sample SD across the nine matched cell means.

| Configuration | Latency ms/window | Speed-up vs Full S1 | Reduction vs Full S1 |
|---|---:|---:|---:|
| Full S1 FP32 CPU | 22.239369560666667 ± 0.6728327613850233 | 1.0× | 0.0% |
| K1 FP32 CPU | 11.729429466000001 ± 0.2985267903329993 | 1.896032× | 47.258265% |
| Packed Q8(K1) CPU | 11.391178557166667 ± 0.07278868534948313 | 1.952333× | 48.779220% |
| Full S1 FP32 GPU | 7.663847509622574 ± 0.029469210858758463 | 1.0× | 0.0% |
| K1 FP32 GPU | 4.1529419282807245 ± 0.013052227290146436 | 1.845402× | 45.811266% |

No genuine packed INT8 CUDA execution path exists. The CUDA-compatible Q8
simulation dequantizes to FP32 and is not reported as packed INT8 GPU latency.

## GPU memory

Peak allocated memory is represented by the maximum over all four datasets and
nine cells, not an average: Full S1 33.562500 MiB; K1 28.564453 MiB
(approximately 14.89% lower).

## Packed-Q8 agreement audit

On 576 deterministic validation windows, packed dynamic Q8 and the registered
weight-only-Q8/FP32-compute representation produced the same predicted class
for 571 windows (99.131944%): CWRU had four disagreements, JNU zero, HIT zero,
and MaFaulDa one. Maximum absolute logit deviation was 3.987194061. This is
**high class-level agreement**, not exact parity and not identical predictions.

## Evidence

- `../methodology_v2/part6_compression/latency_four_domain/four_domain_efficiency.json`
- `../methodology_v2/part6_compression/latency_four_domain/per_dataset_efficiency.csv`
- `../methodology_v2/part6_compression/latency_four_domain/latency_by_cell.csv`
- `../methodology_v2/part6_compression/latency_four_domain/latency_raw.csv`
- `../methodology_v2/verification/four_domain_verification.md`
