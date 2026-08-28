# Results

This directory contains the curated result tables, figures, summaries and controlled-baseline evidence used for the dissertation.

## Primary aggregate

The dissertation reports **Macro-4** as the principal cross-dataset metric:

**Macro-4 = (F1_CWRU + F1_JNU + F1_HIT + F1_MaFaulDa) / 4**

This matches the executed four-domain downstream protocol, in which CWRU, JNU, HIT and MaFaulDa all contribute supervised loss through dataset-specific heads.

| Model | Macro-4 | Historical Macro-3 |
|---|---:|---:|
| Full S0 | 0.7711 ± 0.0330 | 0.9214 |
| Full S1 | 0.7708 ± 0.0344 | 0.9199 |
| K1 | 0.7931 ± 0.0288 | 0.936913 |
| Q8(K1) | 0.793382 | 0.937011 |
| InceptionTime baseline | 0.2816 ± 0.0442 | — |

The historical **Macro-3** values average JNU, HIT and MaFaulDa only. They are retained as secondary results from an earlier reporting stage and do not represent three-domain training; both Macro-4 and the historical Macro-3 summaries for PC-STE are derived from models trained under the same executed four-domain protocol.

## Controlled InceptionTime comparison

The completed full-label baseline evidence is under [`baselines/inceptiontime_four_domain/`](baselines/inceptiontime_four_domain/). The baseline was evaluated in nine matched cells (folds 1–3 × seeds 42, 1337 and 2026) with the same frozen downstream partitions, 50-epoch budget, validation-only four-domain checkpoint selector and sealed TEST boundary used for the controlled PC-STE comparison.

Full S1 exceeded InceptionTime in all 9/9 cells. The mean paired Macro-4 difference was **+0.4891**, with exact two-sided sign-flip **p = 0.00390625**. K1 also exceeded InceptionTime in all 9/9 cells, with paired difference **+0.5115** and the same exact p-value.

This is a **complete-method comparison**, not an architecture-only ablation. InceptionTime consumes native-length raw one-second waveforms, whereas Full S1 uses the frozen log-STFT/N2 spectrogram representation and SSL initialisation. The result therefore supports the dissertation's claim only under the common experimental protocol adopted here; it is not a universal ranking against published InceptionTime or other fault-diagnosis methods.

The InceptionTime dataset-level means are CWRU `0.1440 ± 0.0823`, JNU `0.2835 ± 0.1083`, HIT `0.4762 ± 0.2233`, and MaFaulDa `0.2228 ± 0.0817`.

## Four-domain efficiency

The final efficiency evidence is under [`methodology_v2/part6_compression/latency_four_domain/`](methodology_v2/part6_compression/latency_four_domain/), with a read-only verification report at [`methodology_v2/verification/four_domain_verification.md`](methodology_v2/verification/four_domain_verification.md). It uses nine matched cells and an equal-domain mean over CWRU, JNU, HIT and MaFaulDa. The dissertation-facing Full S1/K1 CPU latencies are `22.2394 ± 0.6728`/`11.7294 ± 0.2985` ms, and GPU latencies are `7.6638 ± 0.0295`/`4.1529 ± 0.0131` ms. The older three-domain latency files are retained as explicitly labelled historical provenance.

## Training programme

The final audited programme contains **99 successful training runs** and approximately **661.01 summed GPU-hours**. The controlled InceptionTime baseline contributes nine runs, averaging **10.17 h/run** and totalling **91.49 GPU-hours**. Its timing evidence is included in `baselines/inceptiontime_four_domain/training_cost.csv`.

For detailed provenance, see [`PROVENANCE.md`](PROVENANCE.md). Curated PC-STE numerical outputs are under [`tables/`](tables/), figures under [`figures/`](figures/), compact reports under [`summaries/`](summaries/), and controlled-baseline evidence under [`baselines/inceptiontime_four_domain/`](baselines/inceptiontime_four_domain/).
