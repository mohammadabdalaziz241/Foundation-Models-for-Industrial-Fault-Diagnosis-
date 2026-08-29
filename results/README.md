# Results

This directory contains the dissertation-facing outputs for the four-domain PC-STE evaluation on **CWRU, JNU, HIT, and MaFaulDa**.

## Reporting convention

Final downstream performance is reported with **Macro-4**, the equal mean of the four dataset-level Macro-F1 values:

`Macro-4 = (F1_CWRU + F1_JNU + F1_HIT + F1_MaFaulDa) / 4`.

The same four-domain scope is used for the main S0-S1 evaluation, lightweight-model analysis, efficiency reporting, and controlled InceptionTime comparison.

## Main S0-S1 files

- `tables/classification_summary_s0_vs_s1.csv` — Dissertation Table 5.2: full-label dataset-level results and Macro-4 aggregate.
- `tables/paired_s0_s1.csv` — Dissertation Table 5.3: nine matched full-label Macro-4 cells.
- `tables/statistical_analysis.csv` — Dissertation Table 5.5: exact paired S0-S1 tests across 1%, 5%, 10%, and 100% labels.
- `tables/per_class_cwru_summary.csv` — CWRU full-label per-class results corresponding to the 100% rows of Dissertation Table A.1.
- `tables/per_class_jnu_summary.csv`, `per_class_hit_summary.csv`, and `per_class_mafaulda_summary.csv` — per-class summaries for the other downstream datasets.

At 100% labels, S0 achieves **0.7711 ± 0.0330 Macro-4 F1** and S1 **0.7708 ± 0.0344**, with exact paired two-sided **p = 0.9805**. At 1% labels, the mean Macro-4 F1 increases from **0.4803** for S0 to **0.5051** for S1.

## CWRU per-class result

CWRU is the most challenging downstream domain. With complete labelled data, rolling-element faults are recognised more reliably than the two race-fault classes. For S1, the full-label class-level AUC values are **0.5100** for inner-race fault, **0.3939** for outer-race fault, and **0.8143** for rolling-element fault. The corresponding class-level F1 values are **0.1764**, **0.2486**, and **0.5452**.

## Lightweight and efficiency outputs

The four-domain lightweight analysis compares Full S1, K1, and Q8(K1). Mean Macro-4 is **0.7708** for Full S1, **0.7931** for K1, and **0.7934** for Q8(K1). K1 reduces encoder parameters by 42.24%, counted forward-pass computation by 45.08%, and selective-scan processing by 50% relative to Full S1.

Four-domain runtime artefacts are under `methodology_v2/part6_compression/latency_four_domain/`. The controlled InceptionTime comparison is under `baselines/inceptiontime_four_domain/`.

## Provenance

See `PROVENANCE.md` for the correspondence between repository artefacts and the dissertation tables and sections.
