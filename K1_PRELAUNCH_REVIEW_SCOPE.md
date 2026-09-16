# K1 pre-launch review snapshot

This branch is a REVIEW-ONLY snapshot.

It is not the experimental branch and must not be used as a replacement
for the frozen local experiment history.

Exact source commit reviewed:

`8b08dfd76b014f5a7d464fe78c30fcef6ce71ae2`

Source branch:

`bearing-generalisation-v1`

Purpose:

Independent static review before the first real PCSTE-v2 K1 training run.

Scientific target:

- K1 = historical dissertation Part-6 half_4x1 student
- 4 temporal layers
- forward direction only
- 1,375,953 encoder parameters
- same-cell Full S1 initialization
- same-fold three-seed S1 teacher ensemble
- KD temperature T=4
- alpha=0.5
- relational weight=1.0
- mean_prob_at_T teacher ensemble
- 50 epochs
- exact final-v1 supervised batch stream
- strict validation MacroDomainF1 checkpoint selection
- no early stopping
- TEST blocked during teacher caching, training and validation

The large historical metadata CSV files that prevent the source branch
from being pushed to GitHub are intentionally NOT part of this review
snapshot.

Their omission does not alter the copied scientific implementation.
Runtime stream/checkpoint/protocol parity was frozen separately in:

- analysis/pcste_v2_lightweight_v1/COMPATIBILITY_FREEZE.json
- analysis/pcste_v2_lightweight_v1/K1_REGISTERED_CELLS.csv
- analysis/pcste_v2_lightweight_v1/K1_PRODUCTION_PROTOCOL.json
- analysis/pcste_v2_lightweight_v1/smoke_original_part6/SMOKE_REPORT.json

No K1 training or K1 TEST inference had begun when this review snapshot
was created.
