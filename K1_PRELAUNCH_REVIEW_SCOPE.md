# K1 pre-launch review snapshot

This branch is a REVIEW-ONLY snapshot.

It is not the experimental branch and must not be used as a replacement
for the frozen local experiment history.

Exact source commit reviewed:

`c473fde3891922ef1716b9f4550f10ec6da09e55`

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

## Post-review integrity hardening

Before any K1 training began, the production executor received one
integrity-only patch:

- apply the same frozen native-12 preflight/byte gate used by final-v1;
- require exact TRAIN+VAL teacher-cache window/dataset/split identity.

This changed no architecture, loss, teacher policy, seed, batch stream,
optimizer, scheduler, epoch count, checkpoint selector or TEST policy.

Final executor SHA-256:

`1f5d86a1c6b7f2b637ecdd6d61c279a0b86e9edb54bdbe0490d2a4b20fd62308`
