# Part 6 — PC-STE lightweight study (compression) — protocol directory

Status: **implementation complete · TEMPLATE registry · final seal PENDING ·
no experiment started · no TEST touched.**

Plan: `docs/methodology_v2_lightweight_plan.md` (approved 2026-08-16).
Implementation guide: `docs/methodology_v2_part6_implementation.md`.
Code: `src/methodology_v2/compression/` (no epoch loops) and
`scripts/methodology_v2/part6_compression.py` (CLI/executor).
Tests: `tests/methodology_v2/test_part6_compression.py`.

Files here (JSON content, `.yaml` extension — repository convention):

| file | content | state |
|---|---|---|
| `protocol.yaml` | fixed a-priori constants (T=4, α=0.5, margins 0.02/0.01, push rule, folds/seeds/arms), frozen recipe, **pending decisions with recommendations**, TEST policy | template |
| `student_spec.yaml` | full / Student-D / Student-DW / 4×1 comparator specs, exact parameter counts, scan steps, surgery mapping | template (retained layers PENDING) |
| `quantization_spec.yaml` | Q8 recipe, allow/deny lists, fp32 tensors, deployment representations, measured-size rule, exploratory VAL-only variants | final content |
| `kd_spec.yaml` | loss forms, KL direction, reduction, relational term (weight PENDING), teacher sets, ensemble rule (PENDING), cache contents | template |
| `statistics_spec.yaml` | pre-registered families, sign-flip/NI/Holm/push rules | final content |
| `test_policy.yaml` | single sealed session, ledgers, disclosure of visible Phase-B values | final content |
| `measurement_spec.yaml` | four-axis harness definition | final content |
| `part6_run_registry.csv` | 72 rows: 27 core (enabled) + 18 optional + 27 push (disabled); status TEMPLATE / TEMPLATE_AWAITING_PRIMARY | template |
| `pending_decisions.yaml` | the open pre-registration decisions | — |
| `resolved_decisions.yaml` | **to be written by the researcher** before `seal` (keys = pending decision keys) | absent |
| `part6_hashes.csv` | written by `seal` (master hash; verified fail-closed by every gated command) | absent |
| `LAUNCH_AUTHORIZED` | **empty marker created by the researcher** to authorise execution | absent |
| `pre_test_ledger.csv` | Stage 5 step 1 (must be committed before `test-session`) | absent |
| `scan_backend_parity.json` | human-approved parity record; without it the reference scan is used | absent |

Never placed here: checkpoints, caches, results (those live under
`results/methodology_v2/part6_compression/`, gitignored).
