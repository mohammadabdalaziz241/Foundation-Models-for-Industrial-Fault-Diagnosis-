# Part 6 pre-TEST readiness

Generated after canonical Stage-3 consolidation on 2026-08-22.

- Core Stage-3 runs: **27/27 COMPLETE** (9 K1, 9 C_small, 9 K0).
- RUNNING / WAITING / FAILED / STALE: **0 / 0 / 0 / 0**.
- Static 9/9/9 host assignment, sealed registry, configuration hashes,
  checkpoint hashes, 50-epoch histories, and the frozen best-checkpoint rule
  were verified fail-closed.
- `k1_f1_s42` records parent commit `dd60fc16470e93a883565bdff7df950c26664a0a`.
  Commit `9996e96a2fd31818fc6ca0dec65e6bf6eb2705fa` explicitly documents that this
  run was already running when the scheduling-only static assignment was
  added; its scientific registry/configuration/seal hashes match.
- Scope: TRAIN/VALIDATION artifacts only. No TEST manifest, labels,
  predictions, or report was opened.

## Blocking condition

The final `pre_test_ledger.csv` is intentionally **not generated yet**.
The sealed Stage-5 implementation requires 72 rows: FP32 and Q8 variants of
all 27 enabled core models (54 rows), plus Q8 variants of the 18 primary S0/S1
models. No Stage-1 PTQ records currently exist in the canonical Part-6 results
tree. Generating the ledger now would silently claim Q8 readiness that has not
been established.

Before the ledger can be finalized, validation-side registered Q8/PTQ must be
completed for the 27 core checkpoints and the 18 primary S0/S1 checkpoints,
then all 45 PTQ records and their source checkpoint hashes must be verified.
This includes the confirmatory Q8(K1) vs K1 family and the secondary
Q8(S1) vs S1 and Q8(S0) vs S0 families.
