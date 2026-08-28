# methodology_v2 — Part 5D: historical planning registry

This directory preserves the **pre-execution Part 5D planning freeze** for provenance. It is not the final dissertation-facing list of label regimes.

The planning registry contained 9 SSL runs plus 90 possible downstream rows spanning five candidate fractions (5%, 10%, 25%, 50%, 100%). **The 25% and 50% conditions were not executed and are not used or reported in the dissertation.** They remain in this sealed historical registry only so that the original master hash and pre-execution provenance remain verifiable.

The experiment actually used and reported in the dissertation contains exactly four downstream label regimes:

- **1%** — executed later through `scripts/run_methodology_v2_1pct_extension.py`
- **5%** — `scripts/run_methodology_v2_5pct.py`
- **10%** — `scripts/run_methodology_v2_10pct.py`
- **100%** — `scripts/methodology_v2/experiment_executor.py`

For each executed fraction there are 18 S0/S1 runs: 2 arms × 3 folds × 3 seeds (42, 1337, 2026), giving 9 paired fold–seed cells. The executed protocol uses the same four datasets, optimizer/training recipe and four-domain validation checkpoint-selection rule.

The final dissertation-facing protocol is copied to `configs/dissertation/experiment_protocol.yaml`; that file lists only 1%, 5%, 10%, and 100%.

The sealed planning artifacts below retain their historical contents unchanged: deterministic nested label subsets, sampler specifications, frozen optimization recipe, metric definitions, and the original Part 5D hash chain.

MASTER HASH: a1470bbb2f3d9c324e1c2a1ce06f177a462f905a9b491a0dc422abcfc40fdbac
(`part5d_hashes.csv`; `experiment.registry.verify_part5d_hash()` fails closed).

Code: `src/methodology_v2/experiment/`. Smoke verification: `smoke_test_report.json` (`NOT_AN_EXPERIMENT`; weights discarded). Full historical planning detail: `PART5D_EXPERIMENT_REGISTRY_REPORT.md`.
