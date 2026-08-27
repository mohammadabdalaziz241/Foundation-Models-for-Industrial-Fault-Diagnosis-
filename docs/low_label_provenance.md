# Low-label recovery provenance

The exact 1%, 5% and 10% launchers were recovered from the authoritative execution copies identified by the read-only cross-host audit. Source host names are retained here only as scientific recovery provenance; public-facing instructions are host-neutral.

## Executor equivalence

- Historical executor SHA-256: `63b4705b3b5953a3e8253c0641abf95c91b1fb4f2096215ae88bfb52ed9ff070`
- Publication executor SHA-256: `6d1e8cc63826d7e65603b933cb24b53f390073ec6bac2e65578b97862af21012`
- Status: **scientifically equivalent after publication-only path portability edit**

The publication executor is not byte-identical. Its only textual and AST-level semantic difference is that the historical fixed `REPO / "results" / "methodology_v2"` root is replaced by `PCSTE_RESULTS_ROOT`, with the identical repository-relative `results/` default. No dataset, model, initialization, optimizer, scheduler, sampler, loss, validation, checkpoint-selection, TEST, seed, fold or deterministic-control logic differs.

## Recovered launchers

| Fraction | Recovery host | Historical launcher SHA-256 | Publication launcher SHA-256 | Output naming |
|---|---|---|---|---|
| 1% | `otter133` | `a0a15ee286d1d2097c1d3c451a2bc3c37a42ddb0eb2559518a47903addef3331` | `e7db6cfbb662cf12de07834f1fe1736c2396a4f57931d25db6a8c5a782b2b8e5` | `methodology_v2_1pct_extension/downstream/[s0|s1]_f{fold}_s{seed}_l001` |
| 5% | `otter134` | `1fb449fac29d15a45afacb35fed6754fc6540ddb0c15e46b197c18abb4477352` | `846db345856e65c9398397687edb86ae29141be20ce9647f77fbee06fdae7154` | `methodology_v2_5pct/downstream/[s0|s1]_f{fold}_s{seed}_l005` |
| 10% | `otter146` | `0464ea80e6655b96dd6c5b9ccfe7941f0a2f4a8aa11468b68e7b036a4b206973` | `adc64c320c91410ed6a22b10af5d236d70721c66a45ca9858569533cb7d3256c` | `methodology_v2_10pct/downstream/[s0|s1]_f{fold}_s{seed}_l010` |

Publication-only launcher edits were limited to: (1) resolving fraction output roots through `PCSTE_RESULTS_ROOT` with the historical repository-relative fallback; (2) resolving primary SSL/downstream dependencies through the publication executor's results root; (3) emitting portable provenance paths when the results root is external; and (4) removing the historical single-host assertion. No scientific parameter changed.

All four label fractions (1%, 5%, 10%, 100%) belong to the registered experimental design; they are specified in <path to frozen spec>. The 1% run directories carry a legacy internal status string POST_HOC. That string is a misnomer and is corrected here: it records only that the 1% grid was executed after the 100%, 5% and 10% grids for scheduling reasons on shared compute, not that the condition was introduced in response to any observed result. No TEST result influenced the choice of label fractions. The _1pct_extension launcher and output-directory names are retained unchanged so that the recorded artifact paths and hashes remain valid; "extension" there denotes execution order, not a change of scope.

## Protocol and artifacts

For every fraction: folds are 1, 2 and 3; seeds are 42, 1337 and 2026; arms are S0 and S1; there are 18 runs and 9 paired cells; downstream training lasts 50 epochs. CWRU, JNU, HIT and MaFaulDa contribute CE and encoder gradients through four dataset-specific heads. Checkpoints maximize four-domain validation MacroDomainF1 with strict improvement, retaining the earlier epoch on exact ties. TEST is evaluated once after checkpoint sealing. Principal dissertation reporting uses Macro-4, averaging CWRU, JNU, HIT and MaFaulDa. Historical Macro-3 summaries average JNU, HIT and MaFaulDa only.

Original compact result tables are in `results/tables/low_label/`. Original pairing proofs are in `methodology_v2/low_label_provenance/`; all 27 rows record matching S0/S1 subset hashes and batch-stream hashes. No checkpoint, prediction file, score cache, epoch history, log, run directory or archive is included. The invalid `s0_f1_s42_l005_INTERRUPTED_MISNAMED` directory was explicitly excluded.
