# PART 5D — Final S0/S1 Experimental Registry, Training Protocol, Few-Shot Manifests, and Smoke Tests
## methodology_v2 — the complete frozen experiment (NOT launched)

Tests: **167 passed** suite-wide (14 new Part-5D tests). All smoke
artifacts are labelled `NOT_AN_EXPERIMENT`; every smoke weight was
discarded; no epoch was run; no validation performance inspected; no
TEST signal touched.

## 1. Executive summary

The complete experimental matrix is frozen and sealed under master hash
`a1470bbb2f3d9c324e1c2a1ce06f177a462f905a9b491a0dc422abcfc40fdbac`:
9 SSL pretraining runs + 90 downstream runs (2 arms × 5 fractions ×
3 folds × 3 seeds), with deterministic nested label subsets (9 sealed
manifests), frozen samplers, one shared optimizer recipe, frozen
checkpoint criteria, pre-registered paired statistics, and bounded smoke
verification of every pipeline component. One mechanical finding: the
effective batch 64 requires **micro-batch 32 × gradient accumulation 2**
on this GPU (18.2 GB peak; exact-loss-preserving chunking). One honest
compute finding: measured step time (~2.5 s) makes the full 99-run
matrix ≈ 30 GPU-days sequential on the reference backend — a launch-
scheduling decision, not a protocol change (§22).

## 2. Part-5C checkpoint commit

**`c54c702de41f0c15f26117e1e57a31c31567abe2`** ("methodology-v2: freeze
ssl pretraining design") — 14 files, Part-5C only, seals verified and
153 tests green pre-commit, unrelated files excluded, not pushed.
Frozen Part-5C spec hash: `f5c65a7e021d…` (full value in registries).

## 3. Frozen hypotheses

**Primary (confirmatory): S1 improves MacroDomainF1_test over S0 at
100 % labels.** Secondary: label-efficiency at 5/10/25/50 %.
Exploratory: ablations A1–A4. The experiment will not be redesigned if
the primary hypothesis fails.

## 4. Final S0 definition

Random PC-STE encoder (seed-initialized) → four dataset-specific
`Linear(192, k)` heads (shared deterministic head seed) → supervised
training under the frozen recipe on the selected labelled subset.

## 5. Final S1 definition

PC-STE → masked-reconstruction SSL (frozen Part-5C objective) on 100 %
of TRAIN signals (no labels) → best MacroDomainReconMSE validation
checkpoint → decoder discarded → the SAME heads (same head seed) →
the IDENTICAL supervised training as S0 on the identical labelled
subset.

## 6. S0/S1 fairness proof

`encoder_config(S0) == encoder_config(S1)` (one PCSTEConfig; hash
`962bb1de…`); `heads_config(S0) == heads_config(S1)` (one DatasetHeads
class; paired-head-init equality smoke- and unit-tested); same subsets
(hash-matched in the registry per pair), same sampler code and seed,
same optimizer/schedule/metrics/micro-batching. Only encoder
initialization differs. Unavoidable difference documented: S1's encoder
has consumed pretraining randomness, so its downstream RNG stream
cannot be byte-identical to S0's — all shareable randomness (subset,
head init, batch sequence, masks) is shared.

## 7. Heads

CWRU `Linear(192,3)` (inner_race, outer_race, ball — fault_type field);
JNU `Linear(192,4)` (n, ib, ob, tb); HIT `Linear(192,3)` (0, 1, 2);
MaFaulDa `Linear(192,10)` (frozen folder taxonomy, unmapped). Verified
against the sealed manifests (test), single linear layers only.

## 8. Seeds / folds

Seeds **42, 1337, 2026** control encoder/SSL/decoder/head init, sampler
and mask randomness, and subset construction; folds 1–3 are the sealed
Part-2 global folds. 9 fold×seed cells; 9 paired S0/S1 comparisons per
fraction.

## 9–10. Label fractions and nested subsets

5/10/25/50/100 % of TRAIN labels only (validation/test unchanged; SSL
always uses 100 % of TRAIN signals unlabelled). Per fold×seed×dataset×
class: windows ranked by SHA256(fold|seed|dataset|class|group|window),
selection order = round-robin over groups, fraction f = first
ceil(f·N_class) positions → nesting holds by construction
(test-verified 5⊂10⊂25⊂50⊂100), every class ≥1 window, group coverage
maximal (test). 9 sealed manifests (12,850–13,073 rows each) +
`label_subset_hashes.csv`; future training fails closed on them.
**Disclosed deviation**: smallest classes realise more than requested
at 5 % (e.g. JNU fault classes with 9 windows → 1 window = 11.11 %);
full table in `label_fraction_registry.csv`.

## 11. SSL sampler

Label-free `dataset → group → window`, P(dataset)=0.25, 16×4=64 with
replacement. The sampler structurally cannot see labels (constructor
rejects any manifest view containing label columns — test-enforced).

## 12. Supervised sampler

`dataset → class → group → window` over the selected labelled subset
only; deterministic class cycling balances classes over steps (exact
per-batch equality not forced); groups uniform within class; 16×4=64;
plain unweighted cross-entropy; loss = mean over the four per-dataset
mean CEs (equal dataset influence regardless of batch mechanics).

## 13. SSL training schedule (frozen)

AdamW(3e-4, β=(0.9,0.95), eps=1e-8, wd=0.05), clip 1.0, 60 epochs,
5-epoch linear warm-up → cosine to 1e-6, no early stopping, full-length
runs with retrospective checkpoint selection. Epoch = fixed steps:
ceil(fold TRAIN windows/64) = **202/205/201** (folds 1/2/3). TRAIN
masks vary by (seed, epoch, window); **validation masks fixed per seed**
(sha256("valmask|"+seed), epoch-free — test-enforced).

## 14. Downstream training schedule (frozen, identical S0/S1)

Same AdamW recipe, single LR for encoder+heads (no discriminative LR),
50 epochs, warm-up 5, cosine to 1e-6, no early stopping.
**steps_per_epoch always from the FULL 100 % TRAIN set** (202/205/201)
for every label fraction — low fractions revisit labelled windows via
replacement (documented); validation checkpoint selection protects
against terminal overfitting.

## 15. Validation/checkpoint rules

SSL: minimize **MacroDomainReconMSE** = mean of per-dataset window-mean
masked-cell MSEs (never pooled cells/windows). Downstream: maximize
**MacroDomainF1_val** = mean of the four validation Macro-F1s; exact
ties → earlier epoch. TEST is never consulted.

## 16. Test-evaluation rule

Train → validation checkpoint → freeze → **one** TEST evaluation. No
test-driven epoch/LR/mask/architecture decisions; no re-runs on
"suspicious" scores without a proven, documented implementation
failure. Run states REGISTERED/RUNNING/COMPLETE/FAILED; failed runs may
restart only with the identical registry configuration.

## 17. Metrics

Frozen numpy implementations (sklearn-agreement tested): accuracy,
per-class precision/recall/F1 (zero-division := 0.0 explicit),
confusion matrix, Macro-F1 with the frozen class orders, absent classes
never dropped. **Primary: MacroDomainF1_test** (each dataset exactly
25 %). Per-dataset scores always reported alongside; S1 additionally
reports the frozen SSL reconstruction metrics; parameter counts and
training/inference cost recorded per run.

## 18. Statistical analysis (pre-registered)

Paired Δ = MacroDomainF1(S1) − MacroDomainF1(S0) over the 9 fold×seed
cells per fraction. Report mean S0/S1, mean/median/SD of Δ, all nine
Δs, effect size. Primary inference: **exact two-sided paired sign-flip
permutation test (2⁹ = 512 flips) at 100 % labels**; lower fractions
secondary with Holm correction if inferential p-values are reported
across fractions. Documented caveat: fold×seed cells are paired but not
fully independent — conclusions stay cautious; interpretation never
reduces to significance alone.

## 19. Complete run count

9 SSL + 90 downstream (45 S0 + 45 S1) = **99 main runs**, all
REGISTERED with deterministic IDs (`ssl_f{f}_s{s}`,
`{arm}_f{f}_s{s}_l{pct}`), full config/hash provenance per row, and
S1→SSL dependency links (pairing test-verified).

## 20. Architecture ablation registry (registered, NOT launched)

A1 coordinates→index PE · A2 mixer→masked mean · A3 BiMamba→Transformer
· A4 N2→N1 — full-label condition only; never multiplied across
fractions.

## 21. Smoke-test results (`smoke_test_report.json`, NOT_AN_EXPERIMENT)

- SSL: 2 steps (losses 1.04→1.15 — recorded only as "runs and is
  finite", not as a diagnostic); **mixer gradients nonzero**.
- S0: 2 steps through all four heads; **all heads received gradients**.
- S1 loading: smoke encoder state loads exactly; paired head inits
  bit-identical; supervised step runs after load.
- Feasibility: micro 64 OOMs on the 20 GB GPU (reference-scan backward
  graph); **micro 32 × accumulation 2 works** (18.2 GB peak,
  ~2.5 s/step) with exact-loss-preserving dataset-aligned chunking;
  identical for S0 and S1. All weights discarded.

## 22. Compute feasibility (honest, measured)

At ~2.5 s/step (reference backend, micro 32×2): SSL run ≈ 8.5 h → 9
runs ≈ **77 h**; downstream run ≈ 7.1 h → 90 runs ≈ **~27 days**
sequential; full matrix ≈ **~30 GPU-days** on the single RTX 4000 Ada —
substantially above the Part-5C forward-only estimate (backward through
the reference scan is the difference; per-step window loading adds
more). The **primary confirmatory subset** (9 SSL + 18 downstream
@100 %) ≈ **8.5 days**. Pre-launch options (decisions for the human,
not taken here): official fused Mamba kernels once torch-2.12 wheels
exist (backbone dominates cost); a deterministic derived representation
cache (permitted by Part 4C); prioritized launch order (primary
comparison first, few-shot after). No frozen parameter was changed.

## 23. Master Part-5D hash

**`a1470bbb2f3d9c324e1c2a1ce06f177a462f905a9b491a0dc422abcfc40fdbac`**
over protocol/optimizer/heads/sampler/metric/statistics/ablation specs,
both run registries, the fraction registry and all 9 subset manifests
(`part5d_hashes.csv`); `verify_part5d_hash()` fails closed
(tamper-tested). Reproduced identically across two full runner
executions.

## 24. Git status

HEAD `c54c702d`; Part-5D additions untracked
(`methodology_v2/part5_experiment_registry/`,
`src/methodology_v2/experiment/`, `scripts/methodology_v2/run_part5d.py`,
`tests/methodology_v2/test_part5d_registry.py`); unrelated legacy files
untouched; nothing committed (no authorization yet).

## 25. Remaining issues before launch

1. **Compute schedule** (§22): choose sequential order / caching /
   kernel strategy for the ~30-GPU-day matrix, or authorize the primary
   subset first.
2. Commit the Part-5D registry (and decide on pushing the methodology
   branch).
3. The 5 %-fraction over-realisation on tiny classes (§9–10) is
   disclosed — confirm acceptance.
4. Real-run checkpoint storage location and retention policy
   (`results/methodology_v2/...` paths are registered; disk budget to
   confirm).

**HARD STOP.** No 60-epoch SSL run, no 50-epoch downstream run, no
few-shot experiment, no ablation, no TEST evaluation was launched.
Awaiting explicit human authorization to launch the frozen registry.
