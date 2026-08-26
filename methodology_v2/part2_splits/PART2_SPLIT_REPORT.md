# PART 2 — Frozen Leakage-Safe Train/Validation/Test Protocol
## methodology_v2 split report

Scope: identity/region assignment only. The split precedes all future
windowing, STFT, normalization, resampling, masking, augmentation and
training. No model-ready samples exist yet. Artefacts in this directory
are sealed (SHA-256, §"Sealing"); `verify_frozen_hashes()` fails closed if
any frozen manifest changes.

## 1. Executive summary

Three frozen **global data folds** were generated (fold_id 1–3; data
partitions, NOT model seeds). Each fold assigns every retained
recording/region of CWRU, JNU, HIT and MaFaulDa to exactly one of
train / validation / test at that dataset's approved leakage-control
grouping unit:

| Dataset | Unit assigned | Units per fold (train/val/test) | Mechanism |
|---|---|---|---|
| CWRU | physical fault specimen across all loads (9) | 3 / 3 / 3 | frozen Latin rotation (exhaustive) |
| JNU | temporal macro-blocks A–E within each of 12 recordings | 36 / 12 / 12 regions | frozen block rotation, symbolic guards |
| HIT | session × speed-group recording (134) | 94 / 20 / 20 | seeded grouped split, session-stratified (seeds 101/102/103) |
| MaFaulDa | fault configuration (41) + Normal recordings (49) | 58 / 16 / 16 units | seeded grouped split, class-stratified (seeds 201/202/203) |

All predeclared seeds passed the frozen structural acceptance criteria on
the first attempt — `rejected_split_seeds.json` is empty. Reruns are
byte-identical (tested). S0 and S1 must consume these folds identically
(§9).

## 2. Dataset-specific splitting rules

### CWRU
Retained subset: 48 kHz drive-end fault family, diameters
0.007"/0.014"/0.021", classes InnerRace/Ball/OuterRace — 52 recordings.
Excluded: Healthy (single physical healthy specimen — cannot support
specimen-disjoint splitting), 0.028", the whole 12 kHz fault family.
Atomic group: **physical fault specimen across all motor loads**, OR
clock positions conservatively merged (`cwru48k_IR007` … `cwru48k_OR021`,
9 groups; per `CWRU_GROUPING_RECHECK.md`). Assignment: the approved
deterministic Latin rotation (§7) — no RNG involved. All four load
recordings (and all OR positions) of a specimen inherit one split.

### JNU
12 recordings (4 conditions × 3 speeds), too few for recording-level
coverage → **within-recording temporal holdout** (never to be described
as unseen-bearing / recording-independent / unseen-domain). Each
recording is divided into five contiguous macro-block slots A–E at
nominal boundaries `floor(i·N/5)` (N divisible by 5 for all 12 files:
fault blocks 100,100 samples = 2.002 s; healthy blocks 300,300 = 6.006 s).
Between adjacent blocks sit four **symbolic guard regions** anchored at
the internal boundaries; they instantiate in Part 3 as
`[b − ceil(G/2), b + ceil(G/2))` with `G ≥ effective window span`, carved
from the two adjacent blocks and discarded. Windows must never cross
macro-block boundaries, split boundaries, or guards. Block rotation per
fold (identical for all 12 recordings): fold 1 = ABC/D/E, fold 2 = BCD/E/A,
fold 3 = CDE/A/B → usable ratio 60/20/20 pre-guard.

### HIT
Full Google-Drive release only; the official GitHub `xtrain_*`/`xtest`
shards are **rejected as split authority** (Part-1 finding H2: all 2,340
underlying series feed both official train and test). Atomic group: the
audited **session × speed-group recording** (134 groups; every channel,
series and future window of a group inherits its split). Session-
stratified seeded allocation per fold: within each session,
`n_val = n_test = max(1, round(0.15·n))`, remainder train → 94/20/20
groups (70.1/14.9/14.9 %). Frozen acceptance criteria H1–H4 (group
disjointness; ≥2 groups per class per partition; every session in every
partition; LP-speed tertile coverage 3/2/2) — all satisfied by the
predeclared seeds 101/102/103.

### MaFaulDa
Operational folder taxonomy kept verbatim (documented naming
inconsistency M2 **not** remapped); all 10 classes retained for future
SSL and Task-B use; no forcing into the shared taxonomy. Atomic units:
**fault configuration** (41 configs) for fault classes; **original
recording** for the single-configuration Normal class (49 units —
explicitly weaker independence, §8). Class-stratified seeded allocation
per fold over units: `n_val = n_test = max(1, round(0.15·c))`, remainder
train. Because four classes have only 4 configurations, class coverage
(frozen priority 3) forces 2/1/1 there, giving global proportions ≈
56–57 % / 20–23 % / 20–23 % by recordings — the deliberate,
priority-ordered deviation from 70/15/15. Frozen criteria M1–M4 all
satisfied by predeclared seeds 201/202/203.

## 3. Exact fold statistics (groups/regions per partition)

| Fold | CWRU (spec) | JNU (regions) | HIT (groups) | MaFaulDa (units) |
|---|---|---|---|---|
| 1 | 3 / 3 / 3 | 36 / 12 / 12 | 94 / 20 / 20 | 58 / 16 / 16 |
| 2 | 3 / 3 / 3 | 36 / 12 / 12 | 94 / 20 / 20 | 58 / 16 / 16 |
| 3 | 3 / 3 / 3 | 36 / 12 / 12 | 94 / 20 / 20 | 58 / 16 / 16 |

(train / validation / test; JNU regions = recording×block, pre-guard;
full detail incl. recordings and durations in `fold_statistics.json`.)

## 4. Class distributions (groups · recordings · usable duration)

**CWRU** (each partition = 1 specimen per class, 4 recordings per IR/B
specimen, 4 or 12 for OR):

| Fold | Train | Validation | Test |
|---|---|---|---|
| 1 | IR007+B014+OR021 · 20 rec · 173 s | IR014+B021+OR007 · 20 rec · 166 s | IR021+B007+OR014 · 12 rec · 107 s |
| 2 | IR014+B021+OR007 · 20 rec · 166 s | IR021+B007+OR014 · 12 rec · 107 s | IR007+B014+OR021 · 20 rec · 173 s |
| 3 | IR021+B007+OR014 · 12 rec · 107 s | IR007+B014+OR021 · 20 rec · 173 s | IR014+B021+OR007 · 20 rec · 166 s |

**JNU** (identical structure in every fold by rotation): every partition
contains all 4 classes × 3 speeds; per fold: train 108.1 s (3 blocks ×
12 recordings), validation 36.0 s, test 36.0 s — per class: n 54.1/18.0/
18.0 s, each fault class 18.0/6.0/6.0 s (pre-guard).

**HIT** (identical counts every fold; membership differs by seed):

| Split | Healthy (0) | InnerRace (1) | OuterRace (2) |
|---|---|---|---|
| train | 37 grp · 546 s | 40 grp · 590 s | 17 grp · 251 s |
| validation | 8 grp · 118 s | 8 grp · 118 s | 4 grp · 59 s |
| test | 8 grp · 118 s | 8 grp · 118 s | 4 grp · 59 s |

**MaFaulDa** (groups per class fixed by frozen allocation; recordings
vary with which configs the seed drew — fold 1 shown, folds 2/3 in
`fold_statistics.json`):

| Class | Train | Validation | Test |
|---|---|---|---|
| normal | 35 rec | 7 rec | 7 rec |
| imbalance | 5 cfg · 238 rec | 1 cfg · 48 | 1 cfg · 47 |
| horizontal-mis. | 2 cfg · 99 | 1 cfg · 49 | 1 cfg · 49 |
| vertical-mis. | 4 cfg · 201 | 1 cfg · 50 | 1 cfg · 50 |
| each of 6 bearing subfault×position classes | 2 cfg · 63–98 | 1 cfg · 25–49 | 1 cfg · 37–49 |

Durations: 5 s per recording throughout (train 5,460 s; val 2,060 s;
test 2,235 s in fold 1).

## 5. Operating-condition distributions

- **CWRU**: every specimen group spans all four loads (0–3 hp,
  1718–1797 rpm) → all partitions cover all loads in every fold.
  Severity is the rotation variable: each partition holds one severity
  per class (unseen-severity testing by design). OR positions @3/@6/@12
  travel with their specimen.
- **JNU**: all three speeds (600/800/1000 rpm) in every partition of
  every fold by construction.
- **HIT**: every session contributes to every partition in every fold
  (train 20/17/20/20/17; val 4/4/4/4/4; test 4/4/4/4/4 groups for
  data1–data5). LP-speed tertile coverage verified: train spans all 3
  tertiles, validation and test ≥ 2 (criterion H4; all folds pass).
- **MaFaulDa**: every configuration spans the full ~49-step speed sweep
  (737–3686 rpm), so any partition inherits near-full RPM coverage from
  its configs; Normal validation/test (7 recordings each) verified to
  span ≥ 2 RPM tertiles (criterion M4). Severity diversity: held-out
  configs are unseen severities for misalignment/imbalance/bearing-mass
  by construction.

## 6. Leakage audit

At the dataset grouping level, for every fold (asserted at build time and
re-verified by tests):

- **CWRU**: each of the 9 specimen groups maps to exactly one partition →
  Train∩Val = Train∩Test = Val∩Test = ∅ at specimen level; all loads and
  OR positions of a specimen share its partition.
- **JNU**: partitions are disjoint temporal regions of the same
  recordings; blocks tile `[0, N)` contiguously with zero overlap
  (end_i == start_{i+1}, tested); the four guard anchors sit exactly on
  internal boundaries, are marked `is_usable = False`, belong to no
  partition ("guard"), and will expand symmetrically at instantiation —
  ensuring ≥ G separation between any train and any val/test window.
- **HIT**: 134 groups, each in exactly one partition; the GitHub windowed
  release is not referenced by any manifest row.
- **MaFaulDa**: 41 fault configurations and 49 normal recordings each map
  to exactly one partition.
- Cross-dataset: the four rigs are physically disjoint (Part 1); no
  cross-dataset route exists at raw level.

## 7. CWRU exhaustive rotation proof

Role of each specimen across the three folds (T=train, V=validation,
X=test):

| Specimen | Fold 1 | Fold 2 | Fold 3 |
|---|---|---|---|
| IR007 | T | X | V |
| IR014 | V | T | X |
| IR021 | X | V | T |
| B007 | X | V | T |
| B014 | T | X | V |
| B021 | V | T | X |
| OR007 | V | T | X |
| OR014 | X | V | T |
| OR021 | T | X | V |

Every specimen appears exactly once per role; the three test sets are
disjoint and their union is all nine specimens. Verified by
`test_cwru_latin_rotation_matches_approved_table` against an
independently retyped copy of the approved table.

## 8. Dataset limitations

- **CWRU**: Healthy excluded — only one independent healthy specimen
  could be established, which cannot support the approved
  specimen-disjoint protocol. CWRU is a 3-class fault-type benchmark
  with unseen-severity test sets; specimens are seen across loads within
  their own partition only.
- **JNU**: evaluation is **within-recording temporal holdout** — the same
  physical acquisition contributes to train and test through disjoint,
  guard-separated time regions. It is NOT unseen-bearing,
  recording-independent, or unseen-domain generalisation.
- **HIT**: the original prepared GitHub train/test files were rejected
  because the Part-1 audit traced all 2,340 underlying series into both
  official partitions. Note additionally: all 25 outer-race groups stem
  from one physical bearing/assembly — HIT is a seen-bearing,
  cross-speed-group protocol.
- **MaFaulDa**: fault classes are configuration-grouped; **Normal uses
  recording-level grouping** (single configuration — a weaker unit;
  normal-vs-fault contrasts are seen-setup). Single rig throughout; the
  underhang/overhang subfault pairs share physical defective bearings;
  the website naming inconsistency (M2) is preserved, not repaired.

## 9. S0/S1 fairness declaration

**All later S0 and S1 comparisons must use identical fold assignments.**
Encoded in `split_protocol.json → usage_rules`: S0 optimizes on labelled
TRAIN, selects on VALIDATION, evaluates once on TEST; S1 SSL pretraining
may consume TRAIN signals only (labels withheld) and must not touch
validation or test signals even unlabelled; S1 fine-tuning receives
exactly the S0 labelled train set of the same fold. `fold_id` is a data
partition; `model_seed` is future training stochasticity — never
conflated. Test data must not influence any future design choice
(window length, STFT, architecture, masking, hyperparameters, stopping,
checkpoints, samplers, augmentation); design inspection uses TRAIN
(validation only where appropriate). Test stays sealed until final
evaluation.

## 10. Final protocol table

| Dataset | Classes | Grouping unit | Fold strategy | Main limitation |
|---|---|---|---|---|
| CWRU | IR / Ball / OR (48 kHz DE, 7/14/21 mil) | physical specimen across loads, OR positions merged | exhaustive Latin severity rotation | no Healthy; 1 specimen/class/partition; seen specimen across loads |
| JNU | n / ib / ob / tb × 3 speeds | temporal macro-block within recording | block rotation ABC/D/E → BCD/E/A → CDE/A/B | within-recording holdout only |
| HIT | Healthy / Inner / Outer | session × speed-group recording | seeded session-stratified 94/20/20 | outer class = 1 physical bearing; seen-bearing |
| MaFaulDa | 10 operational classes | fault configuration; Normal by recording | seeded class-stratified 58/16/16 units | ~57/21/22 not 70/15/15 (coverage-first); Normal seen-setup; single rig |

## Sealing

SHA-256 (frozen, also in `split_hashes.csv`):

| File | SHA-256 |
|---|---|
| global_fold_1.csv | 0903276ac911f455a8c7e00439d2d1a8161065648034056dff76fda3c7749cc1 |
| global_fold_2.csv | 3a5d4e2868702dc7a3b79255fa6185a9b2f0862d5662e63a6d8ed7454c1496b6 |
| global_fold_3.csv | df24fa346a671c098ef2ce31cd25e624caee67d01fff4ae7efe8f9f332032135 |
| test_identity_fold_1.csv | d7d7737d9066e198a088192d306a2e30914f7f1d30d2a0e0b6e2da2b53a86f8b |
| test_identity_fold_2.csv | 30f712de09b0ff0a09acf75a2b891cdfe6f76d723fa317b6f32c97ed59bdf8fd |
| test_identity_fold_3.csv | 5ab99b23d8b2b32492c4a58ec9f6f75dfc0152d633ff02ab6d88ec2b42225e61 |
| split_protocol.json | 284e95dacb137c16e0203da8dcf4088b1fb734bddafbbdb32be617016fa55c2a |
| **MASTER_PROTOCOL_HASH** | **527ccc1d449b223a37ecf109ed27be5279e17d85ba4e881abb9c68f4035e69c6** |

Future stages must call
`src.methodology_v2.part2_builder.verify_frozen_hashes()` before consuming
any fold manifest; it fails closed on any byte difference.

## Reproducibility

Command: `.venv/bin/python scripts/methodology_v2/run_part2_splits.py` at
git commit 9047ccbc (branch `bearing-generalisation-v1`; Part-2 code
untracked at generation time). Part-1 artifact hashes (recording
manifest, CWRU re-check report + table, Part-1 report), seeds used,
environment and timestamps: `split_reproducibility.json`. Rerun
determinism is enforced by
`test_rerun_reproduces_byte_identical_manifests`.

**HARD STOP.** Part 3 (resampling, channels, windows, STFT,
normalization, masking, architectures, training, samplers, few-shot,
evaluation) requires explicit approval.
