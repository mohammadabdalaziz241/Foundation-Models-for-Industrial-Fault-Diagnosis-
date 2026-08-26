# PART 1 — Dataset Selection, Audit, and Common Task Definition
## methodology_v2 audit report

Scope: evidence-gathering and dataset audit only. No windows, no splits,
no resampling, no training artefacts were created. Companion deliverables
in this directory: `dataset_census.csv/.md`, `recording_manifest.csv`
(2,209 rows — one per original recording), `grouping_policy.md`,
`label_mapping_candidate.yaml`, `data_integrity_report.md`,
`raw_file_hashes.csv`, `integrity_details.json`, `reproducibility.json`.

---

## 1. Dataset inventory

| Dataset | Raw recordings | Classes (native) | Sampling rate | RPM / speeds | Candidate groups | Problems |
|---|---|---|---|---|---|---|
| CWRU | 112 (116 files; 4 byte-dup normals counted once) | 16 fine labels → Healthy/IR/OR/Ball, severities 7–28 mil, OR clock positions | 12 kHz (fault DE12k) + 48 kHz (48k DE + all normals) | 1730–1797 rpm (4 loads; measured RPM in-file for most) | 64 (specimen × load, 12k/48k twins merged) | mixed rates are class-correlated; normals genuinely 48 kHz (frozen prior audit); same specimen across loads |
| JNU | 12 | n / ib / ob / tb (4) | 50 kHz | 600 / 800 / 1000 rpm | 12 (= recordings) | only 1 recording per (class, speed); healthy files 3× longer, undocumented |
| HIT | 134 (5 sessions × 25–28 speed groups; 2,412 series) | healthy / inner / outer (3) | 25 kHz | LP 1000–5000 × HP 1200–6000, 25–28 pairs/session | 134 (session × speed group) | official GitHub split is window-level random (proven leaky); outer class = 1 physical bearing; data2 float32 |
| MaFaulDa | 1,951 | normal / imbalance / h-mis / v-mis / {under,over}hang × {ball, cage, outer} | 50 kHz | 737–3686 rpm (~49 speed steps per config) | 42 configurations | website fault-name permutation (semantic ambiguity for bearing subfaults); single rig; normal has 1 configuration |

Full field-by-field census: `dataset_census.md`.

## 2. Recommended dataset inclusion

| Dataset | Status | Reasoning |
|---|---|---|
| CWRU | **KEEP** | Verified authentic; large per-class group counts; canonical benchmark. Carry the 48 kHz-normal correction and pick one rate family per task in Part 2. |
| JNU | **KEEP (with disclosed limitation)** | Clean, verified, but only 12 recordings — usable only with temporal macro-blocks + guard gaps; results must be labelled within-recording generalisation. Excluding it would remove the only 4-class × 3-speed single-channel rig; keeping it is justified as long as the limitation is explicit. |
| HIT | **KEEP (full Drive release only)** | Full release preserves session + speed metadata → 134 clean recording groups. The official GitHub windowed split must NOT be used (proof in integrity report H2); this is an inclusion condition, not an exclusion reason. |
| MaFaulDa | **KEEP; bearing-subfault classes CONDITIONAL for the shared taxonomy** | Data are pristine and voluminous. The website's fault-name permutation (M2) leaves residual semantic ambiguity for ball/cage/outer subfault names; harmless under dataset-specific heads (Task B), CONDITIONAL under the shared taxonomy (Task A). Imbalance/misalignment classes are kept but never mapped into bearing classes. |
| Paderborn | **EXCLUDE (per protocol direction)** | Not part of the new primary benchmark at this stage; also carries the known DC-offset shortcut history in this project. No new audit performed. |

Nothing is permanently discarded: excluded/conditional material stays in the
manifest with native labels.

## 3. Recommended future split unit

| Dataset | Recommended group unit | Why | Main limitation |
|---|---|---|---|
| CWRU | specimen × load (`cwru_<spec>_load<L>`), 64 groups — recording level with 12 kHz/48 kHz twins merged | closes the near-duplicate twin route; large per-class coverage; matches "easier than unseen-bearing" intent | same physical specimen appears at different loads → seen-bearing protocol |
| JNU | temporal macro-blocks with guard gaps inside each of the 12 recordings (train→guard→val→guard→test, fixed temporal order) | recording-level split cannot cover classes without 1-recording-per-cell fragility | within-recording only; NOT unseen-bearing, NOT unseen-recording — must be disclosed |
| HIT | recording = session × speed group (134 groups) | highest unit with class coverage (session level fails: outer has 1 session); speed metadata verified | outer class recordings share one bearing/assembly |
| MaFaulDa | fault configuration (42 groups) for fault classes; recording (sequence) for the single-config normal class | prevents near-identical neighbouring-speed leakage; severity held out with the config | normal class is seen-setup; single rig overall |

Details and evidence: `grouping_policy.md`.

## 4. Shared-taxonomy feasibility (Task A)

Classes that genuinely overlap: **Healthy, InnerRace, OuterRace,
RollingElement** — but no dataset covers all four alongside the others:

| Shared class | CWRU | JNU | HIT | MaFaulDa |
|---|---|---|---|---|
| Healthy | 4 rec / 4 grp | 3 / 3 | 53 / 53 | 49 / 1 |
| InnerRace | 28 / 16 | 3 / 3 | 56 / 56 | — (no inner-race class under folder taxonomy) |
| OuterRace | 52 / 28 | 3 / 3 | 25 / 25 | 372 / 8 (CONDITIONAL, naming ambiguity M2) |
| RollingElement | 28 / 16 | 3 / 3 | — | 323 / 8 (CONDITIONAL) |

Excluded from Task A by rule: MaFaulDa imbalance (333 rec), horizontal +
vertical misalignment (498), cage faults (376, no cross-dataset match);
they remain available for SSL pretraining and Task B.

Verdict: Task A is **feasible but structurally incomplete** — the
label×dataset matrix has two holes (HIT×RollingElement,
MaFaulDa×InnerRace) plus two CONDITIONAL cells; any 4-class shared
classifier is dominated by CWRU+JNU for RollingElement and CWRU+JNU+HIT
for InnerRace. Dataset identity partially predicts label support —
a confound that must be acknowledged if Task A is chosen.

## 5. Dataset-specific-head feasibility (Task B)

Shared SSL encoder + per-dataset heads fits the actual class structures
strictly better:

- every dataset keeps its full native taxonomy (CWRU 4-way or fine-grained;
  JNU 4-way; HIT 3-way; MaFaulDa 6-way system-level or 10-way full),
  including the 1,207 MaFaulDa recordings that Task A cannot use as
  bearing classes;
- the MaFaulDa naming ambiguity (M2) becomes irrelevant — labels stay
  dataset-native;
- no forced physics-violating merges; the supervised-vs-SSL comparison the
  dissertation needs is per-dataset matched either way;
- cost: no single cross-dataset confusion matrix; cross-dataset transfer
  claims need per-dataset reporting.

**Assessment: Task B is the more scientifically appropriate primary
evaluation; Task A remains viable as a secondary harmonized-subset
analysis** (recommendation only — decision deferred to the researcher,
§7).

## 6. Leakage risks discovered (complete list)

1. **HIT official split**: window-level stratified-random; all 2,340
   contributing series feed both train and test (proof: integrity H2).
   Do not inherit; do not carve validation out of the official shards.
2. **CWRU 12k/48k twins**: same physical experiment recorded at two rates
   → must share a group (closed by the recommended group id).
3. **CWRU byte-duplicate normals** across the two local directories →
   counted once (closed in manifest).
4. **CWRU rate–class correlation**: 28-mil only at 12 kHz, OR021@3 only at
   48 kHz, normals only at 48 kHz → rate choice must be made explicitly in
   Part 2 or rate becomes a shortcut feature.
5. **CWRU specimen across loads** (accepted, disclosed: seen-bearing
   protocol).
6. **JNU temporal blocks**: same acquisition on both sides of any split —
   inherent to the dataset's size; guard gaps mitigate adjacency, not
   identity. Disclose in all JNU reporting.
7. **HIT series adjacency**: 18 series per 15 s recording are contiguous —
   a recording must never be split across partitions.
8. **HIT outer class**: single bearing/assembly (accepted, disclosed).
9. **MaFaulDa neighbouring-speed siblings**: sequences of one configuration
   differ only by ~60 rpm steps → configuration-level grouping required
   for fault classes.
10. **MaFaulDa shared defective bearings**: underhang/X and overhang/X use
    the same physical specimen (site prose) — disclosed non-independence.
11. **MaFaulDa single healthy configuration**: normal-vs-fault on MaFaulDa
    is seen-setup by construction; split normal by recording and disclose.
12. **Dataset-identity shortcuts in any pooled task**: rigs, rates and
    channels differ per dataset; in Task A the dataset is partially
    label-informative (§4). Any pooled classifier must be evaluated with
    this confound in mind.

## 7. Open decisions requiring approval before Part 2

1. **Task definition**: adopt Task B (dataset-specific heads) as primary
   with Task A as secondary harmonized analysis — or another arrangement?
2. **CWRU rate family**: use 12 kHz DE fault set (+ 48 kHz normals
   downsample-decision deferred), 48 kHz throughout, or handle both
   explicitly? (Interacts with leakage risk 4 and with any future common
   preprocessing; no resampling was performed in Part 1.)
3. **CWRU 28-mil severities** (12 kHz-only, DE-channel-only): include or
   drop from the benchmark?
4. **JNU acceptance**: confirm inclusion under the disclosed
   within-recording limitation, and approve the macro-block + guard-gap
   concept (values to be fixed in Part 2).
5. **HIT official split**: formally reject it in favour of
   session × speed-group recording splits (evidence H2), yes/no?
6. **MaFaulDa bearing subfaults in Task A**: accept the CONDITIONAL
   OuterRace/RollingElement mappings despite naming ambiguity M2, restrict
   MaFaulDa to Task B, or attempt independent adjudication (e.g. envelope
   spectrum vs BPFO/BSF characteristic frequencies — this would be a
   Part 2 diagnostic, not performed in Part 1)?
7. **MaFaulDa non-bearing classes** (imbalance/misalignment): include in
   SSL pretraining pool and Task B heads only — confirm?
8. **Channel policy per dataset** (CWRU DE vs DE+FE; HIT 6 channels;
   MaFaulDa 8 channels; JNU 1): defer to Part 2, but flag that channel
   count is dataset-correlated (shortcut risk in pooled settings).
9. **Dataset-balanced SSL sampling** (§8): approve the principle of
   ~equal per-dataset sampling probabilities (implementation in a later
   part).

## 8. Balance analysis (recordings/groups, no windows)

| Dataset | Recordings | Groups | Total duration | Share of duration |
|---|---|---|---|---|
| CWRU | 112 | 64 | 1,050 s (17.5 min) | 8.1 % |
| JNU | 12 | 12 | 180 s (3.0 min) | 1.4 % |
| HIT | 134 | 134 | 1,976 s (32.9 min) | 15.2 % |
| MaFaulDa | 1,951 | 42 | 9,755 s (162.6 min) | 75.3 % |

MaFaulDa holds ~75 % of all signal time (54× JNU); HIT contributes the
most independent recordings among the bearing-only rigs. Per-class
recording counts are in §4; native-label counts in `dataset_census.md`.

**Conclusion: later SSL pretraining will need dataset-balanced sampling**
(approximately equal per-dataset batch probability, as anticipated:
P(CWRU) ≈ P(JNU) ≈ P(HIT) ≈ P(MaFaulDa)), otherwise MaFaulDa dominates
the objective. Not implemented in Part 1. Note the compute-budget rule
already frozen for this project: fix the SSL step budget to the reference
condition; do not let epoch length scale with pool size.

## 9. Reproducibility

- Audit command: `.venv/bin/python scripts/methodology_v2/run_part1_audit.py
  --datasets CWRU JNU HIT MAFAULDA --mafaulda-workers 4`
- Code: `src/methodology_v2/` + `scripts/methodology_v2/run_part1_audit.py`
  (new, isolated; imports no legacy pipeline or training module — enforced
  by runtime assertion and tests).
- Tests: `tests/methodology_v2/test_part1_audit.py` (21 passed).
- Git commit at audit time, package versions, timestamps, dataset source
  URLs, clone commits (JNU 75b33611, HIT ef176559) and the MaFaulDa
  archive sha256: `reproducibility.json`.
- Raw-data immutability: 2,094 sha256 hashes in `raw_file_hashes.csv`;
  spot-check re-hash runs inside the test suite.
- No previous experimental manifest, result, or metadata file was
  modified; all Part 1 outputs live under `methodology_v2/part1_audit/`
  and new raw data under `data/` (gitignored), directories
  `data/raw_jnu/`, `data/raw_hit/`, `data/raw_mafaulda/`.

## HARD STOP

Part 1 ends here. Train/val/test manifests, temporal blocks, guard-gap
values, resampling, filtering, normalization, window length, STFT,
masking, architectures, supervised/SSL training and few-shot experiments
are all deferred pending approval of §7.
