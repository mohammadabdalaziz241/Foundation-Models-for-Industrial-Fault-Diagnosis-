# Grouping policy — methodology_v2 Part 1

Purpose: for each candidate dataset, identify the **highest independent
experimental unit that genuinely exists in the released data** and propose
the provisional grouping unit that later train/validation/test partitions
must respect. No partitions are created here.

Conceptual hierarchy used throughout (highest → lowest):

1. physical bearing / specimen
2. experimental run / session (one assembly of the rig)
3. original recording (one continuous acquisition)
4. contiguous temporal macro-block within a recording
5. window — **never a grouping unit**

The new protocol deliberately does **not** require physical-bearing holdout;
the target is the highest unit that still supports a practical
classification benchmark with train/val/test class coverage.

---

## CWRU

### Available independence structure
- **Physical specimen**: one seeded-fault bearing per (fault type ×
  diameter), plus the healthy motor. Specimen identity is *inferable* from
  the fault specification (the same seeded bearing was re-tested at all
  four loads; canonical file numbers are consecutive across loads).
  Specimen serials are not published. Whether the three outer-race clock
  positions (@3/@6/@12) of the same diameter reused one physical bearing
  is **not documented**; positions are conservatively treated as distinct
  specimens here.
- **Recording**: one `.mat` file = one acquisition. 112 distinct
  recordings exist locally (60 at 12 kHz DE, 52 fault recordings at
  48 kHz DE, and the 4 normal-baseline recordings which are genuinely
  48 kHz; the copies of the normals inside `data/raw_cwru_48k/` are
  byte-identical duplicates and are counted once).
- **Critical nuance**: the 12 kHz and 48 kHz fault recordings of the same
  (specimen, load) are *separate acquisitions of the same physical
  experiment* (e.g. IR007 load0 exists as canonical file 105 at 12 kHz and
  109 at 48 kHz). Placing one in train and the other in test would put
  nearly identical experiments on both sides.

### Candidate group_id
`cwru_<spec>_load<L>` — fault specimen × motor load, **merging the 12 kHz
and 48 kHz twins** of the same experiment into one group.

### Evidence
- Internal MATLAB variable names carry canonical file numbers (X105 →
  105.mat), verified for all 116 local files; known official quirks
  reproduced exactly (98.mat missing RPM; 99.mat carrying leftover X098;
  175.mat carrying stray X217; 217.mat carrying leftover X215; short
  173.mat; 28-mil files DE-only with internal ids X048–X051/X056–X059).
- Sampling rates per the frozen audit `docs/cwru_legacy_rate_impact_note.md`
  (fault `_DE12k` = 12 kHz; 48k directory and all normals = 48 kHz).

### Number of groups
64 (15 fault specs × 4 loads = 60, plus Normal × 4 loads = 4).
At the stricter specimen level there would be 16 groups (15 fault + 1
healthy).

### Class coverage (groups per class, specimen×load unit)
Normal 4 · InnerRace 16 · Ball 16 · OuterRace 28. Every class can appear
in train, validation and test with many groups to spare.

### Advantages
- Recording-level independence with the near-duplicate 12k/48k twin
  leakage route closed.
- Large group count per class → flexible, statistically stable splits.
- Matches the intended "easier than unseen-bearing" protocol.

### Leakage risks
- Same physical specimen appears in different groups (different loads).
  A model may recognise a *specimen signature* rather than fault
  physics. This is an accepted, disclosed concession of the easier
  protocol — CWRU results must be described as *seen-bearing,
  cross-recording* classification, not bearing generalisation.
- If both rate families are used simultaneously in one task, sampling-rate
  itself becomes a class-correlated shortcut (28-mil exists only at 12 kHz;
  OR021@3 only at 48 kHz). Part 2 should select one rate family per task
  (or handle rate explicitly) — open decision.

### Limitations
- Specimen identity for OR positions unresolved (not documented).
- The healthy class has only 4 recordings from 1 physical condition.

### Recommended grouping unit
**specimen × load group (`cwru_<spec>_load<L>`), i.e. recording-level with
12k/48k twins merged.**

### Confidence
High (file-level provenance verified against canonical numbering; rates
frozen by prior audit).

---

## JNU

### Available independence structure
- 12 original recordings, one per (condition × speed):
  4 conditions (n, ib, ob, tb) × 3 speeds (600/800/1000 rpm).
- Fault recordings: 500,500 samples = 10.01 s at 50 kHz.
  Healthy recordings: 1,501,500 samples = 30.03 s (exactly 3× longer).
  A discontinuity probe at the 1/3 and 2/3 boundaries of all three healthy
  files found **no** concatenation evidence (boundary jumps 0.05–0.33× the
  local 99.9th-percentile first-difference), so they behave as continuous
  recordings; their 3× length remains undocumented.
- One physical seeded specimen per condition (inferred from the source
  description: one wire-cut dent per element), reused across speeds.
  Filename suffixes (`_2`, `_3_2`) are undocumented and recorded verbatim.

### Candidate group_id
`jnu_<cond>_<speed>` — the original recording (identical to recording_id;
each recording is its own group).

### Evidence
Direct enumeration of the cloned official repository
(commit 75b33611); durations computed from the files.

### Number of groups
12 (one per recording); 3 per class; 4 per speed.

### Class coverage
A whole-recording split **cannot** give per-class coverage in all three
partitions without reducing every (class, partition) cell to at most one
recording:
- Speed-disjoint assignment (train=600, val=800, test=1000) covers all 4
  classes everywhere but confounds partition with speed and rests on a
  single recording per (class, split) cell.
- Any latin-square assignment likewise leaves exactly one recording per
  (class, split): recording-idiosyncratic signatures (sensor mounting,
  ambient state) become class evidence — statistically fragile.

### Advantages of recording-level (if it were used)
True acquisition-level independence; no temporal leakage.

### Leakage risks
- With temporal macro-blocks (below), train and test blocks come from the
  *same* physical acquisition: same specimen, same mounting, same ambient
  conditions. This measures within-recording generalisation over time,
  **not** unseen-bearing and not even unseen-recording generalisation.
  This must be stated explicitly wherever JNU results are reported.

### Limitations
Only 12 recordings; the dataset cannot support anything stronger without
dropping classes or speeds.

### Recommended grouping unit
**Contiguous temporal macro-blocks with discarded guard intervals inside
each of the 12 recordings** (hierarchy level 4), applied consistently:
TRAIN block → guard → VALIDATION block → guard → TEST block, in fixed
temporal order to prevent future-to-past leakage. Guard concept (values to
be fixed in Part 2, not here): at least several shaft revolutions and
longer than any window/augmentation receptive field; at 600 rpm one
revolution is 100 ms, so guards of the order of seconds discard many
revolutions while costing little of the 10 s recordings.

### Confidence
High for the structure facts; the necessity of temporal blocking follows
arithmetically from 12 recordings / 4 classes / 3 speeds.

---

## HIT

### Available independence structure
- **Session / physical bearing** (highest): five test campaigns
  (data1–data5), each one assembly of the aero-engine with one bearing:
  2× healthy, 2× inner-ring (distinct specimens: 0.5×0.5 mm and
  0.5×1.0 mm faults), 1× outer-ring (paper Tables III & VI).
- **Recording**: within each session, one 15 s acquisition per planned
  (LP, HP) speed group at 25 kHz, released as contiguous 20480-sample
  series; the speed pair survives per series (row 7: LP at sample 0, HP at
  sample 1), verified against the paper's Table V speed plan.
  data1: 28 speed groups (504 series) · data2: 25 (450) · data3: 28 (504)
  · data4: 28 (504) · data5: 25 (450) → **134 recordings**, 2412 series.
- **Official GitHub split**: xtrain_1..4/xtest are 12,060 pre-windowed
  2048-sample rows of channel 1 with all metadata destroyed. The ytest
  class counts (954/1008/450) are *exactly* one fifth of the total pool per
  class — the split is stratified-random at window level. Windows from the
  same 20480-sample series (same 15 s recording) sit on both sides (see
  `integrity_details.json` → HIT_github_release.provenance for the direct
  window-to-series matching evidence). **The official split is therefore
  not leakage-safe at any level of the hierarchy and should not be
  inherited.** A validation set carved out of the official training shards
  would inherit the same defect.

### Candidate group_id
`hit_<session>_rec<k>` — one (session × speed-group) recording.

### Evidence
Empirical decoding of the full Drive release (shapes, label row, speed
row) cross-checked against paper Tables V/VI; series counts match the
published 504/450/504/504/450.

### Number of groups
134 recordings: healthy 53 (28+25), inner 56 (28+28), outer 25.

### Class coverage
Ample at recording level: every class has ≥25 groups.
At session level coverage fails: outer has a single session, so
session-disjoint train/test is impossible for that class.

### Advantages
- Uses the highest unit that still covers all classes.
- Speed-pair metadata allows stratifying or systematically holding out
  speed groups later.

### Leakage risks
- All 25 outer-race recordings share one physical bearing and one assembly;
  healthy and inner have only two each. Recording-level results therefore
  measure cross-speed-group generalisation *within* seen bearings —
  a seen-bearing protocol, to be disclosed.
- Series within one recording are temporally adjacent segments of one 15 s
  record; the recording must move as a unit (never split its series).
- data2 is stored as float32, the others float64 — a session-correlated
  encoding artefact; if healthy-vs-fault contrasts ever hinge on data2
  idiosyncrasies this deserves re-inspection (noted; low expected impact
  as data1 is float64 and also healthy).

### Limitations
Speed groups within a session were recorded in one campaign; rig state
drifts (temperature, lubrication) are shared across a session.

### Recommended grouping unit
**Recording = session × speed-group (134 groups).** Do **not** adopt the
official GitHub split (window-level random; evidence above). If Part 2
wants an extra safety margin, assign whole speed groups (LP/HP pairs) to
partitions so the same speed pair of the same session never straddles
train/test.

### Confidence
High (structure verified directly from the released arrays against the
paper).

---

## MAFAULDA

### Available independence structure
- **Single rig throughout** (SpectraQuest MFS ABVT); no independent
  machines. Three manufacturer-supplied defective bearings (ball, cage,
  outer race), each installed alternately at the underhang **and**
  overhang positions → underhang/X and overhang/X share one physical
  defective specimen. Misalignment/imbalance/normal use the healthy
  bearings of the same rig.
- **Fault configuration** (= one physical setup instance): 42 exist —
  normal (1), imbalance (7 masses), horizontal misalignment (4 offsets),
  vertical misalignment (6 offsets), underhang 3 subfaults × 4 added
  masses (12), overhang likewise (12).
- **Recording**: one 5 s, 8-channel, 50 kHz CSV per (configuration,
  rotation speed); 1951 total, ~37–51 speeds per configuration
  (verified counts match the website tables exactly, per configuration).
- Repeated recordings under *almost identical conditions* exist in the
  sense that neighbouring speed steps (~60 rpm apart) of the same
  configuration are physically near-identical experiments.

### Candidate group_id
`mafaulda_<configuration>` (e.g. `mafaulda_imbalance_6g`,
`mafaulda_underhang_ball_fault_20g`), i.e. hierarchy level 2
(setup instance), with the **normal class as the single exception**
(one configuration only — see below).

### Evidence
Directory census of the extracted official full.zip (1951 CSVs; per-config
counts equal the website's tables); channel/duration facts from the
website and verified per file in `integrity_details.json`.

### Number of groups
42 configurations; per class: normal 1 · imbalance 7 · horizontal 4 ·
vertical 6 · each bearing subfault × position 4 (mass levels), i.e. 12 per
position across 3 subfaults.

### Class coverage
- Config-level train/val/test coverage works for every fault class
  (≥4 configs each) but **fails for normal** (1 config → cannot appear on
  both sides of a config-disjoint split).
- Recording-level (sequence-level) grouping gives abundant coverage
  everywhere (≥137 sequences per bearing subfault×position, 49 for
  normal).

### Advantages
- Config-level grouping prevents the strongest sibling leakage:
  recordings of the same physical setup at neighbouring speeds landing on
  both sides of a split.

### Leakage risks
- Sequence-level splits within a configuration are near-duplicate
  experiments (only speed differs) — this is the main reason NOT to use
  plain recording-level grouping for fault classes.
- Underhang/overhang pairs share physical bearings; if both positions'
  data are used as separate classes or merged, the shared specimen is a
  disclosed non-independence.
- For the normal class any split necessarily reuses the single healthy
  configuration; normal-vs-fault contrasts on MaFaulDa are therefore
  seen-setup contrasts. Mitigation option for Part 2: split normal by
  recording (distinct speeds per partition) and disclose.
- Severity (mass/offset) correlates with configuration; holding out whole
  configs means testing on unseen severities — scientifically desirable
  but lowers ceiling accuracy; disclose.

### Limitations
One rig, one healthy configuration, no session/date metadata.

### Recommended grouping unit
**Fault configuration** for all fault classes (level 2);
**recording (sequence)** for the single-configuration normal class, with
explicit disclosure. Windows never cross recording boundaries in any case.

### Confidence
High for structure and counts; medium for the physical-bearing sharing
inference (documented in prose on the official site: three defective
bearings "placed one at a time in two different positions").

---

## Cross-dataset note

The four datasets come from four physically disjoint rigs; there is no
cross-dataset leakage route at the raw-data level. The only shared-machine
structure is *within* MaFaulDa (single rig) and *within* HIT (single
engine, five assemblies).
