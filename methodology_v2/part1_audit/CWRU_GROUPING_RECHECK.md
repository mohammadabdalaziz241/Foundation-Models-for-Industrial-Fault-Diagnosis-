# CWRU grouping re-check — bounded Part-1 addendum

Scope: the retained 48 kHz drive-end benchmark subset only (Normal +
InnerRace/Ball/OuterRace at 0.007"/0.014"/0.021"; 0.028" and the 12 kHz
fault family excluded). Audit-only: no splits, no windows, no
preprocessing, no training; raw data untouched. This document supersedes
the CWRU section of `grouping_policy.md` for the new benchmark; the Part-1
manifest is unchanged (its `group_id_candidate` was defined for the
two-rate-family manifest and is explicitly a candidate).

Machine-readable companions (regenerable via
`scripts/methodology_v2/run_cwru_grouping_recheck.py`):
`cwru_grouping_recheck_table.csv` (56 rows — one per retained recording)
and `cwru_grouping_recheck_stats.json`. Tests:
`tests/methodology_v2/test_cwru_grouping_recheck.py` (5 tests; suite total
26 passed).

---

## 1. Executive conclusion

**The future CWRU group_id should be the physical fault specimen across
ALL motor loads, with outer-race clock positions merged into their
specimen** (`cwru48k_IR007`, `cwru48k_B014`, `cwru48k_OR021`, …,
`cwru48k_Normal`) — Option C2 below. The Part-1 `specimen × load`
proposal is **rejected** for the new 48 kHz benchmark: within a single
rate family it is identical to a plain recording split (56 = 56 groups)
and allows the same physical seeded bearing to appear in train and test
under different loads — precisely the condition-wise leakage documented in
the recent literature.

Consequence that must not be hidden: the Healthy class has **one**
physical specimen, so a strictly bearing-wise four-class train/val/test
benchmark is **impossible** on CWRU. A disclosed Healthy concession (§6)
requires explicit approval.

## 2. Evidence about physical specimen identity

**Documented (official Bearing Data Center welcome page, quoted
verbatim):**

> "Motor bearings were seeded with faults using electro-discharge
> machining (EDM). Faults ranging from 0.007 inches in diameter to 0.040
> inches in diameter were introduced separately at the inner raceway,
> rolling element (i.e. ball) and outer raceway. Faulted bearings were
> reinstalled into the test motor and vibration data was recorded for
> motor loads of 0 to 3 horsepower (motor speeds of 1797 to 1720 RPM)."

This describes **one seeded bearing per (fault location × diameter)**,
reinstalled and measured over the 0–3 hp load sweep. It does not literally
say "the same bearing at every load", so structural evidence was checked:

- **Consecutive official numbering across loads within every fault spec**
  (109–112, 122–125, 135–138, …, 262–265; verified for all 52 retained
  fault recordings against the official 48k DE table fetched 2026-08-11,
  and against the hash-frozen per-file source URLs in
  `metadata/vibrationclip_v1/cwru_48k_enumeration.json`). The four loads
  of a spec form one contiguous acquisition block — consistent with a
  single installed specimen swept through loads, not four separately
  seeded bearings.
- **In-file measured RPM decreases monotonically with load within every
  spec** (≈1797 → ≈1720), matching a single load-sweep campaign.
- **No documentation anywhere on the site describes manufacturing more
  than one specimen per fault spec.**

Answer to the key question: **yes — the four load recordings of a fault
spec are repeated measurements of the same installed fault specimen**
(confidence: high; grade "documented + structural", not a verbatim
single-sentence proof).

**Literature rationale.** Hendriks, Dumond & Knox (MSSP 169, 2022,
108732) demonstrate on CWRU that condition-based train/test construction
leaves "the same physical bearings … in both training and testing sets"
and that CNNs then learn bearing-specific features that do not generalise
to other bearings. Vieira, Bauler, Rosa & Silva (arXiv:2509.22267)
formalise the split hierarchy (segment-wise / condition-wise /
bearing-wise) and evaluate CWRU under bearing-wise partitioning, the only
level with "no overlap between the physical components used for training
and testing". This project's own Paderborn work
(`docs/dissertation/split_protocol_argument.md`) reached the same
conclusion independently (near-perfect bearing-identity decoding from
windows). Option B is a condition-wise split in this taxonomy.

## 3. Recording/specimen table

Full 56-row table: `cwru_grouping_recheck_table.csv`, columns include
internal variable id, official download number, frozen sha256 prefix,
class, diameter, OR position, load, in-file RPM, duration and the group
ids under every option. Compressed view (official numbers per load
0/1/2/3):

| Specimen (C2) | Class | Diameter | Recordings | Official numbers | Notes |
|---|---|---|---|---|---|
| cwru48k_Normal | Healthy | — | 4 | 97/98/99/100 | genuinely 48 kHz; 98 lacks RPM var; 99 carries leftover X098 |
| cwru48k_IR007 | InnerRace | 7 mil | 4 | 109/110/111/112 | |
| cwru48k_IR014 | InnerRace | 14 mil | 4 | 174/175/176/177 | 174.mat internally X173 (numbering quirk); 174 is short (1.33 s); 175 carries stray X217 |
| cwru48k_IR021 | InnerRace | 21 mil | 4 | 213/214/215/217 | 216 skipped officially; 217 carries leftover X215 |
| cwru48k_B007 | Ball | 7 mil | 4 | 122/123/124/125 | |
| cwru48k_B014 | Ball | 14 mil | 4 | 189/190/191/192 | |
| cwru48k_B021 | Ball | 21 mil | 4 | 226/227/228/229 | |
| cwru48k_OR007 | OuterRace | 7 mil | 12 | @6:135–138 · @3:148–151 · @12:161–164 | 3 clock positions (§5) |
| cwru48k_OR014 | OuterRace | 14 mil | 4 | @6:201–204 | @3/@12 not collected at 48 kHz |
| cwru48k_OR021 | OuterRace | 21 mil | 12 | @6:238–241 · @3:250–253 · @12:262–265 | 3 clock positions (§5) |

All 14 load-0 recordings are systematically short (1.33–5.1 s vs ~10.1 s
at loads 1–3) — an acquisition-length pattern, not corruption.

## 4. Comparison of grouping options

Computed over the 56 retained recordings
(`cwru_grouping_recheck_stats.json`):

| | **A — recording** | **B — specimen × load** | **C1 — specimen, OR positions separate** | **C2 — specimen, OR positions merged** |
|---|---|---|---|---|
| Groups | 56 | **56 (identical to A within one rate family)** | 14 | 10 |
| Groups per class (H/IR/B/OR) | 4/12/12/28 | 4/12/12/28 | 1/3/3/7 | 1/3/3/3 |
| Groups per severity (7/14/21 mil) | 20/12/20 | 20/12/20 | 5/3/5 | 3/3/3 |
| Groups touching each load | 14 per load | 14 per load | all (each group spans 4 loads) | all |
| Same physical specimen can cross partitions | **yes** | **yes** | **yes** (via OR positions, §5) | **no** |
| Class coverage in 3-way split | trivial | trivial | yes | yes for faults; **impossible for Healthy** |
| Leakage risk | window leakage prevented; specimen leakage open | same as A — condition-wise split (Hendriks et al.) | closed except possible OR-position specimen sharing | closed (bearing-wise) |
| 70/15/15 difficulty | easy (but meaningless) | easy (but meaningless) | possible only by leaving OR-heavy groups; still 1-specimen classes elsewhere | **impossible** — finest per-class granularity is 33 % (§7) |
| Limitations | measures within-specimen generalisation | as A, plus a false sense of grouping | asymmetric OR structure; residual OR identity risk | few, large groups; test severity necessarily unseen |

Key structural fact: with the 12 kHz family excluded, **Option B collapses
into Option A** — every (specimen × load) cell contains exactly one
recording. The protection Part 1 intended with `specimen × load` (merging
12k/48k twins of the same experiment) has no object inside a single rate
family; what remains is condition-wise leakage.

## 5. Outer-race identity analysis

For OR007 and OR021, recordings exist at 3, 6 and 12 o'clock; OR014 only
at 6 o'clock (48 kHz family).

- The official apparatus text says experiments were conducted "with outer
  raceway faults located at 3 o'clock …, at 6 o'clock …, and at 12
  o'clock", and the welcome page describes faults "introduced separately
  at the inner raceway, rolling element (i.e. ball) and outer raceway" —
  i.e. the documentation enumerates fault *specs*, not per-position
  specimens. **No document states whether one outer-race bearing per
  diameter was re-oriented or three bearings were manufactured.**
- Physical parsimony: an outer-race defect is fixed in the outer ring;
  changing its clock position is achieved by rotating the ring at
  installation ("faulted bearings were reinstalled"). Machining three
  identical-diameter EDM defects into three bearings solely to vary
  orientation would be an unusual and undocumented effort.
- Empirical discrimination (same-specimen fingerprint across positions)
  is not attempted here: position changes the transfer path to the
  sensor, so signal similarity would be inconclusive in both directions.

**Verdict: physical identity across OR positions cannot be established
with documentary certainty** — stated explicitly per the task. The
per-diameter positions are *probably* one specimen re-oriented
(moderate-to-high plausibility), and grouping must be chosen so that
being wrong is harmless:

- treating positions as **one specimen (C2)** is safe under both
  hypotheses (if they are distinct bearings, we have merely been
  conservative and lose two OR groups);
- treating them as **separate specimens (C1)** is unsafe if they share a
  bearing (possible specimen leakage across partitions).

Hence C2, with the uncertainty recorded. This is a merge chosen for
leakage-dominance, not a claim that identity is proven; position is kept
as metadata (`or_position_oclock`) for stratified reporting.

## 6. Healthy-class limitation

- The official baseline is a single series: files 97–100, one per load,
  collected with "normal bearings" installed; no second healthy bearing
  or repeated healthy campaign is documented anywhere. The four
  recordings are measurements of the **same installed healthy bearing
  set** at four loads (consecutive numbering 97–100; single baseline
  page). Nothing supports inventing more than **one** healthy specimen.
- Under strict specimen grouping the Healthy group can sit in only one
  partition → **a strictly bearing-wise four-class CWRU
  train/validation/test benchmark is impossible.** This is reported
  plainly, as required, rather than silently weakening the fault-class
  grouping.
- Options for Part 2 (require approval; none is implemented here):
  1. **Healthy recording-level exception** — split the four load
     recordings across partitions (e.g. 2 train / 1 val / 1 test),
     explicitly disclosed as seen-bearing for the Healthy class only;
     mirrors the Part-1 MaFaulDa normal-class exception. Cost: healthy
     train/test share the physical bearing; healthy-vs-fault separation on
     CWRU keeps a seen-bearing asymmetry (and fault recordings measure
     the same motor's DE position, so "healthy" is partly a rig baseline).
  2. **Temporal macro-block exception** within each normal recording
     (train/guard/val/guard/test in time), keeping all four loads on all
     sides; same disclosure, JNU-style.
  3. **Drop Healthy on CWRU** → three-class fault-type benchmark
     (IR/B/OR), fully bearing-wise; changes the task and breaks four-class
     comparability with JNU (HIT has no Ball; MaFaulDa has no InnerRace
     anyway).
- Recommendation: option 1 (recording-level exception), consistent with
  how the single-configuration normal class is handled for MaFaulDa, with
  the limitation stated in every CWRU results table.

## 7. Feasibility of three repeated ~70/15/15 splits

Under the recommended C2 grouping (9 fault specimen groups + 1 healthy
specimen):

- **Class coverage:** every partition can contain all four classes only
  via the Healthy exception (§6). For the fault classes, any
  specimen-disjoint three-way split with full coverage forces **exactly
  one specimen per class per partition** (3 specimens per class).
- **70/15/15 is not attainable at specimen level.** The finest per-class
  granularity is 33 %. By recordings the forced allocation gives each
  partition 4 recordings per fault class (except OuterRace: the partition
  holding OR014 gets 4 recordings ≈35 s, the others 12 ≈101 s). A
  defensible alternative shape, if a larger training share is wanted:
  keep **test** strictly bearing-wise (1 specimen per class) and allow the
  **validation** set to share specimens with train (condition-wise val,
  e.g. one held-out load per training specimen), disclosed as such —
  model selection then sees no unseen-bearing signal, but the test
  estimate stays clean. Choosing between "1/1/1 specimens (≈33/33/33)"
  and "2-specimen train + condition-wise val + 1-specimen test
  (≈50/17/33 by recordings)" is a Part-2 decision.
- **Severity coverage:** with severity = specimen, train severities are
  necessarily ≠ test severities (unseen-severity testing). Coverage
  *within* each partition spans all loads and, for OR007/OR021, all clock
  positions.
- **Load coverage:** every C2 group spans all four loads → all partitions
  see all loads by construction. (Load-0 recordings are short, §3.)
- **Three repeated splits:** exactly **three** severity rotations exist
  (test specimen = 007s / 014s / 021s, jointly rotated). They are
  *exhaustive*, not random draws: test sets are fully disjoint across
  repeats (each specimen is tested exactly once), which is genuine,
  meaningful variation — but repeat identity is confounded with test
  severity, and only three distinct repeats are possible at all. More
  than three repeats would require reusing test specimens (per-class
  independent rotations give 3×3×3 assignments but with overlapping test
  membership). Verdict: the planned "3 repeated splits" fits CWRU's
  structure exactly — as the three severity rotations, reported per
  rotation, not as i.i.d. resamples.

Option A/B would make 70/15/15 and arbitrary repeats trivially easy —
and scientifically empty, since specimen identity would sit on both sides
(Task 6 rule applied: structure over convenience).

## 8. Recommended CWRU grouping rule

```
group_id := cwru48k_<SPEC>
  SPEC = IR007|IR014|IR021|B007|B014|B021|OR007|OR014|OR021|Normal
  (fault specimen across ALL loads; OR clock positions merged;
   or_position and load retained as metadata only)
```

with: (i) the Healthy exception of §6 (subject to approval), (ii) test
partitions always specimen-disjoint from train, (iii) any validation
concession explicitly labelled condition-wise if adopted, (iv) every CWRU
results table stating the grouping unit ("bearing-wise, 48 kHz DE
family").

## 9. Confidence level and unresolved uncertainties

- Same specimen across loads (fault specs): **high confidence**
  (documented procedure + consecutive numbering + monotone RPM); not a
  verbatim single-sentence statement.
- Healthy = single specimen: **high confidence** (single documented
  baseline series; absence of any second-healthy documentation).
- OR positions share one specimen per diameter: **plausible, unproven**
  (§5) — C2 is chosen to be correct-or-conservative under both readings.
- Official numbering quirks (174↔X173, 216 skipped, leftover variables):
  verified directly; no residual uncertainty.
- Whether EDM specimen "signatures" are actually learnable at 48 kHz on
  this rig is not quantified here (the Paderborn precedent in this repo
  and Hendriks et al. on CWRU both showed bearing-identity learning is
  real); the grouping removes the route regardless.

## 10. Exact implications for Part 2

1. Replace the Part-1 CWRU `group_id_candidate` (specimen × load) with
   the C2 specimen groups when building split manifests; the mapping is
   already materialised per recording in
   `cwru_grouping_recheck_table.csv` (no Part-1 file was edited).
2. CWRU cannot honour 70/15/15 at group level; adopt either the 1/1/1
   specimen rotation (≈33/33/33) or 2-train-specimen + condition-wise
   val + 1-test-specimen (≈50/17/33 by recordings). Approve one.
3. The three repeated splits for CWRU = the three severity rotations;
   report per-rotation results (test severity differs by design). S0 and
   S1 must consume identical frozen rotations.
4. Approve the Healthy exception (recording-level split of the single
   healthy specimen) or an alternative from §6.
5. OR positions: approve the conservative merge (C2). If C1 is preferred
   despite §5, that is a documented risk acceptance.
6. Benchmark is single-rate (48 kHz) including the genuinely-48 kHz
   normals — the Part-1 rate-shortcut risk (risk 4) is closed by this
   subset choice; do not reintroduce 12 kHz material into this task.
7. Windowing budgets must respect the short load-0 recordings (1.33 s
   minimum: official 174) — a Part-2 sizing constraint, not acted on
   here.
