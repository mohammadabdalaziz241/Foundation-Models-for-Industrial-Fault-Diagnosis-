# PART 3A — Signal-Input, Channel, Sampling-Rate and Window-Duration Design Study
## methodology_v2

Audit/design study only: no resampling, no filtering, no windows, no
STFT, no normalization, no training. The Part-2 seal (master hash
`527ccc1d449b223a37ecf109ed27be5279e17d85ba4e881abb9c68f4035e69c6`) was
verified before and after every run; Part-2 artefacts are byte-unchanged.
Raw **TEST** signal content was never read: the only signal-content
diagnostics are explicitly **GLOBAL-FOLD-1 TRAIN-ONLY** (guard-enforced,
fail-closed; JNU sealed at temporal-region level via bounded reads).
Machine-readable artefacts: `channel_census.csv`,
`sampling_rate_study.csv`, `window_duration_study.csv`,
`window_count_estimates.csv`, `jnu_guard_study.csv`,
`cwru_load0_study.csv`, `fold1_train_diagnostics.json`,
`channel_policy_candidates.yaml`, `part3a_recommendations.yaml`,
`part3a_reproducibility.json`. Tests: 55 passed
(`tests/methodology_v2/`, incl. 13 new Part-3A tests).

## 1. Executive summary

- **Channel**: one fixed radial/vertical accelerometer per dataset
  (CWRU DE · JNU only channel · HIT ch3 · MaFaulDa underhang radial) is
  the closest physically comparable input; adaptive channel selection is
  rejected as label leakage.
- **Sampling rate**: 25 kHz is the strongest *common-rate* candidate —
  the only one that upsamples nothing and keeps HIT bit-identical — but
  fold-1-train diagnostics show JNU (and partly MaFaulDa) carry most raw
  energy above 12.5 kHz, a truncation that is structural to every common
  rate and must be consciously accepted (or native-rate processing
  chosen instead).
- **Window duration**: 1.0 s is recommended — ≥10 shaft revolutions at
  every documented operating point of every dataset (binding case: HIT
  relative inter-shaft speed, 3.3 revolutions at its slowest group);
  2.0 s is infeasible (kills JNU internal blocks and one CWRU
  recording); 0.25 s leaves <1 relative revolution for slow HIT groups.
- **Overlap**: train 50 % / validation 0 % / test 0 % (Policy B).
- All choices remain **PENDING HUMAN APPROVAL** (§15).

## 2. Channel census

Full census: `CHANNEL_CENSUS.md` + `channel_census.csv`. Summary: CWRU
has DE/FE housing accelerometers (radial, 12 o'clock; DE sits on the
faulted bearing); JNU one vertical accelerometer; HIT two LP-rotor
displacement channels (excluded — different quantity) and four
normal-to-casing accelerometers (ch3–ch6, exact stations undocumented);
MaFaulDa a tachometer, two 3-axis accelerometer sets (underhang IMI
601A01, overhang IMI 604B31) and a microphone. All channels within each
dataset are synchronous.

## 3. Channel-policy candidates

`channel_policy_candidates.yaml`. Primary: CWRU `DE_time`, JNU
`acc_vertical`, HIT `ch3`, MaFaulDa `col3_underhang_radial`.
Alternatives: CWRU `FE_time`, HIT `ch4–ch6`, MaFaulDa
`col6_overhang_radial`. Future multi-channel ablations documented but
not implemented. Rejected: fault-position-adaptive MaFaulDa channel
(encodes the label), and any displacement/microphone/tachometer main
input (non-comparable quantity).

## 4. Common sampling-rate comparison

| Rate | CWRU (48 k) | JNU (50 k) | HIT (25 k) | MaFaulDa (50 k) | Nyquist | Main advantages | Main risks |
|---|---|---|---|---|---|---|---|
| native | none | none | none | none | 24/25/12.5/25 kHz | full fidelity everywhere | identical STFT params ≠ identical physical bins; shared encoder sees inconsistent frequency axes |
| 20 kHz | ↓ 5/12 | ↓ 2/5 | ↓ 4/5 | ↓ 2/5 | 10 kHz | small tensors | needlessly discards native HIT content; worst truncation |
| 24 kHz | ↓ 1/2 | ↓ 12/25 | ↓ 24/25 | ↓ 12/25 | 12 kHz | elegant CWRU 2:1; repo precedent (vibrationclip_v1) | HIT 24/25 conversion buys nothing physical; truncation as 25 kHz |
| **25 kHz** | ↓ 25/48 | ↓ 1/2 | **none (native)** | ↓ 1/2 | 12.5 kHz | no upsampling anywhere; HIT bit-preserved; two exact half-rate decimations; one shared physical frequency axis | CWRU ratio 25/48 non-trivial (standard polyphase); JNU/MaFaulDa HF truncation (below) |
| 32 kHz | ↓ 2/3 | ↓ 16/25 | **UPSAMPLE 32/25** | ↓ 16/25 | 16 kHz | wider band for 50 k sets | fabricates HIT samples (no new information) — disqualifying for a common rate |

Downsampling requires proper anti-alias low-pass at the new Nyquist
(design belongs to Part 3B). Upsampling creates no information and is
rejected as a standardization tool.

**Fold-1 TRAIN energy diagnostic** (corroborative, never a cross-fold
parameter source; DC removed; candidate primary channels;
`fold1_train_diagnostics.json`):

| Dataset (native) | ≤5 kHz | ≤10 kHz | ≤12 kHz | ≤12.5 kHz | ≤16 kHz |
|---|---|---|---|---|---|
| CWRU DE (48 k), mean/min | .994/.983 | 1.000/1.000 | 1.000 | 1.000 | 1.000 |
| JNU (50 k), mean/min | .211/.047 | .288/.122 | .311/.145 | **.324/.152** | .521/.358 |
| HIT ch3 (25 k), mean/min | .879/.681 | .990/.965 | .999/.996 | 1.000 (native) | 1.000 |
| MaFaulDa col3 (50 k), mean/min | .645/.087 | .667/.091 | .670/.093 | **.670/.094** | .671/.095 |

Reading: CWRU and HIT lose essentially nothing at 12.5 kHz Nyquist. JNU
holds only ~32 % (min 15 %) of raw energy below 12.5 kHz — its
impact-excited resonance carriers lie mostly in 12.5–25 kHz — and even a
16 kHz Nyquist recovers only ~52 %. Parts of MaFaulDa behave similarly
(min 9 %). **No candidate common rate preserves JNU fidelity; only
native-rate processing does.** The truncation is identical for all
classes and splits within a dataset (representation-quality cost, not a
leakage or fairness issue), and raw energy is not the same thing as
diagnostic information — low-frequency defect harmonics remain — but the
cost must be accepted explicitly, not silently.

## 5. Window-duration comparison

Rotations per window from documented RPM (`window_duration_study.csv`):

| Duration | Min rotations (where) | Median (across datasets) | Max (where) | Dataset compatibility | Estimated relative cost |
|---|---|---|---|---|---|
| 0.25 s | **0.83 (HIT relative 200 rpm)**; 2.5 JNU@600 | ~3.5–15 | 20.8 (HIT LP 5000) | all fit | 1× frames, most windows |
| 0.50 s | 1.67 (HIT rel); 5.0 JNU@600 | ~7–30 | 41.7 | all fit | 2× |
| **1.00 s** | **3.33 (HIT rel)**; 10.0 JNU@600; 12.1 MaF@725; 28.6 CWRU | ~13–60 | 83.3 | all fit (JNU thin but complete) | 4× |
| 2.00 s | 6.67 (HIT rel) | ~27–120 | 166.7 | **JNU internal blocks: 0 windows; CWRU official 174 eliminated** | 8× |

Per-dataset shaft-RPM detail (min/median/max rotations):

| Dataset | RPM range | 0.25 s | 0.5 s | 1.0 s | 2.0 s |
|---|---|---|---|---|---|
| CWRU | 1718–1797 | 7.2/7.3/7.5 | 14.3/14.7/15.0 | 28.6/29.4/29.9 | 57.3/58.7/59.9 |
| JNU | 600/800/1000 | 2.5/3.3/4.2 | 5.0/6.7/8.3 | 10.0/13.3/16.7 | 20.0/26.7/33.3 |
| HIT (LP shaft) | 1000–5000 | 4.2/15.2/20.8 | 8.3/30.4/41.7 | 16.7/60.8/83.3 | 33.3/121.7/166.7 |
| HIT (relative HP−LP, the fault-driving speed) | 200–2400 | 0.83/3.5/10.0 | 1.7/7.0/20.0 | 3.3/14.0/40.0 | 6.7/28.0/80.0 |
| MaFaulDa | 725–3736 | 3.0/9.0/15.6 | 6.0/18.0/31.1 | 12.1/36.0/62.3 | 24.2/72.1/124.5 |

Trade-off: longer windows add repetitive fault context (TA requirement)
but reduce independent examples (counts §6 scale ~1/W), enlarge future
STFT time axes (memory/GPU ∝ W at fixed hop), and JNU's 2.002 s
macro-blocks cap the usable duration at ~1.0 s once guards (G = W) are
carved. Recordings shorter than the window: none at ≤1.0 s anywhere;
at 2.0 s: 1 CWRU recording + 27 of 36 JNU fault blocks per fold.

## 6. Future window-count estimates

Counts only — no windows were materialised. Full table (3 folds × 4
datasets × 3 splits × 4 durations × {0 %, 50 %} × counting bases):
`window_count_estimates.csv`. At the recommended 1.0 s, per fold
(train@50 % / val@0 % / test@0 %):

| Fold | CWRU | JNU | HIT | MaFaulDa | Total |
|---|---|---|---|---|---|
| 1 | 320 / 162 / 105 | 120 / 24 / 24 | 2632 / 280 / 280 | 9828 / 2060 / 2235 | 12 900 / 2526 / 2644 |
| 2 | 307 / 105 / 169 | 108 / 24 / 24 | 2632 / 280 / 280 | 10 026 / 2205 / 1980 | 13 073 / 2614 / 2453 |
| 3 | 198 / 169 / 162 | 120 / 24 / 24 | 2632 / 280 / 280 | 9900 / 2225 / 2030 | 12 850 / 2698 / 2496 |

Train at 0 % overlap instead of 50 % (fold-mean, 1.0 s): CWRU 275→145,
JNU 116→72, HIT 2632→1316, MaFaulDa 9918→5510 — overlap contributes a
≈1.8–1.9× multiplier. Duration scaling (fold-mean totals, train@50 %):
0.25 s ≈ 55 k, 0.5 s ≈ 27 k, 1.0 s ≈ 13 k, 2.0 s ≈ 6 k windows.

HIT counting basis: primary counts treat each 14.7456 s recording as
contiguous — justified by the uniform 18-series structure and the
fold-1-train continuity diagnostic (1,598 boundaries; boundary jumps
median 0.20×, max 1.19× the within-series 99.9th-percentile first
difference — joints look like ordinary neighbouring samples). Under the
conservative series-constrained alternative (windows confined to single
0.8192 s series), 1.0 s and 2.0 s windows are impossible for HIT (0
windows) and 0.5 s yields 1 window/series. **Adopting the contiguous
basis is therefore load-bearing for any window ≥1.0 s and needs explicit
approval** (§15).

## 7. CWRU short-recording analysis

`cwru_load0_study.csv`. The retained 3-class benchmark holds 13 load-0
recordings (Part 1's count of 14 included the now-excluded Healthy
baseline). Shortest: official 174 (IR014, 0 hp) at 1.329 s.

| Window | load-0 recs eliminated | min windows/rec @0 % | total load-0 windows @0 % |
|---|---|---|---|
| 0.25 s | 0 | 5 | 205 |
| 0.50 s | 0 | 2 | 102 |
| 1.00 s | 0 | 1 | 49 |
| 2.00 s | **1 (official 174)** | 0 | 20 |

Load 0 is the only 1797 rpm/no-load condition; dropping it would bias
operating-condition coverage, so a 2.0 s window's elimination of a
load-0 recording counts against that candidate. At ≤1.0 s all load-0
recordings contribute.

## 8. JNU guard-width implications

Frozen rule: `G ≥ effective window span`; instantiation
`[b − ceil(G/2), b + ceil(G/2))` around each of the four frozen anchors
per recording (`jnu_guard_study.csv`):

| Window | Guard G | samples @50 k (native) | @25 k | @24 k | @20 k | @32 k | fault rec usable | healthy rec usable | internal fault block fits ≥1 window? |
|---|---|---|---|---|---|---|---|---|---|
| 0.25 s | 0.25 s | 12 500 | 6 250 | 6 000 | 5 000 | 8 000 | 9.01 s (90 %) | 29.03 s (97 %) | yes |
| 0.50 s | 0.50 s | 25 000 | 12 500 | 12 000 | 10 000 | 16 000 | 8.01 s (80 %) | 28.03 s (93 %) | yes |
| **1.00 s** | 1.00 s | 50 000 | 25 000 | 24 000 | 20 000 | 32 000 | 6.01 s (60 %) | 26.03 s (87 %) | yes (exactly 1 @0 %) |
| 2.00 s | 2.00 s | 100 000 | 50 000 | 48 000 | 40 000 | 64 000 | 2.01 s (20 %) | 22.03 s (73 %) | **no (0.002 s left)** |

The sealed Part-2 anchors are untouched; no guards were permanently
instantiated.

## 9. Overlap-policy comparison

| | A: 0/0/0 | **B: 50/0/0** | C: 50/50/50 |
|---|---|---|---|
| training examples (1.0 s, fold mean) | ~7.0 k | ~13.0 k (≈1.86×) | ~13.0 k |
| near-duplicate windows | none | train only (adjacent windows share 50 % of samples — augmentation-like redundancy, acceptable for optimization) | also in val/test |
| metric dependence | independent eval windows | independent eval windows | correlated eval windows → optimistic variance, double-counted samples |
| evaluation interpretability | clean | clean | compromised |
| computational cost | lowest | ≈1.9× train epoch cost | ≈1.9× everywhere |

Recommendation: **Policy B**. Evaluation counts stay at 0 %-overlap
levels (§6), which is the price of interpretable metrics.

## 10. Recommended channel policy (NOT implemented)

Single fixed radial/vertical accelerometer: CWRU `DE_time` ·
JNU `acc_vertical` · HIT `ch3` · MaFaulDa `col3_underhang_radial`;
alternatives and rejected policies per §3.

## 11. Recommended common sampling rate (NOT implemented)

**25 kHz**, with the JNU/MaFaulDa high-frequency truncation explicitly
disclosed (§4) and native-rate processing named as the alternative if
full JNU fidelity is prioritised over a shared physical frequency axis.

## 12. Recommended window duration (NOT implemented)

**1.0 s** (fallback 0.5 s if JNU's 24 evaluation windows per split are
judged too thin — at the cost of HIT relative-speed context dropping to
1.7 revolutions at the slowest group).

## 13. Recommended overlap policy (NOT implemented)

Train 50 % / validation 0 % / test 0 % (Policy B).

## 14. Uncertainties / compromises

1. JNU (and partial MaFaulDa) HF truncation under any common rate — the
   central compromise of 25 kHz; energy ≠ information, but the risk that
   resonance-carrier loss disproportionately hurts JNU classification
   cannot be quantified without downstream experiments (which must not
   drive this choice).
2. HIT contiguous-recording windowing rests on structural evidence + a
   fold-1-train diagnostic; formally an assumption for folds 2/3
   (structure is identical; the diagnostic is corroborative only).
3. HIT ch3's axial station relative to the inter-shaft bearing is
   undocumented (ch3–ch6 ablation reserved).
4. Sensor proximity asymmetry across rigs (CWRU on-fault-housing vs HIT
   casing vs MaFaulDa fixed underhang) is irreducible and disclosed.
5. JNU evaluation thinness at 1.0 s (24 windows/split from 12
   recordings) — statistical resolution of JNU metrics will be low.
6. CWRU 25/48 polyphase conversion is exact but non-trivial;
   anti-alias filter design deferred to Part 3B.

## 15. Decisions requiring human approval

1. Channel policy (§10) — including MaFaulDa fixed-underhang choice.
2. Common rate 25 kHz vs native-rate processing (§11) — with explicit
   acceptance of the JNU/MaFaulDa truncation if 25 kHz is chosen.
3. Window duration 1.0 s (or 0.5 s fallback) (§12).
4. Overlap Policy B (§13).
5. HIT contiguous-recording windowing basis (windows may cross series
   boundaries within a recording) (§6).
6. JNU guard instantiation at G = W upon Part-3B window freeze (§8).
7. Commit checkpoint: Parts 1–3A remain **uncommitted** (see
   reproducibility); a methodology checkpoint commit is recommended
   before any preprocessing/model development.

**HARD STOP.** No channel selected, no resampling performed, no windows
generated, no guards instantiated, no STFT/normalization/encoder/
training work done. Awaiting approval.
