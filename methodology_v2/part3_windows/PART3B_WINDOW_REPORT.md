# PART 3B — Frozen Raw-Signal Window Extraction
## methodology_v2

Manifest-first design: the frozen artefacts are window IDENTITIES
(source → frozen group → frozen split → channel → exact sample interval);
signals are served lazily and unmodified by
`src/methodology_v2/part3b_reader.py`. No raw data were duplicated, no
resampling/filtering/STFT/normalization/training was performed. The
Part-2 seal (`527ccc1d…`) was verified byte-for-byte before and after
generation and is untouched. Tests: 74 passed suite-wide (19 new
Part-3B tests).

## 1. Executive summary

54,254 deterministic 1-second raw-window identities were generated
across the three frozen global folds (fold 1: 18,070 · fold 2: 18,140 ·
fold 3: 18,044), at native sampling rates, on the frozen per-dataset
channels, with train 50 % / eval 0 % overlap. **Every actual count
matches the Part-3A estimate exactly (zero discrepancies in all 36
fold × dataset × split cells).** JNU guards were instantiated at
G = 1.0 s from the sealed Part-2 anchors; HIT logical streams were
reconstructed by ordered fragment concatenation with full per-window
provenance. Everything is sealed under Part-3B master hash
`99ffde7e5c0e2cb9b05713801aedcb10b11ccc229d4c2d10a58a1506db10bb51`.

## 2. Frozen parameters

- Channels: CWRU `DE` · JNU `acc_vertical` (only channel) · HIT `ch3`
  (array row 2) · MaFaulDa `col3_underhang_radial` (CSV column index 2).
  No label- or condition-dependent switching; tachometer/microphone/
  displacement/alternative accelerometers excluded.
- **Native rates kept — no resampling** (CWRU 48 kHz, JNU 50 kHz, HIT
  25 kHz, MaFaulDa 50 kHz; asserted per recording from the frozen
  manifests). The Part-3A 25 kHz candidate was rejected by approved
  decision after the train-only energy audit (JNU HF truncation).
- Window: exactly 1.000 s → 48,000 / 50,000 / 25,000 / 50,000 samples
  (derived from the verified rate, not hard-coded).
- Stride: train 0.5 s (50 % overlap — documented as training
  augmentation, not independent evidence), validation/test 1.0 s (0 %).
- JNU guards: G = 1.000 s = 50,000 native samples;
  `[b − 25,000, b + 25,000)` around each of the 48 frozen anchors.
- HIT: ordered concatenation ONLY within one audited
  session × speed-group × ch3 × label stream, preserved acquisition
  order, no value modification at joins.
- No padding, no partial windows, remainders discarded.

## 3. Actual window counts

| Fold | Split | CWRU | JNU | HIT | MaFaulDa | Total |
|---|---|---|---|---|---|---|
| 1 | train | 320 | 120 | 2,632 | 9,828 | 12,900 |
| 1 | validation | 162 | 24 | 280 | 2,060 | 2,526 |
| 1 | test | 105 | 24 | 280 | 2,235 | 2,644 |
| 2 | train | 307 | 108 | 2,632 | 10,026 | 13,073 |
| 2 | validation | 105 | 24 | 280 | 2,205 | 2,614 |
| 2 | test | 169 | 24 | 280 | 1,980 | 2,453 |
| 3 | train | 198 | 120 | 2,632 | 9,900 | 12,850 |
| 3 | validation | 169 | 24 | 280 | 2,225 | 2,698 |
| 3 | test | 162 | 24 | 280 | 2,030 | 2,496 |

Grand total 54,254. Per-class detail for every fold/split:
`window_statistics.json → counts` (e.g. fold-1 HIT train
1,036/1,120/476 for healthy/inner/outer; fold-1 MaFaulDa train spans
all 10 classes, 315–2,142 windows each; CWRU per-specimen 28–66 train
windows, 28–35 eval windows).

## 4. CWRU load/RPM coverage

Every fold/split covers all four loads (e.g. fold-1 train 0hp:35,
1–3hp:95 each; fold-1 test 0hp:15, 1–3hp:30 each). All **13** retained
load-0 recordings contribute ≥1 window in every fold (shortest,
official 174 at 1.329 s, yields exactly 1). RPM metadata (1718–1797,
in-file measured) travels on every row.

## 5. JNU class × speed × block coverage

All 12 class × speed cells appear in train, validation and test of
every fold (12/12/12; verified). Usable regions after guard carving:
fault blocks A/E 75,100 samples, B/C/D 50,100; healthy A/E 275,300,
B/C/D 250,300. Fold roles unchanged (F1 ABC/D/E · F2 BCD/E/A ·
F3 CDE/A/B). JNU evaluation remains **within-recording temporal
holdout** — never unseen-bearing.

## 6. HIT logical-stream reconstruction

134 streams (one per audited session × speed-group), each exactly 18
fragments × 20,480 samples = 368,640 samples (14.7456 s) in preserved
order — full geometry in `hit_logical_stream_manifest.csv` (hashed).
3,192 windows per fold. Because a 25,000-sample window exceeds the
20,480-sample fragment, **100 % of HIT windows cross ≥1 fragment
boundary** — an arithmetic property of the approved concatenation
policy, not an anomaly: 2,508 windows/fold cross exactly one boundary,
684 (21.4 %) cross two. Every window records its contributing fragment
ids and the exact boundary positions crossed. Lazy fragment-sliced
extraction is test-proven value-identical (bit-exact, dtype-exact) to
reference full-stream concatenation, including across boundaries and
for the float32 session data2.

## 7. MaFaulDa configuration/class coverage

Inherited exactly from Part 2: 58/16/16 group units per fold
(41 fault configurations + 49 normal recordings); all 10
folder-taxonomy classes present in every split of every fold; taxonomy
strings preserved verbatim (naming caveat M2 not remapped); channel
fixed to underhang radial regardless of fault position (adaptive
selection remains rejected).

## 8. Part-3A estimate vs actual

Zero difference in all 36 cells (`window_statistics.json →
estimate_comparison`). The Part-3A duration-based formulas and the
Part-3B sample-exact enumeration agree everywhere; no explanation
clauses were needed.

## 9. Integrity checks

- Manifest-exhaustive: exact samples/window per native rate; start ≥ 0;
  end ≤ source length (HIT: ≤ stream length); no zero-length/partial
  windows; no duplicate window ids; valid channels only.
- Signal-level (through the lazy reader, every 50th window = 1,086
  windows across all folds/datasets/splits): 0 wrong lengths,
  0 non-finite, 0 constant/near-constant windows. Test-window checks
  were mechanical only (length/finite/constant) — no test-signal
  characteristic was inspected for design.
- Part-1 file-level integrity (no NaN/Inf/constant channels anywhere in
  the retained sources) provides full-coverage backing.

## 10. Leakage checks

- CWRU: every specimen's windows in exactly one split per fold
  (group → 1 split verified); assignments equal Part-2 rows.
- HIT: every session × speed-group's windows in exactly one split; no
  window spans two Part-2 groups (windows live inside single streams by
  construction; verified against stream bounds).
- MaFaulDa: no configuration and no normal recording crosses splits.
- JNU: every window inside its assigned macro-block's usable region;
  zero guard overlap (48 guards × all windows checked); block roles
  match the frozen rotation.
- Evaluation windows tile disjointly (0 % overlap, no duplicate
  intervals); train overlap is exactly the declared 50 % stride.

## 11. Limitations

- **Native sampling rates**: the shared encoder will receive STFT
  representations originating from different acquisition rates
  (48/50/25/50 kHz). Physical-frequency alignment is deferred to
  Part 4 / architecture design rather than discarding high-frequency
  information by forced 25 kHz resampling.
- **JNU**: still within-recording temporal holdout; 24 evaluation
  windows per split (thin, known and accepted).
- **HIT**: logical 1-second streams require ordered concatenation of
  source fragments within the audited session × speed-group; every
  window crosses ≥1 fragment joint (continuity evidence: Part-3A
  fold-1-train diagnostic). Outer-race class still rests on one
  physical bearing (Part-2 disclosure).
- CWRU remains 3-class (no Healthy), seen-specimen across loads within
  partitions, unseen-severity test sets.

## 12. Files/hashes

| File | SHA-256 |
|---|---|
| window_manifest_fold_1.csv | ccd0b9a2f616c90b12d541a4ffbedd539fefc1bd7d6204062724c0166a5a63ab |
| window_manifest_fold_2.csv | 2be033786cbe88fb88df72ed66148caa113fac2e3d3edf5297b0c7412b3ffeb8 |
| window_manifest_fold_3.csv | f776a61f65145884a6b2f3150074d2ce83c2eeade947aa74c31de15812330faf |
| jnu_guards_1s.csv | f145365331266f10321b8f09a58934c46b4690fc01f8bf114b36ee9cafaada92 |
| hit_logical_stream_manifest.csv | ba9d8109a347c6e1534da23ed1cfedb6ac82eaf042a76ec0ca24e1d70d404be3 |
| **PART3B_MASTER_HASH** | **99ffde7e5c0e2cb9b05713801aedcb10b11ccc229d4c2d10a58a1506db10bb51** |

Future STFT generation must call
`src.methodology_v2.part3b_windows.verify_part3b_hashes()` (and the
Part-2 verifier) first — both fail closed. Reruns are byte-identical
(tested). Window IDs are deterministic composites
(`f{fold}:{dataset}:{recording}:{channel}:{split}:{start}-{end}`), no
UUIDs.

## 13. Decisions remaining for Part 4

1. STFT parameterisation per native rate and the physical-frequency
   alignment strategy across 25/48/50 kHz inputs.
2. Log/amplitude scaling and the train-only normalization statistics
   (rule already frozen: fit on TRAIN of the fold, reuse for val/test).
3. Encoder architecture, masking strategy, SSL objective, samplers
   (dataset-balanced principle pre-approved, not implemented).
4. Whether to build an optional deterministic window cache (reader
   currently lazy-only).
5. Methodology checkpoint commit — Parts 1–3B remain uncommitted.

**HARD STOP.** Awaiting approval before Part 4.
