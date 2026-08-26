# Data integrity report — methodology_v2 Part 1

All checks were non-destructive (read-only). Machine-readable per-file
results: `integrity_details.json`; file hashes: `raw_file_hashes.csv`
(2,094 hashed files/objects). Raw data were not modified; the spot-check
test `test_original_data_unmodified_spot_check` re-hashes a sample on every
test run.

## Summary verdict

| Dataset | Files checked | Missing | Unreadable | NaN/Inf | Constant | Short | Exact duplicates | Verdict |
|---|---|---|---|---|---|---|---|---|
| CWRU | 116 .mat | 0 | 0 | 0 | 0 | 0 (known-short 48k load-0 files documented) | 4 (expected, see C2) | PASS |
| JNU | 12 .csv | 0 | 0 | 0 | 0 | 0 | 0 | PASS |
| HIT full (Drive) | 5 .npy (134 recordings) | 0 | 0 | 0 | 0 | 0 | 0 | PASS |
| HIT github shards | 10 .mat | 0 | 0 | 0 | 0 | — | 360 duplicate windows within training shards (H3) | PASS with findings |
| MaFaulDa | 1,951 .csv | 0 | 0 | 0 | 0 | 0 | 0 | PASS |

All 1,951 MaFaulDa CSVs are exactly 250,000 rows × 8 columns; all HIT
recordings and CWRU/JNU signals contain no NaN, no infinities, no constant
channels, and no unexpectedly short signals beyond the documented CWRU
cases below.

## Findings

### CWRU

- **C1 — provenance verified.** Every local file's internal MATLAB variable
  names carry the canonical Bearing Data Center file number (X105 →
  105.mat). The copy reproduces the known official quirks exactly, which is
  strong evidence of authenticity: 98.mat lacks its RPM variable; 99.mat
  contains leftover `X098_*` variables and an `ans` variable; 175.mat
  (48 kHz IR014, 1 hp) contains stray `X217` variables; 217.mat contains
  leftover `X215_*` variables; 173.mat is short (63,788 samples ≈ 1.3 s);
  the 28-mil 12 kHz files are DE-channel-only with internal ids
  X048–X051/X056–X059 (official download names 3001–3008).
- **C2 — byte-identical normal duplicates.** The four `Normal_*` files in
  `data/raw_cwru_48k/normal/` are byte-identical to those in
  `data/raw/normal/` (sha256 match). They are listed once in the manifest;
  double-counting them as independent recordings would be an error.
- **C3 — sampling-rate correction carried.** Per the frozen audit
  `docs/cwru_legacy_rate_impact_note.md`, the normal-baseline files are
  genuinely 48 kHz (not 12 kHz as legacy pipelines assumed). The manifest
  records 48 kHz for all normals. Any future within-CWRU task must not let
  sampling rate become a class-correlated shortcut (28-mil specs exist only
  at 12 kHz; OR021@3 only at 48 kHz; normals only at 48 kHz).
- **C4 — short 48 kHz load-0 files.** Several 48 kHz fault recordings at
  0 hp are ~2.6–5.1 s (e.g. 173.mat 1.3 s) vs ~10.1 s for loads 1–3.
  Documented in the manifest durations; not corruption.

### JNU

- **J1 — healthy files are 3× longer.** `n*_3_2.csv` have 1,501,500
  samples (30.03 s) vs 500,500 (10.01 s) for all fault files. The source
  documents no reason. A discontinuity probe at the 1/3 and 2/3 boundaries
  found no concatenation evidence (boundary first-differences at 0.05–0.33×
  the local 99.9th percentile). Treated as continuous recordings; the
  asymmetry remains undocumented and is disclosed.
- **J2 — undocumented filename suffixes.** `_2` on all files and `_3_2` on
  the healthy files are not explained by the source; recorded verbatim.

### HIT

- **H1 — full-release structure verified.** data1–data5.npy have shape
  (n_series, 8, 20480): rows 1–2 displacement, 3–6 casing acceleration,
  row 7 = LP speed at sample 0 / HP speed at sample 1, row 8 = label at
  sample 0. Series counts (504/450/504/504/450) and speed plans (28/25/28/
  28/25 contiguous speed groups) match paper Tables V–VI; labels match the
  documented per-session bearing states everywhere.
- **H2 — official GitHub split is window-level random (leaky).**
  All 12,060 released 2048-sample windows were matched byte-exactly (as
  float32) to aligned sub-windows of **channel 1** of the full release.
  They originate from 2,340 of the 2,412 series, and **all 2,340 of those
  series contribute windows to BOTH the training shards and the test
  shard**. ytest class counts (954/1008/450) are exactly one fifth of the
  per-class pool — a stratified random window split. The official split
  provides no run-, speed-, or recording-level separation and should not
  be inherited by methodology_v2.
- **H3 — 360 duplicate windows inside the official training shards.**
  12,060 released windows = 11,700 unique + 360 exact duplicates
  (all within/between training shards; none crosses into xtest).
- **H4 — dtype inconsistency.** data2.npy is float32 while the other four
  sessions are float64. Session-correlated encoding artefact; low expected
  impact (both healthy sessions exist, data1 is float64), noted for
  Part 2 preprocessing decisions.
- **H5 — 72 series unused by the official release.** 2,412 series exist;
  the released windows touch 2,340. The identity of the excluded series is
  recoverable from the matching if ever needed.

### MaFaulDa

- **M1 — sequence counts match the official documentation exactly**, per
  configuration (49 normal; 333 imbalance; 197/301 misalignment; 558
  underhang; 513 overhang; 1,951 total).
- **M2 — fault-type naming permutation on the official website.** The
  website's summary table (§1.4) and the archive folder taxonomy agree:
  per position, cage_fault=188, outer_race=184(underhang)/188(overhang),
  ball_fault=186(underhang)/137(overhang). The website's *detailed*
  per-weight tables (§1.4.5) assign those exact count patterns to
  differently-named fault types, consistently permuted:
  folder `ball_fault` ↔ detailed table titled "inner track";
  folder `outer_race` ↔ detailed table titled "rolling elements";
  folder `cage_fault` ↔ detailed table titled "outer track".
  Because summary table, folder names and the published MaFaulDa papers
  agree on {ball, cage, outer}, the folder taxonomy is treated as
  operational ground truth and the detailed-table headings as the likely
  error — but the semantic identity of MaFaulDa's bearing subfaults
  carries a documented residual ambiguity. Consequence: Task-A mappings of
  MaFaulDa `outer_race`/`ball_fault` are CONDITIONAL (see
  `label_mapping_candidate.yaml`); Task B is unaffected (labels stay
  dataset-native).
- **M3 — website typo.** §1.4.4 lists vertical misalignment "17.8 mm" —
  the archive folder is 1.78 mm (all other severities match). Cosmetic.
- **M4 — no duplicates, no gaps.** No duplicate payloads among the 1,951
  sequences; every file parsed cleanly as 8-column float CSV.

### Cross-dataset

- **X1 — no cross-dataset duplicate raw bytes** (sha256 across all 2,094
  hashed objects collide only for the four expected CWRU normal pairs).
- **X2 — archive integrity.** MaFaulDa full.zip (12,896,655,242 bytes,
  server content-length matched; sha256 in `reproducibility.json`) passed
  a full CRC test via Python zipfile. Note: the system `unzip` (Info-ZIP
  6.00) cannot read this zip64 archive — use Python.

## Suspicious-label scan

- CWRU: filename-encoded specs vs internal canonical ids cross-checked —
  no contradictions.
- HIT: per-series label row vs documented session state — no
  contradictions.
- MaFaulDa: labels are the directory taxonomy itself; M2 is the only
  (documentation-level) concern.
- JNU: labels are filename prefixes per the readme; no contradictions.
