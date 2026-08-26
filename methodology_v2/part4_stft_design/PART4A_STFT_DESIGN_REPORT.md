# PART 4A — Training-Only STFT / Spectrogram Design and Sharpness Audit
## methodology_v2

Bounded representation-design study. Part-2 seal (`527ccc1d…`) and
Part-3B seal (`99ffde7e…`) verified byte-for-byte before and after every
run — both intact. **All raw signal values came from Fold-1 TRAIN windows
served by the frozen Part-3B lazy reader**, guard-enforced fail-closed;
no validation/test signal content was read. No classifier, probe or any
label/performance signal was used to score configurations. No
full-dataset spectrograms were generated; no resizing, RGB, or image
quantization exists in the model-domain path. Tests: 88 passed suite-wide
(14 new Part-4A tests).

Machine-readable artefacts: `part4a_development_windows.csv`,
`stft_candidate_grid.csv`, `stft_resolution_table.csv`,
`stft_sharpness_metrics.csv`, `stft_dynamic_range_metrics.csv`,
`stft_memory_estimates.csv`, `hit_boundary_audit.csv`,
`frequency_coordinate_study.csv`, `part4a_signal_access_log.json`,
`part4a_recommendations.yaml`, `part4a_reproducibility.json`; 22
deterministic figures under `figures/`.

## 1. Executive summary

The TA's "not blurry" requirement is met without any resizing: the
recommended representation is the **native-resolution log1p magnitude of
a 1024-point, 25 %-hop, periodic-Hann, uncentred, unpadded one-sided
STFT**, giving (513 × 184/192/94) float32 tensors at 0.19–0.39 MB per
1-s window. Visual and quantitative audits show impact trains and
resonance bands sharply resolved on all four datasets at native rates;
HIT fragment joints produce **no** systematic time-frequency artefacts;
multi-resolution complementarity is real but modest (verdict MAYBE —
deferred ablation); the frequency-axis mismatch across native rates is
quantified and Strategy C (physical-Hz coordinates) is recommended for
Part 4B/5. All recommendations await approval (§20).

## 2. Development-data declaration

66 windows, all Fold-1 TRAIN, selected by a fixed label-blind rule (per
dataset × class: first/middle/last recording in sorted order, then the
temporally middle train window of each — no signal values or visual
appearance involved): CWRU 15 (5 labels × 3), JNU 12 (4 × 3), HIT 9
(3 × 3), MaFaulDa 30 (10 × 3). Exact IDs: `part4a_development_windows.csv`
and `part4a_signal_access_log.json` (which also lists the 53 HIT
boundary-audit windows, likewise all Fold-1 TRAIN). A test fails if any
non-TRAIN window enters either set; re-selection is byte-deterministic.

## 3. STFT candidate grid

n_fft ∈ {256, 512, 1024, 2048, **4096**} × hop ∈ {25 %, 50 %} of the
analysis window, per native rate (40 rows, `stft_candidate_grid.csv`).
4096 was added beyond the required minimum with an explicit reason: at
48–50 kHz even n_fft = 2048 gives ~24 Hz bins, coarser than the slowest
shaft frequencies (10 Hz), so a longer candidate was needed to
characterise the resolution ceiling. Convention (identical everywhere,
no library defaults): periodic Hann `0.5 − 0.5cos(2πn/N)`;
**center = False**; **no padding** — only complete frames,
`n_frames = ⌊(N − n_fft)/hop⌋ + 1`; one-sided `numpy.fft.rfft`
(torch.stft/scipy conventions reviewed and rejected because both default
to centred/padded frames that mix synthetic zeros into edge frames).

## 4. Physical time/frequency resolution (hop 25 %)

| n_fft | CWRU 48 kHz | JNU/MaF 50 kHz | HIT 25 kHz |
|---|---|---|---|
| 256 | Δf 187.5 Hz · 5.3 ms frame · (129, 743) | 195.3 Hz · 5.1 ms · (129, 774) | 97.7 Hz · 10.2 ms · (129, 387) |
| 512 | 93.8 Hz · 10.7 ms · (257, 372) | 97.7 Hz · 10.2 ms · (257, 387) | 48.8 Hz · 20.5 ms · (257, 192) |
| **1024** | **46.9 Hz · 21.3 ms · (513, 184)** | **48.8 Hz · 20.5 ms · (513, 192)** | **24.4 Hz · 41.0 ms · (513, 94)** |
| 2048 | 23.4 Hz · 42.7 ms · (1025, 91) | 24.4 Hz · 41.0 ms · (1025, 94) | 12.2 Hz · 81.9 ms · (1025, 45) |
| 4096 | 11.7 Hz · 85.3 ms · (2049, 43) | 12.2 Hz · 81.9 ms · (2049, 45) | 6.1 Hz · 163.8 ms · (2049, 21) |

Shapes are (freq bins, time frames) for one 1-s window; column spacing =
hop (e.g. 1024/25 %: 5.3/5.1/10.2 ms). Full table incl. 50 % hops:
`stft_resolution_table.csv`.

## 5. Transform comparison (Fold-1 TRAIN dev windows, 1024/25 %)

| Transform | p0.1 | median | p99.9 | range (p99.9−p0.1) | Notes |
|---|---|---|---|---|---|
| magnitude | 0.04 | 1.89 | 153.2 | 153.2 | peak-dominated; weak harmonics invisible |
| power | 0.01 | 15.4 | 74 266 | 74 266 | worst conditioning |
| **log1p** | **0.04** | **0.83** | **4.48** | **4.44** | bounded; zeros→0 exactly; weak structure visible |
| dB (20log10(·+1e-10)) | −36.8 | −0.79 | 38.7 | 75.5 | usable; eps floor (−200 dB on true zeros) and wide range |

All transforms finite on every dev window. log1p is the recommended
value transform (§16): best-conditioned range, no epsilon floor,
monotone compression that exposed weak upper-band CWRU harmonics
invisible in raw magnitude, and a smooth non-negative masked-
reconstruction target.

## 6. Sharpness / information diagnostics

Descriptive metrics on log1p (66 windows; `stft_sharpness_metrics.csv`);
no combined "best score" was formed. Key patterns (hop 25 %, means):

- **Resolvable peaks/frame grow ≈linearly with n_fft** (CWRU 10.6→179.9,
  MaFaulDa 6.1→69.6 from 256→4096): finer bins keep resolving real
  structure — the signals are not band-limited blurs.
- **Peak-to-neighbour contrast rises with n_fft** for HIT (0.65→0.93)
  and MaFaulDa (0.79→0.91): spectral lines sharpen with longer frames.
- **Time-gradient energy stays flat or peaks near 1024** (JNU 0.54→0.63
  at 1024, then flat): transient localisation is preserved up to 1024
  and begins to saturate beyond — the temporal side of the trade-off.
- Normalized spectral entropy rises with n_fft for all datasets
  (more bins → flatter normalized distribution); reported for
  completeness, interpreted only jointly with the peak metrics.

Limitations: these are contrast/gradient/counting heuristics on the
development subset; they characterise resolution, not diagnosability,
and were not optimised against.

## 7. Dataset-specific visual findings (all figures TRAIN-only)

- **CWRU** (fs 48 k): impact striations + resonance bands at 2.5–4.5 kHz
  razor-sharp at 1024/256; log1p exposes upper-band harmonic ladders
  invisible in raw magnitude (`figures/class_CWRU_*.png`).
- **JNU** (fs 50 k): fault impulses appear as broadband streaks
  concentrated **above 12.5 kHz**, visually confirming the Part-3A
  energy audit and the native-rate decision; streaks are crisply
  time-localised at 5.1 ms hop.
- **HIT** (fs 25 k): casing spectrum dominated by 2–4 kHz bands with
  dense low-frequency rotor content; structure clean at 1024/256
  (41 ms frames acceptable at 25 kHz).
- **MaFaulDa** (fs 50 k): class-dependent harmonic ladders and HF
  content; both resolved; imbalance/misalignment show low-frequency
  order structure, bearing classes show HF bands.

## 8. Native-rate / physical-frequency alignment problem

Confirmed and quantified (`frequency_coordinate_study.csv`, n_fft=1024):

| bin k | CWRU Hz | JNU Hz | HIT Hz | MaFaulDa Hz |
|---|---|---|---|---|
| 21 | 984.4 | 1025.4 | 512.7 | 1025.4 |
| 85 | 3984.4 | 4150.4 | 2075.2 | 4150.4 |
| 256 | 12 000 | 12 500 | 6250 | 12 500 |
| 512 | 24 000 | 25 000 | 12 500 | 25 000 |

The same row index means different physics (HIT rows ≈ half the Hz of
the 50 kHz sets). This is the expected, accepted consequence of the
frozen native-rate decision — resolved by coordinate encoding (§9), not
by resampling.

## 9. Frequency-coordinate strategy comparison

- **A — raw bin index**: rejected; silently misaligns physics across
  datasets (table above).
- **B — f/Nyquist**: single [0,1] coordinate but 0.8 = 10 kHz (HIT) vs
  20 kHz (50 k sets); physical misalignment survives.
- **C — physical Hz coordinate**: attach `f_k = k·fs/n_fft` to every
  bin/patch; enables physical-frequency positional encoding; preserves
  alignment with zero information loss. **Recommended for Part 4B/5.**
- **D — shared low band + dataset-specific high band**: conceptually
  viable (all rates share 0–12.5 kHz), but it is an architectural
  special case that can be expressed inside C later if needed.

## 10. Multi-resolution analysis

512/128 vs 4096/1024 matched comparisons on the same TRAIN windows
(`figures/multires_*.png`): the short config isolates impulse timing,
the long config resolves harmonic ladders (~4× peaks/frame, higher
contrast). Complementarity is real but **modest** — the 1024/256 middle
ground already shows both structure families. **Verdict: MAYBE** — start
single-resolution; hold {512/128 + 4096/1024} (~0.7 MB/window combined)
as a pre-registered Part-5 ablation triggered only by encoder deficits
on slow-shaft/harmonic-dominated cases, not implemented now.

## 11. HIT fragment-boundary spectrogram audit

60 joints in 53 deterministic Fold-1 TRAIN HIT windows
(`hit_boundary_audit.csv`, `figures/hit_boundary_*.png`), boundaries
untouched (no smoothing/cross-fade):

- time-domain jump ratio (|Δx| at joint ÷ window p99.9 |Δx|): median
  0.23, max 1.23 — joints behave like ordinary samples;
- boundary-frame energy z-scores: mean −0.02, max 2.58 across 300
  boundary-touching frames — within ordinary variation, no broadband
  vertical streaks;
- boundary-frame spectral flatness 0.8182 vs window mean 0.8178 — no
  excess broadbandness.

**Verdict: ordered concatenation introduces no detectable artificial
time-frequency structure.** Nothing to fix; nothing was hidden.

## 12. Representation memory / cost (float32, per 1-s window)

| Config | CWRU | JNU/MaF | HIT | batch-64 (50 k sets) |
|---|---|---|---|---|
| 512/25 % | 0.36 MB | 0.38 MB | 0.19 MB | 24 MB |
| **1024/25 %** | **0.38 MB** | **0.39 MB** | **0.19 MB** | **25 MB** |
| 2048/25 % | 0.36 MB | 0.37 MB | 0.18 MB | 24 MB |
| 4096/25 % | 0.34 MB | 0.35 MB | 0.16 MB | 22 MB |
| dual 512+4096 | 0.70 MB | 0.73 MB | 0.35 MB | 47 MB |

Element counts are nearly n_fft-invariant at fixed hop ratio (bins ×
frames trade off); batch memory is dominated by hop ratio. 50 % hops
halve everything (`stft_memory_estimates.csv`). Note the datasets differ
in frame count (94–192) — the encoder must handle variable time length
or the batching must group by dataset (Part-5 concern).

## 13. Normalization strategy analysis (nothing fitted)

Fold-1 TRAIN dev evidence: per-dataset log1p scale offsets are large
(means 0.53 CWRU / 1.30 JNU / 1.01 HIT / 1.24 MaFaulDa; p99 2.98–5.06).
Consequences: **global multi-dataset statistics would leave dataset
identity as a dominant scale cue** (shortcut risk) and mis-scale CWRU;
per-window standardisation removes absolute level (possibly informative)
but is leakage-trivial; per-dataset per-frequency-bin TRAIN statistics
preserve within-dataset amplitude structure while removing cross-dataset
scale. Frozen rule reaffirmed: any statistic is fit on the fold's TRAIN
only; val/test reuse stored TRAIN statistics; nothing was fitted in 4A.
Recommendation for 4B: compare per-dataset-per-bin vs per-window;
reject global-stats as primary.

## 14–18. Recommendations (all PENDING APPROVAL, none implemented)

14. **Single-resolution STFT**: n_fft 1024, hop 256 (25 %), periodic
    Hann, center=False, no padding, one-sided — native rates.
15. **Multi-resolution**: MAYBE — deferred pre-registered ablation
    {512/128 + 4096/1024}; not built now.
16. **Value transform**: log1p(|STFT|).
17. **Physical-frequency treatment**: Strategy C — physical-Hz
    coordinates / positional encoding at Part-5.
18. **Normalization direction**: per-dataset per-frequency-bin TRAIN
    stats vs per-window standardisation to be decided in Part 4B;
    global multi-dataset stats rejected as primary.

## 19. Remaining uncertainties

- Shaft-frequency lines (10–30 Hz) are sub-bin at every candidate;
  rotational periodicity lives in impulse-train time structure — an
  accepted limit of 1-s fixed-rate STFT (order tracking out of scope).
- Sharpness metrics are descriptive heuristics on 66 dev windows;
  fold-1-train scope means folds 2/3 rest on identical geometry, not
  identical measurements.
- HIT 41 ms frames at n_fft 1024 are 2× longer (in ms) than the 50 kHz
  sets at equal n_fft — an accepted native-rate asymmetry, visible in
  the resolution tables.
- Variable frames-per-window across datasets (94–192) defers a real
  architectural decision (padding/packing/per-dataset batching) to
  Part 5.

## 20. Decisions requiring human approval before Part 4B

1. Freeze single-resolution STFT = 1024/256 Hann (center=False,
   no padding, one-sided) at native rates.
2. Freeze value transform = log1p.
3. Adopt Strategy C (physical-Hz coordinates) as the Part-4B/5 frequency
   treatment.
4. Approve the normalization comparison plan for 4B (per-dataset-per-bin
   vs per-window; TRAIN-only fitting).
5. Confirm multi-resolution is deferred as a pre-registered ablation.
6. Accept the HIT joint-audit verdict (no artefact; no mitigation).
7. **Checkpoint commit**: approved methodology Parts 1–3B (and now 4A)
   remain uncommitted — they should be checkpoint-committed before
   Part 4B / full representation generation.

**HARD STOP.** No full-dataset STFT generation, no frozen representation,
no normalization fitting, no encoder/SSL/supervised work was performed.
Awaiting approval.
