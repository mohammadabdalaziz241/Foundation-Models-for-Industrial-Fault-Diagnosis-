# PART 4C — Fit, Seal, and Expose Fold-Specific N2 Normalization
## methodology_v2 — final frozen representation

Implementation stage (no design/search): the approved representation is
now fully implemented, sealed and exposed through the final reader. No
encoder, masking, SSL, supervised model, heads or few-shot work was
performed; no spectrogram files were precomputed. Tests: 115 passed
suite-wide (13 new Part-4C tests).

## 1. Executive summary

All 12 fold × dataset N2 normalizers were fitted TRAIN-only with a
streaming float64 parallel-Welford accumulator over 38,823 TRAIN windows
(58k–1.92M frames per normalizer), with **zero floored bins anywhere**.
Artifacts are byte-deterministic, registered, and sealed under the
**Part-4C master representation hash
`ee9414e8988c36b8a1ecad7d2622a54439a4bcc180a6a4e6a50b2f256160064f`**.
The final reader `get_representation(window_id, fold_id)` serves the
frozen model-domain tensor (float32, native-resolution, physically
coordinated) after fail-closed verification of the Part-2, Part-3B and
Part-4C seals. Validation/test access is test-proven to leave the sealed
statistics byte-unchanged.

## 2. Checkpoint commit information

Approved methodology Parts 1–4B were checkpoint-committed **before**
Part 4C began: commit `5d1b1e027766a736f632e8356d720e56e1cd50c9`
("methodology-v2: freeze datasets splits windows and spectrogram
design"), 133 files / 115,742 insertions, containing only the
`methodology_v2/`, `src/methodology_v2/`, `scripts/methodology_v2/`,
`tests/methodology_v2/` paths; unrelated pre-existing untracked files
were excluded; not pushed.

## 3. Frozen upstream seals

Part-2 master `527ccc1d449b223a37ecf109ed27be5279e17d85ba4e881abb9c68f4035e69c6`
and Part-3B master `99ffde7e5c0e2cb9b05713801aedcb10b11ccc229d4c2d10a58a1506db10bb51`
verified before and after fitting — intact. The approved Part-4B spec
was hashed before proceeding
(`part4b_approved_spec_sha256 = 7751ccd9b70521d2…`, full value in
`part4c_reproducibility.json`).

## 4. Final representation definition

`representation_spec.yaml` (FROZEN, sealed): 1.0 s sealed raw window →
per-dataset physically matched STFT (CWRU/JNU/MaFaulDa 1024/256, HIT
512/128; periodic Hann, center=False, no padding, one-sided, explicit
implementation) → `log1p(abs(·))` → N2 (per fold × dataset × bin TRAIN
statistics, denominator floor rule std<1e-6 → 1.0) → float32
(freq_bins, time_frames) with physical Hz and seconds coordinates.
Forbidden: resampling, resizing, RGB, dB/power targets,
validation/test statistic fitting.

## 5. Fold-specific normalizer fitting

| Fold | Dataset | TRAIN windows | Frames/bin | Frequency bins | Floored bins |
|---|---|---|---|---|---|
| 1 | CWRU | 320 | 58,880 | 513 | 0 |
| 1 | JNU | 120 | 23,040 | 513 | 0 |
| 1 | HIT | 2,632 | 505,344 | 257 | 0 |
| 1 | MaFaulDa | 9,828 | 1,886,976 | 513 | 0 |
| 2 | CWRU | 307 | 56,488 | 513 | 0 |
| 2 | JNU | 108 | 20,736 | 513 | 0 |
| 2 | HIT | 2,632 | 505,344 | 257 | 0 |
| 2 | MaFaulDa | 10,026 | 1,924,992 | 513 | 0 |
| 3 | CWRU | 198 | 36,432 | 513 | 0 |
| 3 | JNU | 120 | 23,040 | 513 | 0 |
| 3 | HIT | 2,632 | 505,344 | 257 | 0 |
| 3 | MaFaulDa | 9,900 | 1,900,800 | 513 | 0 |

Window counts equal the sealed Part-3B TRAIN manifests exactly
(test-verified). Fitting used only rows with `split == train` and
`fold_id == F` — enforced by a fail-closed guard that is unit-tested
against test, validation and wrong-fold rows. Streaming statistics agree
with direct reference computation to ≤1e-8 on real windows (tested).

## 6. Normalization sanity results

By construction, each normalizer gives exactly zero mean and unit
standard deviation per frequency bin over the full TRAIN frame set it
was fitted on. Sampled empirical checks through the final reader (every
200th TRAIN window): per-window means fluctuate around 0 as expected
(large samples |mean| ≤ 0.11–0.22; the tiny JNU sample of 1 window sits
at 0.73–0.80, consistent with within-dataset window variability that N2
deliberately preserves); per-window stds 0.72–1.14; 100 % finite; no
floored bins. Validation/test received **mechanical checks only**
(finite, shape, normalizer exists, artifacts byte-unchanged — all pass);
no test-distribution summaries were produced.

## 7. Final shapes (verified through the reader, all folds)

CWRU **(513, 184)** · JNU **(513, 192)** · HIT **(257, 192)** ·
MaFaulDa **(513, 192)** — float32, single numeric channel,
(freq_bins, time_frames). `representation_shapes.csv`.

## 8. Physical-coordinate grids

Frequency: `f_k = k·fs/n_fft`, stored in every normalizer — JNU/HIT/
MaFaulDa share identical 48.828125 Hz spacing (HIT ends at 12.5 kHz,
the 50 kHz sets at 25 kHz); CWRU is 46.875 Hz to 24 kHz, **not**
interpolated. Time: `t = frame_index · hop / fs` (5.33 ms CWRU,
5.12 ms others). Embeddings deliberately not implemented (Part 5).

## 9. Representation-reader specification

`src.methodology_v2.part4c_reader.get_representation(window_id, fold_id)`
→ `(tensor, meta)`: verifies all three seals once per process (fail
closed); resolves the window in the sealed fold manifest; loads the raw
window via the Part-3B lazy reader; computes the frozen STFT + log1p;
applies the sealed `normalizer[fold][dataset]`; returns float32
(bins, frames) plus meta (frequency_hz, time_seconds, split, labels,
provenance). float64 is used internally for the STFT and statistics
accumulation only (documented); no caching layer exists — the manifests
and normalizers remain the sole source of truth.

## 10. Reconstruction-target declaration

**Frozen: the future SSL masked-reconstruction target is this
N2-normalized log1p-magnitude STFT tensor.** Mask ratio/geometry,
decoder, loss form beyond the target domain, and weighting are Part-5
decisions, deliberately undefined here.

## 11. N1 future ablation declaration

N1 (per-window zero-mean/unit-std) is registered in the frozen spec as a
pre-registered representation ablation only; it was not fitted or run at
scale. N2 is the primary representation.

## 12. Leakage/integrity proof

- Fitting inputs: exactly the sealed TRAIN manifests per fold (counts
  match; guard fail-closed; tested against val/test/wrong-fold rows).
- Fold independence: three separately fitted normalizer sets; per-dataset
  statistics differ across folds (tested), as required because groups
  swap train/test roles between folds.
- Dataset independence: per-dataset vectors on their own grids; no
  cross-dataset or cross-grid sharing (grid sizes tested).
- Validation/test: representations generated with stored TRAIN
  statistics; normalizer artifacts byte-identical before/after access
  (tested and re-verified in the runner).
- All upstream seals re-verified after every stage.

## 13. Hashes

`normalizer_hashes.csv` seals `representation_spec.yaml`,
`normalizer_registry.csv` and the 12 npz artifacts.
**PART4C_MASTER_REPRESENTATION_HASH =
`ee9414e8988c36b8a1ecad7d2622a54439a4bcc180a6a4e6a50b2f256160064f`.**
Part-5 code must fail closed on Part-2, Part-3B and Part-4C seals
(`verify_frozen_hashes()`, `verify_part3b_hashes()`,
`verify_part4c_hashes()` — the reader already does). Note: `*.npz` is
gitignored, so the committed authority is the hash registry; artifacts
regenerate byte-identically (deterministic zip metadata, tested).

## 14. Known limitations (open disclosures)

- Dataset-specific normalization: the shared encoder operates on
  **per-dataset-standardised** spectrograms (approved N2 disclosure).
- Native Nyquist frequencies differ (12.5/24/25 kHz).
- Frequency dimensions differ (257 vs 513 bins).
- CWRU time dimension is slightly shorter (184 vs 192 frames).
- Future batching requires explicit masking/padding or coordinate-based
  tokenization — deliberately unsolved here.
- JNU statistics rest on the smallest TRAIN pool (108–120 windows from
  12 recordings; within-recording protocol caveat carries forward).

## 15. Items deferred to Part 5 (human discussion required)

Patchification; padding/batching; frequency/time positional embeddings
(Strategy C); encoder architecture (CNN/Transformer/Mamba);
multi-resolution fusion ablation; masking design; SSL objective; S0
supervised baseline; classification heads; samplers (dataset-balanced
principle pre-approved); few-shot subsets; N1 representation ablation.

**HARD STOP.** Part 4C complete; awaiting human approval before Part 5.
