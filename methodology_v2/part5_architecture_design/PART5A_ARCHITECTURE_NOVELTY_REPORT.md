# PART 5A — Architecture and Novelty Audit
## methodology_v2 — research + design only (nothing implemented)

Seals verified before and after all work: Part-2 `527ccc1d…`, Part-3B
`99ffde7e…`, Part-4C `ee9414e8…` — intact; no upstream artefact touched.
**Part-4C commit status: approved but UNCOMMITTED** (five paths untracked
on top of checkpoint `5d1b1e02`); no unrelated files staged; not
committed (no authorization). Tests: 123 passed suite-wide (7 new
Part-5A tests incl. a guard that rejects any surviving "First…" claim
and an AST guard against model-layer imports).

## 1. Executive summary

The audit found the field substantially more crowded than a naive
proposal would assume: **VibFM** already builds a masked-spectrogram
vibration foundation model (16 datasets, 128×128 standardization +
rate/resolution conditioning); **FISHER → ECHO** already process
machine signals band-split at native rates with seconds-specified STFT;
**SSAMBA/Audio Mamba** already do masked SSL with (bi)directional Mamba
on spectrograms. Every "first-of-kind" claim is therefore rejected. What
survives is narrower and honest: (i) a **methodological** contribution
(sealed leakage-controlled multi-dataset protocol + matched S0/S1),
which is the dissertation's strongest card; (ii) a **mechanism-level**
architectural contribution candidate — tokens carrying **absolute
physical-Hz coordinates** (vs ECHO's normalized f/Nyquist) and a
**frequency-gated cross-band mixer** (FISHER/ECHO leave bands
non-interacting); (iii) a compact (~2.1M) per-band temporal BiMamba
encoder as the recommended embodiment, with a mature axial-Transformer
fallback. Primary recommendation: **Candidate A "PC-STE"**, Small tier,
16×8 patches, S2 sequence organization, C2 absolute-coordinate Fourier
features, F3 gated mixer, masked mean pooling — all pending approval.

## 2. Frozen input representation

Single-channel float32 N2-normalized log1p STFTs at native rates:
CWRU (513, 184) · JNU (513, 192) · HIT (257, 192) · MaFaulDa (513, 192),
with physical Hz/seconds coordinates; JNU/HIT/MaFaulDa share a
48.828125 Hz bin grid, CWRU 46.875 Hz; no resizing, no RGB. The encoder
must accept variable frequency extent without fabricating HIT content
above 12.5 kHz.

## 3. Literature search methodology

Primary sources only (arXiv/publisher pages, official GitHub); 8
recorded web searches on 2026-08-12 (`search_terms_log.txt`) covering the
six named works plus vibration/machinery-specific sweeps (foundation
models, Mamba diagnosis, masked autoencoders, variable sampling rates).
Registry: `literature_registry.csv` (10 entries).

## 4. Closest prior works (key findings)

- **VibFM** ([PHM Society EU 2026](https://papers.phmsociety.org/index.php/phme/article/view/4912)):
  Transformer, masked time-frequency patch reconstruction; 16 vibration
  datasets ≈400 h; **128×128 log-STFT standardization** with a
  conditioning vector encoding sampling rate and time/frequency
  resolution; downstream Paderborn 3-class with **leakage-resistant
  bearing-level evaluation**; frozen-feature and fine-tuning transfer.
  Architectural parameters not stated in the accessible abstract
  (recorded as a residual risk).
- **ECHO** ([arXiv:2508.14689](https://arxiv.org/abs/2508.14689), v3 2026):
  seconds-specified STFT (25 ms/10 ms) → rate-independent frame rate and
  ~40 Hz bins; uniform 32-bin sub-bands, count ∝ rate; **normalized
  f/Nyquist sinusoidal frequency PE**; per-sub-band ViT with CLS concat
  (**no cross-band interaction**); EAT-style teacher–student with dual
  global/frame alignment; pretrained on **audio** (AS2M/MTG/Freesound);
  evaluated on DCASE + CWRU/MAFAULDA/IIEE/IICA via k-NN; 5.5M/22M.
- **FISHER** ([arXiv:2507.16696](https://arxiv.org/abs/2507.16696)):
  fixed-duration STFT window/hop; predefined-bandwidth sub-bands
  processed **individually**; EMA teacher–student self-distillation;
  multi-modal industrial signals; RMIS benchmark; up to 16× smaller than
  90M encoders.
- **SSAMBA** ([SLT 2024](https://arxiv.org/abs/2405.11831)):
  bidirectional Mamba, SSAST-style joint discriminative+generative
  masked-patch SSL; tiny/small/base; ≈92.7 % faster and ≈95.4 % less
  memory than SSAST (tiny, 22k tokens).
- **Audio Mamba** ([Interspeech 2024](https://arxiv.org/abs/2406.02178)):
  masked log-mel patches, selective SSM, outperforms SSAST baselines.
- **SepTr** ([Interspeech 2022](https://arxiv.org/abs/2203.09581)):
  sequential within-time then within-frequency attention; linear
  parameter scaling with input size.
- **SpecTNT** ([ISMIR 2021](https://arxiv.org/abs/2110.09127)):
  spectral encoder emits a per-frame frequency-class token; temporal
  transformer exchanges them — hierarchical spectral→temporal.
- Vibration-specific: **VibrMamba** (Measurement 2025; supervised
  lightweight Mamba on 1D signals), **OpenMAE** (IMWUT 2025; vibration
  MAE with open-domain enrichment), masked-SSL+Swin bearing diagnosis
  (2025). None combines SSL + SSM + heterogeneous vibration pretraining.

## 5. Prior-art component matrix

`prior_art_component_matrix.csv` — 33 features × 9 models
(YES/NO/PARTIAL/UNCLEAR, source-anchored). Headline: native-rate
retention exists (ECHO/FISHER), absolute-Hz coordinates exist **nowhere**
in the reviewed set; cross-band interaction modules are absent from the
band-split machinery models; SSL-Mamba exists only in audio.

## 6. What is definitely NOT novel

STFT/log input; masked spectrogram reconstruction (VibFM/SSAST/MAE);
Transformer or Mamba on spectrograms (AST/SSAMBA/AuM); bidirectional
Mamba (SSAMBA); handling multiple sampling rates without resampling
(FISHER/ECHO); band splitting (FISHER/ECHO); seconds-specified/physically
matched STFT (FISHER/ECHO — our Part-4B config is honest engineering,
not novelty); separate time/frequency modelling (SepTr/SpecTNT);
multi-resolution STFT (extensive audio prior); conv-before-SSM
(WCamba-family); multi-task/dataset-specific heads (standard);
frequency positional encoding as a concept (ECHO).

## 7. Potential novelty spaces (candidates, wording-sensitive)

1. **Absolute physical-Hz token coordinates** — mechanically different
   from ECHO's normalized f/Nyquist: identical mechanical frequency
   bands (e.g. a 3 kHz resonance) align across rates in our scheme,
   whereas normalized encodings align *relative* positions. Physically
   motivated for machinery (defect/resonance bands live at absolute Hz).
2. **Frequency-gated cross-band mixer** conditioned on absolute Hz —
   FISHER and ECHO process sub-bands independently until embedding
   concatenation; an explicit, gated cross-band exchange is absent there.
3. **System combination** — SSL + bidirectional SSM + heterogeneous
   *vibration* pretraining at native rates (audio precedents exist;
   vibration gap remains) — combination claim only, with technical
   justification (linear scaling, streaming extensibility, band-local
   masking), never "first-of-kind".

## 8. Patch geometry comparison

`patch_geometry_study.csv`. At the favoured **16×8** (freq × time):
750/781.25 Hz × 42.7/41.0 ms patches; tokens CWRU 759 (33×23),
JNU/MaFaulDa 792 (33×24), HIT 408 (17×24); padding only on the
frequency remainder (15 bins: 2.84 %, HIT 5.51 %) — CWRU time axis is
exact. 8×8 doubles tokens (1495–1560) for little structural gain; 16×16
halves temporal granularity (12 time patches); 32×8 (1.5 kHz patches)
blurs band structure. **16×8 recommended** — patch spans ~1–2 shaft
revolutions in time at mid speeds and a physically meaningful ~0.75 kHz
band, with moderate token counts for masked reconstruction.

## 9. Variable-shape strategy comparison

`variable_shape_options.csv`. Recommended: **V1 minimal
completion-padding + validity masks**, riding on S2's band-ragged
property — datasets simply contribute different band *counts* (HIT 17,
others 33), so no fake high-frequency content and no interpolation ever;
V3's physical framing is thereby achieved without a dedicated mechanism;
V4 (pure band tokenization) is the FISHER/ECHO core and is used only as
the neutral band definition, never claimed.

## 10. Coordinate encoding comparison

`coordinate_encoding_options.csv`. Recommended: **C2 — fixed Fourier
features of absolute (f_c kHz, t s) + linear projection**: deterministic
for any Hz (arbitrary-rate compatible), minimal parameters, smooth
generalization to unseen coordinates, and mechanically distinct from
ECHO (normalized) and VibFM (rate-conditioning vector). C3 (discretized
embeddings) fails unseen grids; C4 reproduces ECHO/VibFM.

## 11. Sequence organization comparison

`sequence_organization_options.csv`. Recommended: **S2 — temporal
modelling within each frequency band (shared encoder), then cross-band
fusion**: matches machinery physics (resonance bands evolve in time;
impacts are broadband and need cross-band exchange), yields short clean
1D sequences (23–24 tokens/band), makes variable bandwidth natural, and
differs from ECHO/FISHER exactly where they are weakest (no cross-band
module). S4 axial attention is the fallback family (SepTr); S5
hierarchical is rejected as the highest-overlap family (SpecTNT/ECHO).

## 12. Backbone comparison

`backbone_options.csv`. Recommended primary: **B2 bidirectional Mamba**
(4 blocks, d=192; SSL-masking viability evidenced by SSAMBA/AuM; linear
scaling; streaming extensibility) with the **honest caveat** that at
per-band sequence length 23–24 a compact Transformer is computationally
equivalent — the backbone choice is therefore pre-registered as an
equal-capacity ablation (B1 ≈1.95M vs B2 ≈2.13M vs TCN ≈1.66M), and the
dissertation must not claim efficiency wins it cannot demonstrate at
these lengths. B1 is the safest fallback.

## 13. Cross-frequency mixer comparison

`frequency_mixer_options.csv`. Recommended: **F3 — gate =
MLP(band summary ⊕ Fourier(absolute f_c)) → gated exchange across valid
bands** (~3d² params): supports variable Nyquist, missing high bands
(absent bands = masked), differing grids (gate reads Hz, not index).
Verified absent from FISHER/ECHO (bands independent; CLS concat).
F4 frequency-attention is the conservative alternative; F1 mean pooling
is the ablation baseline; F5 frequency-axis SSM rejected (bands are not
a causally ordered sequence).

## 14. Global pooling comparison

Masked **mean pooling** over valid band-time tokens recommended:
simplest, unbiased under variable band counts, and keeps the S0/S1
architecture identical. CLS tokens add sequence-position semantics the
SSM does not need; attention pooling is the documented alternative;
hierarchical/gated pooling folds into F3's job.

## 15. Parameter-budget study

`parameter_budget_estimates.csv` (closed-form, formulas documented in
`part5a_analysis.py`): Tiny 0.74M (d=128, 3 blocks) · **Small 2.13M
(d=192, 4 blocks) — recommended primary** · Medium 5.50M (d=256, 6).
Rough forward activation ≈5.8 MB/window (Small, 792 tokens) ≈371 MB at
batch 64 — comfortable for available GPUs. Context: ECHO-Tiny 5.5M,
ECHO-Small 22M — our Small tier undercuts ECHO-Tiny by ~2.5×,
supporting the lightweight-model objective.

## 16. S0/S1 fairness analysis

Non-negotiable constraint honoured by all candidates: the encoder
(stem + coordinates + temporal blocks + mixer + pooling) is byte-wise
the same architecture for S0 (random init, supervised) and S1 (SSL →
fine-tune); only the masked-reconstruction decoder and masking logic are
S1-pretraining-only and are discarded before fine-tuning. No coordinate
feature, mixer, or mask token may exist only in S1's fine-tuned encoder.
The comparison isolates pretraining history, not architecture.

## 17. Novelty-claim stress test

`novelty_claim_stress_test.csv` — 9 claims adversarially tested:
4 "First…" claims **REJECTED** (VibFM; ECHO/FISHER; SSAMBA/AuM), 1
**TOO STRONG** (native-rate no-resize — FISHER/ECHO equivalent), 3
**POSSIBLY DEFENSIBLE** (absolute-Hz conditioning; Hz-gated cross-band
mixer; SSL-SSM-vibration combination — all with safe revised wording
recorded), 1 **DEFENSIBLE** (sealed leakage-controlled matched-S0/S1
protocol — methodological). A test enforces that no "First…" claim can
ever be marked defensible in the artifact.

## 18. Contribution taxonomy

- **Methodological (strongest)**: hash-sealed leakage-controlled
  4-dataset protocol; matched S0/S1; TRAIN-only diagnostics discipline.
- **Representation (sound practice, not novelty)**: physically matched
  native-rate log1p STFT (cite FISHER/ECHO), N2 fold-sealed
  normalization, explicit physical coordinates.
- **Architectural (modest, mechanism-level)**: absolute-Hz token
  coordinates + frequency-gated cross-band mixer inside a compact
  per-band temporal SSM.
- **Empirical**: what SSL adds over identical supervised training under
  sealed leakage control, incl. label efficiency, on four heterogeneous
  bearing datasets.

## 19–21. Final architecture candidates

`architecture_candidates.yaml`. **Primary: Candidate A "PC-STE"** —
16×8 conv patch stem → absolute-coordinate Fourier features → per-band
temporal BiMamba ×4 (d=192, shared) → Hz-gated cross-band mixer →
masked mean pooling; ≈2.13M params; tokens 408–792; linear complexity;
closest prior FISHER/ECHO/SSAMBA; medium novelty & implementation risk.
**Fallback: Candidate B** — axial compact Transformer (SepTr-style) with
the same stem/coordinates/pooling; ≈1.95–2.4M; zero architectural claim,
minimal risk. **Ambitious: Candidate C** — interleaved mixers +
optional multi-resolution branch; highest risk, houses the
multi-resolution ablation if ever triggered.

## 22. Recommended ablations (4, pre-registered)

1. Physical coordinates → learned index PE (does physical calibration
   matter?).
2. F3 gated mixer → masked mean pooling (does cross-band interaction
   matter?).
3. BiMamba → equal-capacity Transformer (does the SSM matter here?).
4. N2 → N1 normalization (already pre-registered in Part 4C).
(Conditional 5th, from 4A: multi-resolution branch, only on slow-shaft
deficit evidence.)

## 23. Remaining novelty risks

ECHO v3/FISHER are actively evolving — re-verify immediately before
submission; VibFM's full text (conditioning mechanism detail) not yet
inspected beyond the abstract; an unfound work using absolute-Hz
conditioning may exist outside searched venues — all claims carry "to
our knowledge"; the B2-vs-B1 efficiency story must not overreach at
per-band L=24.

## 24. Decisions requiring human approval

1. Adopt Candidate A (PC-STE) as the primary architecture; B as
   fallback.
2. Freeze patch 16×8, S2 organization, C2 absolute-Hz coordinates,
   F3 mixer, masked mean pooling, Small tier (~2.1M).
3. Approve the 4 pre-registered ablations.
4. Approve the exact contribution wording
   (`part5a_recommendations.yaml → contribution_wording`).
5. Commit Part 4C (and 5A) — both approved-but-uncommitted work now
   sits on top of checkpoint `5d1b1e02`.

**HARD STOP.** No patchification, encoder, Mamba/Transformer/mixer/
pooling/masking/decoder/head/optimizer/training code was implemented.
Awaiting approval before Part 5B.
