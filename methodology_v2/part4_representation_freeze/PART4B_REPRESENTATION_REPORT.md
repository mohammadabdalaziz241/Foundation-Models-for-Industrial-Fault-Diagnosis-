# PART 4B — Final STFT Verification, Normalization Comparison, Representation Freeze
## methodology_v2

Bounded, lightweight design/freeze study. **No full-dataset STFT
generation** (645 + 66 Fold-1 TRAIN windows read in total), no final
normalizer fitting, no models, no training. Tests: 99 passed suite-wide
(11 new Part-4B tests). Artefacts:
`physically_matched_stft_verification.csv`,
`normalization_comparison.csv`, `amplitude_retention_metrics.csv`,
`dataset_scale_metrics.csv`, `normalization_numerics.csv`,
`part4b_development_windows.csv`, `part4b_signal_access_log.json`,
`proposed_representation_spec.yaml`, `part4b_recommendations.yaml`,
`part4b_reproducibility.json`, 4 figures.

## 1. Executive summary

The approved physically matched STFT is verified with no contradiction:
all four datasets now share ~20.5–21.3 ms analysis frames, ~5.1–5.3 ms
hops and ~46.9–48.8 Hz bins at native rates, with HIT (512/128) exactly
matching the 50 kHz sets and the time-frame mismatch reduced to 184 vs
192. The TRAIN-only N1/N2 study gives an unambiguous picture on the
predeclared axes: **N2 (per-dataset, per-frequency-bin TRAIN
normalization) is recommended** — it removes cross-dataset scale
shortcuts while retaining window-energy information (rank corr
0.96–1.00) that N1 erases by construction, and it yields the richer
masked-reconstruction target. The dataset-identity implication of N2 is
analysed openly (§11). The full proposed representation is specified in
`proposed_representation_spec.yaml`; the normalization field awaits your
approval (§15).

## 2. Upstream seal verification

Part-2 master hash `527ccc1d…` and Part-3B master hash `99ffde7e…`
verified byte-for-byte before and after every run — both intact; no
upstream artefact modified. Part-4A artefacts inspected; the only
refinement relative to 4A's universal-1024 recommendation is the
approved HIT 512/128 physical matching (a human decision, not a
contradiction).

## 3. Physically matched STFT verification

| Dataset | fs | n_fft | hop | frame | hop | Δf | bins | frames | Nyquist |
|---|---|---|---|---|---|---|---|---|---|
| CWRU | 48 kHz | 1024 | 256 | 21.333 ms | 5.333 ms | 46.875 Hz | 513 | 184 | 24 kHz |
| JNU | 50 kHz | 1024 | 256 | 20.480 ms | 5.120 ms | 48.828 Hz | 513 | 192 | 25 kHz |
| HIT | 25 kHz | **512** | **128** | 20.480 ms | 5.120 ms | 48.828 Hz | 257 | 192 | 12.5 kHz |
| MaFaulDa | 50 kHz | 1024 | 256 | 20.480 ms | 5.120 ms | 48.828 Hz | 513 | 192 | 25 kHz |

Design goal met (20–21.3 ms / 5.1–5.3 ms / 46.9–48.8 Hz). HIT is now
physically identical to JNU/MaFaulDa in frame, hop and bin spacing; the
frame-count mismatch fell from 94-vs-192 (universal 1024) to 184-vs-192.
A useful emergent property: **JNU, HIT and MaFaulDa share an identical
bin→Hz grid** (48.828 Hz spacing; HIT simply stops at 12.5 kHz); only
CWRU's grid differs slightly (46.875 Hz). Conventions unchanged and
test-enforced: periodic Hann, center=False, no padding, one-sided rfft,
explicit implementation.

## 4. Exact representation shapes

CWRU (513, 184) · JNU (513, 192) · HIT (257, 192) · MaFaulDa (513, 192)
— (freq bins, time frames), 1 numeric channel, float32 target, native
resolution, no resizing.

## 5. N1 definition

`X_norm = (X − mean(X)) / max(std(X), 1e-8)` per window; windows with
std < 1e-8 are flagged (none occurred). Identity-free, leakage-trivial,
affine per window.

## 6. N2 definition

Per fold × dataset × physical frequency bin: `mu[D,f]`, `std[D,f]`
fitted over TRAIN frames only; `X_norm[:,f] = (X[:,f] − mu[D,f]) /
max(std[D,f], 1e-8)`. Statistics are dataset- and frequency-grid-
specific — never shared across incompatible grids or across folds.
The Part-4B statistics are **dev-sample estimates for comparison only**
(645 deterministic Fold-1 TRAIN windows, every 20th; 2,944–94,272 frames
per dataset; zero floored bins); final per-fold normalizers are NOT
fitted (awaiting approval, §15).

## 7. Numerical comparison (Fold-1 TRAIN dev windows)

- N1: post mean 0, std 1 exactly (machine precision); range
  [−1.34, 5.10]; all structure exactly preserved (affine).
- N2: post ranges [−3.44, 4.46] (p1/p99 ≈ −1.6/2.4); temporal-profile
  correlation with pre-norm mean 0.96 (min 0.64); frequency-profile
  correlation mean 0.35 — the per-bin baseline subtraction is the
  intended rig-baseline compensation; peak contrast slightly reduced
  (e.g. CWRU 4.12 → 3.38 in σ units) but strong.
- Both finite on every window; no floored bins/windows anywhere.

## 8. Amplitude-information comparison

Over the 645-window sample (Spearman rank correlation of
pre-normalization window energy vs post-normalization aggregate):
N1 = 0 exactly for every dataset (per-window standardisation erases
first-order level); **N2 = 0.959 (CWRU) / 1.000 (JNU) / 0.977 (HIT) /
0.992 (MaFaulDa)**, with window-mean spread retained at 0.29–0.64.
Implication: severity/load-related intensity variation survives N2 and
is unavailable to any model under N1. We do not claim amplitude *must*
be preserved — but N1's removal is irreversible, while N2 keeps the
information available for the encoder to use or ignore.

## 9. Dataset-scale shortcut comparison

Unnormalized log1p window means: 0.535 / 1.221 / 0.863 / 1.156
(CWRU/JNU/HIT/MaFaulDa) — a trivial dataset cue. After N1: exactly 0/1
everywhere. After N2: between-dataset means all ≈0 (by construction on
the fitting sample) while within-dataset spread remains (0.29–0.64).
Both candidates eliminate the obvious first/second-order shortcut; no
dataset-origin classifier was trained (per protocol).

## 10. Reconstruction-target implications

N1: decoder can only reconstruct relative within-window structure; all
windows share identical first/second moments — a flatter SSL task.
N2: decoder reconstructs deviations from a stable per-rig spectral
baseline, including genuine window-level energy variation — a richer,
physically interpretable target. Visual evidence: under N2 the
high-frequency fault striations (masked by the resonance baseline in
raw log1p) become prominent (`figures/norm_compare_CWRU.png`).
Methodological recommendation only; no reconstruction model was trained.

## 11. Shared-foundation-model implications

N2 conditions preprocessing on **dataset identity**. Stated openly:
- applied identically to S0, S1 pretraining, S1 fine-tuning, validation
  and test within each fold — no arm ever sees a different pipeline;
- dataset identity is already integral to the benchmark (dataset-specific
  Task-B heads, dataset-balanced sampling), so no new information channel
  is introduced;
- the honest claim becomes "a shared encoder over per-dataset-
  standardised physical spectrograms", analogous to per-corpus feature
  normalization in speech;
- if a fully dataset-agnostic pipeline is ever required for the claim,
  N1 is the documented ablation. N2 is not rejected for using dataset
  identity, but the claim wording must carry this disclosure.

## 12. Recommended normalization

**N2 — per-dataset, per-frequency-bin TRAIN normalization, fitted
independently per fold.** The evidence is unambiguous on the predeclared
axes (amplitude retention, shortcut removal, reconstruction richness,
numerics), matching the pre-study working preference. Not frozen here —
approval requested (§15).

## 13. Proposed final representation specification

See `proposed_representation_spec.yaml`: 1.0 s native-rate raw
acceleration on the frozen channels → physically matched Hann STFT
(center=False, no padding, one-sided) → log1p magnitude → [pending N2]
→ single-channel float32 tensor at native resolution with physical
time/frequency coordinates (`f_k = k·fs/n_fft`) as metadata. No RGB, no
colormaps, no resizing, no image preprocessing. Frequency embedding
(Strategy C) and multi-resolution remain Part-5 items.

## 14. Remaining architecture problems (deferred to Part 5, not solved)

- variable frequency dimension (257 vs 513 bins) and slightly variable
  time dimension (184 vs 192);
- physical-Hz coordinate encoding (Strategy C);
- batch collation across shapes (candidates only: padding with explicit
  masks, coordinate patch sequences, ragged batching, shared
  low-frequency trunk + optional high-frequency patches);
- masking design for SSL.

## 15. Decisions requiring human approval

1. **Freeze normalization = N2** (per-dataset per-bin, per-fold,
   TRAIN-only), with the §11 disclosure adopted into the methodology
   text — or choose N1.
2. Approve `proposed_representation_spec.yaml` as the frozen Part-5
   input contract (with the normalization field resolved).
3. Authorize the bounded post-approval generation step: fit
   `normalizer_fold_{1,2,3}` from each fold's full TRAIN partition and
   seal them (not performed in Part 4B).
4. **Checkpoint commit** — Parts 1–4B remain uncommitted; the approved
   methodology should be committed before representation generation or
   Part-5 architecture work.

**HARD STOP.** No full-dataset STFTs, no final normalizer, no encoder,
masking, SSL, supervised training, heads, probes or few-shot work was
performed. Awaiting approval.
