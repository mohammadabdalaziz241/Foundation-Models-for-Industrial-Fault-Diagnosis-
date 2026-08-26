# PART 5C — SSL Masking, Decoder, and Reconstruction-Objective Design Study
## methodology_v2 (design only — no training of any kind)

All signal content: Fold-1 TRAIN dev windows (the 66-window Part-4A
subset). No labels or downstream performance influenced any choice.
Tests: **153 passed** suite-wide (10 new Part-5C tests, including the
non-negotiable mixer-gradient requirement). All upstream seals verified
before and after (§3, §23 of the summary).

## 1. Executive summary

The proposed S1 pretraining problem (full spec:
`proposed_ssl_spec.yaml`, PENDING approval): **independent random
masking of 60 % of valid 16×8 patches (M1)** — a data-driven choice, as
the redundancy audit shows the simplest scheme is already non-trivial —
with a **learned mask token (E1)**, a **119.6k-parameter per-token MLP
decoder (D1)** receiving **additive post-mixer band context (X1)**,
reconstructing the frozen N2 target under **masked valid-cell MSE
averaged per window then per batch**. The gradient path through the
Hz-gated mixer is proven by a bounded dummy backward and enforced by a
permanent test. Trivial-baseline MSEs (0.88–1.85) are recorded as the
bar a trained model must beat. SSL is computationally feasible even on
the reference Mamba backend (~11–14 h for 60 epochs × 3 folds).

## 2. Mamba parity-gate result (Stage 1)

**PASSED.** Our `selective_scan` vs the OFFICIAL
`selective_scan_ref` (state-spaces/mamba, pinned commit
`e9594ce1c732d97440f0332fdc43170a2294dbfa`, vendored verbatim,
Apache-2.0; `einops` 0.8.2 added solely for it): 10 bounded synthetic
cases (batch 1–4, seq 8–24, d_inner 32–384, d_state 8–16, scales
0.1–3.0, ±z-gating, delta-bias+softplus, variable B/C, D skip) —
**forward BIT-EXACT (max abs err 0.0)**; gradient deviations ≤1.95e-3
absolute on gradients of magnitude in the thousands = **max relative
error 2.0e-7** (float32 accumulation-order noise; several gradients
bit-exact). Transparency: the initially declared absolute-only gradient
tolerance failed on the extreme-scale cases and was revised to
`abs≤1e-4 OR rel≤1e-6`, with the original failure recorded in
`part5_encoder/mamba_reference_parity.json`. Full official *block*
parity is not constructible without the uninstallable package (stated);
the scan — the only non-standard computation — is verified. Equations
and architecture hash unchanged (`962bb1de…`).
**Part-5B checkpoint commit: `2eb0127ac01c2f2143b74e95391d51c85d0cf1bd`**
(33 files, Part-5B only, not pushed).

## 3. Frozen encoder contract

PC-STE per commit `2eb0127a`: 16×8 patches, absolute-Hz/seconds Fourier
coordinates, shared per-band BiMamba ×4 (d=192), Hz-gated cross-band
mixer, masked pooling; 2,382,033 params; token grids CWRU 33×23, JNU
33×24, HIT 17×24, MaFaulDa 33×24. Seals verified pre/post: Part-2
`527ccc1d…`, Part-3B `99ffde7e…`, Part-4C `ee9414e8…`, Part-5B hash
`962bb1de…`.

## 4. Representation redundancy audit (`mask_redundancy_study.csv`)

At the 16×8 **patch** level (66 TRAIN windows): temporal lag-1 patch
correlation is LOW — CWRU 0.086, JNU 0.057, HIT 0.183, MaFaulDa 0.077 —
and frequency-neighbour correlation is 0.003–0.354. The
neighbour-interpolation residual ratio (residual energy of predicting a
patch from its two temporal neighbours ÷ patch variance) is **1.24–1.67,
i.e. >1**: copying neighbours is WORSE than predicting each patch's own
mean. Despite the 75 % frame-level STFT overlap, N2 whitening and the
41 ms patch span decorrelate patches strongly. Consequence: **isolated
random masking is NOT trivial** for this representation.

## 5. Mask-geometry comparison (`mask_geometry_options.csv`)

M1 random (1×1) · M2 blocks (2×3 ≈ 1.5 kHz × 125 ms) · M3 time spans
(1×4 ≈ 165 ms) · M4 band spans (3×1 ≈ 2.3 kHz) · M5 mixed — all
implemented in one deterministic generator, all hitting the exact
target count over valid patches, none producing fully-masked bands or
time columns at 60 % (0/20 seeds, all datasets).
**Recommendation: M1** — the instruction is the simplest sufficiently
non-trivial scheme, and §4 plus the baselines (§17) show M1 already
defeats copy-based prediction. M2 stays registered as the secondary
harder variant. (My prior favoured M2; the measurements overruled it.)

## 6. Mask-ratio comparison (`mask_ratio_study.csv`)

40–75 % over valid patches: at the recommended 60 %, masked/visible =
455/304 (CWRU), 475/317 (JNU/MaF), 245/163 (HIT). 60 % balances
context sufficiency against difficulty; 75 % (MAE-style) is unnecessary
given §4's low redundancy, 40 % leaves the task dominated by visible
context. Encoder cost is ratio-independent (full-sequence encoding with
mask tokens — the band-sequence structure precludes MAE-style token
dropping). Padding never counts toward the ratio.

## 7. Physical mask interpretation

One masked patch ≈ 0.75–0.78 kHz × 41–43 ms everywhere — the same
physical statement across datasets (patch grids are physically matched
by construction), so a single mask policy is dataset-fair; no
dataset-specific tuning exists.

## 8. Mask-token strategy

**E1 learned mask token recommended.** E2 (zero embedding) is ambiguous
in N2 space: zero = "dataset-average level", and genuinely quiet
patches occur, so the encoder could not distinguish "hidden" from
"average-valued". One d=192 learned vector (192 params) removes the
ambiguity. The token replaces the stem output entirely; coordinates are
still added — the model knows a patch exists at 7.5 kHz / 0.4 s but
never sees its values.

## 9. Decoder comparison (`decoder_options.csv`)

D1 per-token MLP + X1 context: **119,552 params (5.0 % of encoder)** —
recommended; cross-time/cross-band modelling belongs to the encoder,
which is exactly what SSL should force. D2 (1-block d=128 transformer
decoder, ~246k) registered as the fallback if D1 cannot beat the P1/P2
baselines during 5D; D3 conv decoder rejected (duplicates encoder's
local modelling). All far below the ≤30 % budget.

## 10. Post-mixer context injection

**X1 additive: `q_{i,t} = z_{i,t} + P(h'_i)`** with P = Linear(192,192)
— the simplest of X1–X3 and sufficient: h'_i enters every masked
prediction of band i. X2/X3 add parameters without a design argument.

## 11. Proof that the mixer receives reconstruction gradients

Gradient path: `loss → pred → q_{i,t} = z_{i,t} + P(h'_i)`; the second
term is `h'_i = h_i + g_i ⊙ W_c c` with `c = Σ_j α_j V h̃_j` — mixer
parameters (score/V/gate/context) and ALL valid bands sit on the path
of every masked prediction. Verified empirically with a bounded dummy
backward on synthetic mixed-shape batches: max |grad| stem 0.0023,
coords 0.0078, temporal 0.0033, **mixer 0.0008 > 0**. Enforced
permanently by `test_masked_reconstruction_loss_trains_the_mixer`.
Leakage control also test-proven: perturbing values inside masked
patches changes NO prediction (embedding fully replaced) — only the
loss target changes.

## 12. Reconstruction target

Frozen: the original N2-normalized log1p magnitude 16×8 patches; valid
cells only (completion padding excluded from loss); no RGB/dB/waveform/
label targets; no renormalization.

## 13. Loss comparison (`loss_options.csv`)

**L1 masked valid-cell MSE recommended as both training loss and the
reported metric** (TA requirement). L2 SmoothL1 registered as fallback
only if 5D training curves show outlier instability (a training-signal
decision, not label-driven). L3 frequency-balanced MSE unnecessary: N2
already gives every bin unit TRAIN variance by construction.

## 14. Loss normalization / fairness

**Per-window mean over masked valid cells → mean over batch windows.**
This makes each window contribute equal weight regardless of native
bandwidth: HIT (257 bins, 245 masked patches) counts exactly as much as
a 513-bin window (455–475 masked patches). Global cell-mean was
rejected (bandwidth-proportional weighting); per-patch-mean adds
nothing over per-window given equal patch sizes. Implemented exactly
this way in the probe and test-verified.

## 15. Variable-bandwidth handling

The generator masks only valid patches (test: HIT-like validity — bands
≥17 never masked; exact counts on 759- and 408-patch grids). Structured
masks operate on each window's own valid grid, so HIT's fewer bands do
not change its masked *proportion*. Expected masked patches @60 %:
CWRU 455 · JNU/MaF 475 · HIT 245.

## 16. Deterministic masking design

Per-window generator: `Philox(sha256(global_seed | epoch | window_id)
[:8])`. Same seed → identical masks (tested); different epochs →
different masks (tested); different S1 seeds → independent reproducible
masks; no stored masks; window IDs come from the sealed Part-3B
manifests; no test data involved.

## 17. Trivial reconstruction baselines (`baseline_reconstruction_metrics.csv`)

Measured on the 66 TRAIN dev windows under the recommended M1@60 %
(per-dataset mean masked valid-cell MSE):

| Baseline | CWRU | JNU | HIT | MaFaulDa |
|---|---|---|---|---|
| P0 predict zero | 1.053 | 1.452 | 0.999 | 1.851 |
| P1 temporal-neighbour mean | 1.138 | 1.605 | 1.086 | 0.894 |
| P2 frequency-neighbour mean | 1.019 | 0.881 | 1.566 | 1.049 |

Reading: copy/interpolation baselines barely beat — and mostly lose
to — predicting zero; the best trivial predictor per dataset sits at
0.88–1.05. **A trained S1 must clearly undercut these numbers to claim
non-trivial reconstruction**; these rows become the descriptive
comparison in the future SSL report.

## 18. Dataset-balanced SSL sampler (specification only)

Label-free hierarchy `dataset → group → window` with approximately
equal dataset probability per batch; masking policy identical across
datasets; groups from the sealed Part-2/3B manifests. No sampler code
exists yet (Part 5D).

## 19. Compute feasibility (`ssl_compute_estimate.csv`)

Reference backend (honest upper bound; Python-loop scan timings are NOT
representative of fused kernels): steps/epoch ≈ 804/402 (bs 16/32);
est. epoch 3.6/4.8 min per fold; **60 epochs × 3 folds ≈ 10.9–14.3 h**;
forward memory ≤1.1 GB at bs 64 (training roughly 3–4× — well inside
20 GB). **Verdict: full SSL is practically feasible even without the
optimized kernels.**

## 20. Recommended final SSL configuration

`proposed_ssl_spec.yaml` — M1@60 % masking with E1 token, D1+X1
decoder (119.6k), N2-target masked valid-cell MSE with per-window
normalization, deterministic per-window masks, dataset-balanced
label-free sampler, validation-based early stopping/checkpoint
selection, decoder discarded after pretraining (S0 never sees it; the
downstream architecture hash excludes it), no
contrastive/language/pseudo-label objectives in primary S1.

## 21. Remaining risks

- Baselines/redundancy come from the 66-window dev subset; full-TRAIN
  values will shift somewhat (they are descriptive bars, not tuned
  parameters).
- If trained reconstruction fails to undercut P1/P2, the registered
  fallbacks are M2 masking and/or the D2 decoder — a 5D decision from
  training signals only.
- The reference backend makes wall-clock estimates upper bounds; kernel
  availability may change them substantially.
- Mask-ratio 60 % is a reasoned default; it was chosen from structure,
  not swept — a future sensitivity note, not a tuned value.

## 22. Decisions requiring human approval

1. Freeze the SSL spec (§20) — geometry M1@60 %, E1, D1+X1, loss and
   normalization policy.
2. Confirm validation-based pretraining early stopping / checkpoint
   selection (test stays sealed).
3. Confirm the sampler principle for implementation in Part 5D.
4. Authorize Part 5D: experiment registry + actual S0/S1 training.
5. Commit the Part-5C artifacts (and the parity-gate addendum already
   included in the Part-5B commit).

**HARD STOP.** No SSL pretraining, no supervised training, no few-shot
work, no samplers implemented. Awaiting approval before Part 5D.
