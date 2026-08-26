# PART 5B — PC-STE Encoder Implementation and Architecture Verification
## methodology_v2

Implementation + verification only: no SSL objective, no masking policy,
no decoder, no supervised heads, no optimizer, no training of any kind.
Contribution framing unchanged from Part 5A: a mechanism-level claim
(absolute-physical-coordinate conditioning + explicit cross-band
exchange over variable-bandwidth vibration representations) — no
first-of-kind claims; BiMamba and physical-frequency encoding are not
claimed novel. Tests: **143 passed** suite-wide (20 new Part-5B tests).

## 1. Executive summary

PC-STE is implemented exactly to the approved Part-5A design and passes
its full verification battery on real Part-4C representations:
**2,382,033 parameters** (Small tier), exact 16×8 patch geometry
(33/17/33 bands; 759/408/792 tokens), **exact mask invariance**
(Δ = 0.0 under 10⁴-magnitude junk in all padding), proven coordinate
sensitivity, and **proven explicit cross-band exchange** (band 0's
summary changes only *after* the mixer when only band 5 is perturbed).
One disclosed implementation decision: the official `mamba_ssm` CUDA
package cannot be built against torch 2.12.0+cu130, so the temporal
backbone uses a reference-faithful pure-PyTorch implementation of the
official Mamba-1 selective scan (architecture unchanged; §7).

## 2. Upstream methodology commit information

- Previous checkpoint: `5d1b1e02` (Parts 1–4B).
- **New checkpoint committed before Part 5B (authorized): `d9e47dbd`**
  ("methodology-v2: freeze final representation and architecture
  design") — 32 files, Part-4C + Part-5A only, no unrelated files, not
  pushed.

## 3. Frozen input contract

Part-2 (`527ccc1d…`), Part-3B (`99ffde7e…`), Part-4C (`ee9414e8…`)
seals verified before and after all Part-5B work — intact. Inputs are
the sealed N2 log1p STFT tensors (CWRU 513×184 · JNU 513×192 ·
HIT 257×192 · MaFaulDa 513×192, float32) with physical Hz/seconds
coordinates from the frozen reader. Nothing upstream modified.

## 4. Patch geometry (verified on real representations)

16 bins × 8 frames per patch. CWRU: 33 bands × 23 time patches = 759
tokens (patch = 750 Hz × 42.7 ms); JNU/MaFaulDa: 33 × 24 = 792
(781.25 Hz × 41.0 ms); HIT: 17 × 24 = 408 (781.25 Hz × 41.0 ms).
Frequency and time axes are handled distinctly throughout
(`patchify.py` dims −2/−1); band/time structure is never flattened away.

## 5. Padding and masking

Completion padding only: 15 frequency bins complete each dataset's last
band (2.84 % of the own-grid cells; HIT 5.51 %); time axis needs zero
padding (184 = 23×8, 192 = 24×8). In mixed batches HIT is mechanically
zero-padded to the batch shape but carries only 17 valid bands — its
real spectrum ends at 12.5 kHz and the maximum valid band centre is
12.5 kHz (verified); **no fabricated high-frequency content exists**.
Masks are functionally enforced: invalid cells are zeroed before the
stem, invalid tokens zeroed after embedding, invalid bands excluded
from softmax, pooling and outputs (§13).

## 6. Coordinate encoder (exact configuration)

Fourier features of ABSOLUTE physical centres: patch centres computed
from real cells only → f_centre in kHz, t_centre in seconds.
8 log-spaced wavelengths per coordinate (f: [0.1, 51.2] kHz;
t: [0.02, 2.56] s), sin+cos → 32 deterministic features →
`Linear(32, 192)` (6,336 params). Combination: `token = patch_embedding
+ coordinate_embedding`. No normalized f/Nyquist, no dataset-specific
index embeddings. φ(f) for the mixer = the 16 frequency-only features.

## 7. Temporal BiMamba (exact design + dependency disclosure)

4 pre-norm residual bidirectional layers, d=192:
`y = x + 0.5·(MambaFwd(LN(x)) + flip(MambaBwd(flip(LN(x)))))`, two
independently parameterized directional blocks per layer, final
LayerNorm; shared across all bands and datasets; per-band sequences of
23–24 tokens. Each directional block is the **official Mamba-1
parameterization** (in_proj 192→768; depthwise causal conv k=4; SiLU;
x_proj→(Δ rank 12, B, C, d_state 16); softplus Δ;
selective scan hₜ = e^{ΔA}hₜ₋₁ + ΔBxₜ, y = Chₜ + Dxₜ; gated SiLU output;
out_proj) — 251,760 params/block.

**DISCLOSED DEPENDENCY DECISION**: `mamba-ssm`/`causal-conv1d` cannot be
built against torch 2.12.0+cu130 (no wheel; source build fails — logged
in `part5b_reproducibility.json`). The blocks therefore execute the
official selective-scan recurrence in pure PyTorch (reference
semantics; unit-tested for causality and input-dependent selectivity).
This is not an RNN substitute: the parameterization and recurrence are
Mamba-1 exactly; only the fused kernel is absent, which is
computationally irrelevant at T=23–24. Swapping in the official kernel
later changes speed, not architecture. This decision is open to
human override.

As pre-registered: no asymptotic-efficiency novelty is claimed for
Mamba at these sequence lengths.

## 8. Cross-band exchange (the central mechanism, exact equations)

For band summaries h₁…h_F with absolute centres f₁…f_F and valid mask:

- **A** aⱼ = w₂ᵀ tanh(W₁[h̃ⱼ ; φ(fⱼ)]), h̃ = LN(h); α = softmax over
  VALID bands only (masked −∞; α zeroed on invalid bands).
- **B** c = Σⱼ αⱼ · V h̃ⱼ — a shared context carrying information from
  ALL valid bands.
- **C** gᵢ = sigmoid(G[h̃ᵢ ; φ(fᵢ) ; c]); h′ᵢ = hᵢ + gᵢ ⊙ W_c c,
  output masked to valid bands.

Information from other bands enters band i through c in both the gate
and the residual update; absolute frequency participates in scores and
gates; invalid bands cannot contribute; HIT simply presents 17 valid
bands. 164,737 params.

## 9. Global embedding

Validity-masked mean over updated band summaries → z_global ∈ ℝ¹⁹².
Verified: manual masked mean equals the module output. API:
`encode_global(...)` and `encode_tokens(...)` (the latter returns
tokens, band summaries pre/post mixer, masks and coordinates for
Part-5C reconstruction use). The forward consumes only tensor values,
physical coordinates and masks — no labels, no dataset-conditional
branches.

## 10. Parameter breakdown (exact, frozen by test)

| Component | Params |
|---|---|
| patch stem (Conv2d 1→8 k3 + Linear 1024→192) | 196,880 |
| coordinate encoder | 6,336 |
| temporal BiMamba ×4 | 2,014,080 |
| Hz-gated cross-band mixer | 164,737 |
| **total** | **2,382,033** |

Within the approved Small tier (1–3M). Deviation from the Part-5A
spreadsheet (~2.13M): +0.25M, caused by the real conv-stem projection
(1024→192 vs the 128·d estimate) and the mixer's gate/score terms —
reported, not hidden; no dimension manipulation performed.

## 11. Dataset-specific token counts

CWRU 759 (33 bands × 23) · JNU 792 (33 × 24) · HIT 408 (17 × 24) ·
MaFaulDa 792 (33 × 24); per-band sequence length 23–24.

## 12. Mixed-dataset batching

`collate_representations` pads to batch max (513, 192) with explicit
cell masks and zero-filled coordinate padding; a real CWRU+JNU+HIT+
MaFaulDa batch runs in one forward with band counts [33, 33, 17, 33]
and finite outputs (test-verified). No interpolation anywhere.

## 13. Mask invariance (measured)

Replacing every padded/invalid cell with ±10⁴ junk changes the global
embedding by **exactly 0.0** (bit-identical); junk written into HIT's
padded frequency rows leaves its embedding bit-identical. Enforced by
construction (zeroing at three stages + masked softmax + masked
pooling), not merely recorded.

## 14. Coordinate sensitivity (measured)

Shifting all absolute frequencies by +5 kHz with identical tensor
values changes the global embedding (max |Δ| = 0.59) — physical
coordinates demonstrably enter the computation. Patch centres verified
against hand-computed values; HIT max 12.5 kHz, CWRU 24 kHz,
JNU/MaFaulDa 25 kHz.

## 15. Cross-band dependency (measured proof)

Perturbing ONLY band 5: band 0's pre-mixer summary is **exactly
unchanged** (0.0) while its post-mixer summary changes (max |Δ| =
0.012) — the mixer is the sole, real exchange path. Junk in masked
bands changes nothing; variable band counts produce no NaNs.

## 16. Gradient-flow verification

One synthetic scalar loss + one backward: nonzero gradients reach the
patch stem (max 4.52), coordinate encoder (4.52), temporal backbone
(6.33) and mixer (5.04). No optimizer step, no loop.

## 17. Compute/memory audit (forward, no-grad, RTX 4000 Ada)

| batch | forward ms | peak forward mem |
|---|---|---|
| 8 | ~135 | 147 MB |
| 16 | ~90 | 277 MB |
| 32 | ~236 | 540 MB |
| 64 | ~527 | 1,060 MB |

Parameter memory ≈ 9.1 MB (float32). Latency is dominated by the
reference scan's Python loop (24 steps × 4 layers × 2 directions) —
acceptable for this scale and improvable by the official kernel later.
Training overhead is explicitly NOT extrapolated from these numbers.

## 18. Transformer ablation compatibility

`TransformerBackbone` (4 pre-norm blocks, 4 heads, MLP×4) implemented
as a pure drop-in behind the same interface — identical stem,
coordinates, mixer and pooling (verified). Total 2,147,793 params
(−234k, −9.8 % vs BiMamba). Registered as ablation A3; not trained,
not tuned.

## 19. S0/S1 architecture identity guarantee

One class, one frozen config: `PCSTEConfig().to_dict()` is the shared
architecture contract; seeded construction is bit-deterministic
(state-dict equality test). S0 and S1 will instantiate this identical
encoder; only pretraining history may differ. The architecture hash
(§20) is what future S0/S1 experiment registries must reference.

## 20. Architecture hash

`part5b_architecture_hash.txt`:
**`962bb1de520a941a1c3a67c63c97d28983783327f7e263a73c9ec0bad0b6711f`**
= SHA-256 over the canonical spec (config, equations, coordinate spec,
parameter breakdown, upstream Part-4C hash). Initialization weights are
deliberately excluded — the hash denotes architecture, not weights.
Reproducibility of the hash from the spec file is test-enforced.

## 21. Limitations (open statement)

- Physical coordinate encoding per se is not claimed novel (ECHO uses
  normalized-frequency encodings; coordinate MLP/Fourier features are
  established); the absolute-Hz variant is a mechanism distinction only.
- BiMamba is not claimed novel (SSAMBA/Audio Mamba precede us in audio).
- The Hz-gated cross-band exchange is a mechanism-level candidate
  contribution — not a first-of-kind claim.
- Per-band sequences are short (23–24), so any Mamba efficiency
  advantage is an empirical question for the pre-registered A3 ablation,
  not an assumption.
- Variable native bandwidth is intentionally retained; HIT contributes
  no content above 12.5 kHz by design.
- The reference-scan implementation trades speed for dependency safety
  (disclosed above).
- **The architecture has not demonstrated any classification
  performance — nothing here validates diagnostic utility.**

## 22. Items deferred to Part 5C

SSL masking policy and ratios; masked-reconstruction decoder and loss
(target already frozen: N2 tensor); S0 supervised head + training
protocol; S1 pretraining + fine-tuning; samplers (dataset-balanced
principle pre-approved); label-fraction/few-shot subsets; experiment
registry referencing the architecture hash; training schedules.

## ADDENDUM — Mamba reference parity gate (post-approval verification)

The required parity gate against the OFFICIAL `selective_scan_ref`
(state-spaces/mamba, pinned commit
`e9594ce1c732d97440f0332fdc43170a2294dbfa`, vendored verbatim under
`src/methodology_v2/encoder/third_party/`) **PASSED** on all 10 bounded
synthetic cases (batch 1-4, seq 8-24, d_inner 32-384, d_state 8-16,
input scales 0.1-3.0, with/without z-gating, delta bias + softplus,
variable B/C, D skip): **forward outputs BIT-EXACT (max abs err 0.0)**;
gradient deviations at most 1.95e-3 absolute on gradients of magnitude
in the thousands = max RELATIVE error 2.0e-7 (float32 accumulation-order
noise between the two autograd graph shapes); several gradients
bit-exact. Transparency note: the initially declared absolute-only
gradient tolerance (1e-4) failed on the extreme-scale cases and was
revised to `abs<=1e-4 OR rel<=1e-6` with the original failure recorded
in `mamba_reference_parity.json`. Full official BLOCK parity is not
constructible without the uninstallable mamba_ssm package (stated
limitation); the scan — the only non-standard computation — is verified,
and our block composes it with standard linear/conv layers. Equations
and spec unchanged; the architecture hash `962bb1de…` is unchanged.
`einops` 0.8.2 was added to the environment solely for the vendored
official code. The scan was refactored into
`ssm.selective_scan(...)` (identical math; all tests pass).

**HARD STOP.** Awaiting human approval before Part 5C.
