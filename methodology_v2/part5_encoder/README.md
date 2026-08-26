# methodology_v2 — Part 5B: PC-STE encoder (implemented + verified)

Physically Calibrated Spectro-Temporal Encoder, exactly per the approved
Part-5A design: 16x8 TF patches -> compact conv/linear stem -> absolute
physical-coordinate Fourier features -> shared per-band temporal BiMamba
(4 blocks, d=192; reference-faithful Mamba-1 selective scan — official
CUDA package unbuildable on torch 2.12+cu130, disclosed) -> Hz-gated
cross-band exchange -> validity-masked pooling. 2,382,033 parameters
(Small tier). Architecture hash:
962bb1de520a941a1c3a67c63c97d28983783327f7e263a73c9ec0bad0b6711f

Code: `src/methodology_v2/encoder/` (patchify, coords, ssm, mixer,
pcste, collate). Tests: `tests/methodology_v2/test_part5b_encoder.py`
(suite total 143 passing). Artifacts here: encoder spec, parameter
breakdown, verified patch/batching geometry, forward compute audit,
ablation registry (A1 coords / A2 mixer / A3 transformer / A4 N1),
architecture hash, reproducibility record.

Verified properties (measured, test-enforced): exact mask invariance;
coordinate sensitivity; explicit cross-band exchange (band i changes
only via the mixer when band j is perturbed); shared temporal weights;
bidirectionality; deterministic init/forward; mixed 4-dataset batching;
gradient flow to every component; S0/S1 identical-architecture contract.

Usage (future S0/S1 — NOT run here):

```python
from src.methodology_v2.encoder import PCSTE, collate_representations
model = PCSTE()                       # same class for S0 and S1
out = model(**collate_representations(items))
```

Status: **Part 5B complete — HARD STOP before Part 5C** (no SSL,
masking, decoder, heads, optimizer, or training).
