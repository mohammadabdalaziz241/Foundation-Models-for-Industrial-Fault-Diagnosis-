# methodology_v2 — Part 4B: representation verification & freeze study

Verifies the approved physically matched STFT (CWRU/JNU/MaFaulDa
1024/256, HIT 512/128 → ~20.5 ms frames, ~5.1 ms hops, ~48.8 Hz bins at
native rates; shapes (513,184)/(513,192)/(257,192)/(513,192)) and
compares the two remaining normalization strategies on **Fold-1 TRAIN
content only** (645-window stats sample + 66 qualitative windows; access
log tested). Full evidence: `PART4B_REPRESENTATION_REPORT.md`.

Recommendation (PENDING approval, `part4b_recommendations.yaml`):
**N2 — per-dataset, per-frequency-bin TRAIN normalization, fitted
independently per fold** — retains window-energy information (rank corr
0.96–1.00; N1 erases it exactly), removes cross-dataset scale shortcuts,
gives the richer masked-reconstruction target; dataset-identity
disclosure documented. N1 remains the identity-free fallback/ablation.

Proposed frozen input contract for Part 5:
`proposed_representation_spec.yaml` (native rates · frozen channels ·
matched Hann STFT center=False/no padding/one-sided · log1p · physical
Hz coordinates · no resizing/RGB · normalization pending).

Not done here (by design): no full-dataset STFT generation, no final
normalizer fitting, no models/training. Part-2 (`527ccc1d…`) and Part-3B
(`99ffde7e…`) seals verified before/after — untouched.

Regenerate: `.venv/bin/python scripts/methodology_v2/run_part4b.py`

Status: **Part 4B complete — HARD STOP before normalizer fitting /
Part 5.** Parts 1–4B remain uncommitted; checkpoint commit recommended.
