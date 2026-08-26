# methodology_v2 — Part 4A: training-only STFT design study

Bounded representation-design study answering the TA's "not blurry"
concern. All signal diagnostics are **Fold-1 TRAIN only** (guard-enforced;
window IDs in `part4a_signal_access_log.json`); Part-2 and Part-3B seals
verified intact before/after. No full-dataset spectrograms, no resizing,
no models, no training. Full evidence: `PART4A_STFT_DESIGN_REPORT.md`.

Key artefacts:

- `part4a_development_windows.csv` — 66 deterministic dev windows.
- `stft_candidate_grid.csv` / `stft_resolution_table.csv` — 5 n_fft × 2
  hops × 4 native rates, exact physical resolutions.
- `stft_sharpness_metrics.csv`, `stft_dynamic_range_metrics.csv` —
  descriptive diagnostics (never a single optimised score).
- `hit_boundary_audit.csv` — 60 fragment joints: NO systematic artefact.
- `frequency_coordinate_study.csv` — same bin ≠ same Hz across rates.
- `stft_memory_estimates.csv` — representation memory per config/batch.
- `part4a_recommendations.yaml` — PENDING approval: 1024/256 Hann
  (center=False, no padding) · log1p · physical-Hz coordinates
  (Strategy C) · multi-resolution deferred (MAYBE) · normalization
  candidates for 4B.
- `figures/` — 22 deterministic TRAIN-only figures (class
  representatives, multi-resolution pairs, HIT joint audits, one
  clearly-labelled resize NEGATIVE CONTROL).

Transform convention (exact): periodic Hann, center=False, no padding,
one-sided `numpy.fft.rfft`, full frames only — implemented explicitly in
`src/methodology_v2/part4a_repdesign.py` (no library defaults).

Regenerate: `.venv/bin/python scripts/methodology_v2/run_part4a.py`
(numeric tables byte-deterministic; tested).

Status: **Part 4A complete — HARD STOP before Part 4B.**
