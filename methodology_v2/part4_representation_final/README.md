# methodology_v2 — Part 4C: final sealed representation

The frozen model-input contract for Part 5, fully implemented:
sealed Part-3B raw windows → dataset-specific physically matched STFT
(CWRU/JNU/MaFaulDa 1024/256, HIT 512/128; periodic Hann, center=False,
no padding, one-sided) → log1p magnitude → **N2 normalization**
(per fold × dataset × frequency-bin TRAIN statistics) → float32
(freq_bins, time_frames) tensor with physical Hz/seconds coordinates.

Contents:

- `representation_spec.yaml` — the FROZEN specification (authoritative).
- `normalizers/fold_{1,2,3}/{cwru,jnu,hit,mafaulda}.npz` — the 12
  sealed normalizers (byte-deterministic archives: mean, std_raw,
  std_denominator, frequency_hz, floored_bins, counts, STFT config).
  Note: `*.npz` is gitignored — the committed authority is
  `normalizer_hashes.csv`; artifacts regenerate byte-identically via
  the runner.
- `normalizer_registry.csv` — human-readable metadata incl. TRAIN
  manifest hash and spec hash per normalizer.
- `normalizer_fit_statistics.csv`, `normalization_sanity.json`,
  `valtest_mechanical_checks.json`, `representation_shapes.csv`.
- `normalizer_hashes.csv` — per-artifact SHA-256 +
  PART4C_MASTER_REPRESENTATION_HASH.
- `PART4C_FINAL_REPRESENTATION_REPORT.md`, `part4c_reproducibility.json`.

Use from Part 5:

```python
from src.methodology_v2.part4c_reader import get_representation
tensor, meta = get_representation(window_id, fold_id)
# tensor: float32 (freq_bins, time_frames), N2-normalized
# meta: frequency_hz, time_seconds, split, provenance, ...
```

The reader verifies the Part-2, Part-3B and Part-4C seals fail-closed
before serving anything; validation/test access uses the sealed TRAIN
statistics and never updates them (test-proven). Lazy generation only —
no precomputed spectrogram files exist; the manifests + normalizers
remain the sole source of truth.

Frozen SSL reconstruction target: this N2-normalized log1p spectrogram.
Pre-registered ablation: N1 (per-window standardisation) — documented,
not fitted.

Regenerate: `.venv/bin/python scripts/methodology_v2/run_part4c.py`

Status: **Part 4C complete — HARD STOP before Part 5** (no patching,
batching, embeddings, encoder, masking, SSL, or training work).
