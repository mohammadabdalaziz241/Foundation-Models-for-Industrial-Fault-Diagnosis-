# methodology_v2 — Part 3B: frozen raw-signal windows

Deterministic 1-second raw-window identities for the three frozen global
folds (54,254 windows total; native rates; frozen channels; train 50 % /
eval 0 % overlap). Full protocol and statistics:
`PART3B_WINDOW_REPORT.md`.

Files:

- `window_manifest_fold_{1,2,3}.csv` — one row per window: full
  provenance (source file, group, split, channel, exact samples,
  fragment ids for HIT, metadata).
- `jnu_guards_1s.csv` — instantiated 1.0 s guards around the 48 sealed
  Part-2 anchors (Part 2 itself unmodified).
- `hit_logical_stream_manifest.csv` — complete reconstruction geometry
  of the 134 HIT logical streams (ordered fragments, offsets, lengths).
- `window_statistics.json` — counts, class/coverage tables, HIT
  boundary stats, Part-3A estimate comparison (all-zero differences),
  sampled signal-integrity results.
- `window_hashes.csv` — SHA-256 seals; master hash
  `99ffde7e5c0e2cb9b05713801aedcb10b11ccc229d4c2d10a58a1506db10bb51`.
- `part3b_reproducibility.json` — git/env/seed-free determinism record.

Read a window (raw values, no transformation):

```python
from src.methodology_v2.part3b_reader import read_window
import pandas as pd
row = pd.read_csv("methodology_v2/part3_windows/window_manifest_fold_1.csv").iloc[0]
x = read_window(row)   # exact native-rate float signal, 1.0 s
```

Before consuming these manifests, future stages MUST call both
`part2_builder.verify_frozen_hashes()` and
`part3b_windows.verify_part3b_hashes()` — both fail closed.

Regenerate (byte-identical):
`.venv/bin/python scripts/methodology_v2/run_part3b.py`

Status: **Part 3B complete — HARD STOP before Part 4** (no STFT, no
spectrograms, no normalization, no models, no training).
