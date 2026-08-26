# methodology_v2 — Part 2: frozen split protocol

Three sealed global data folds (identity/region assignment only — no
windows, no preprocessing, no training artefacts). Full protocol and
statistics: `PART2_SPLIT_REPORT.md`.

Files:

- `global_fold_{1,2,3}.csv` — one row per assigned unit (CWRU recording,
  JNU macro-block/guard, HIT group, MaFaulDa recording); 2,245 rows each.
- `test_identity_fold_{1,2,3}.csv` — sealed test-partition identity.
- `split_protocol.json` — frozen rules, rotations, seeds, acceptance
  criteria, S0/S1 usage rules.
- `split_hashes.csv` — SHA-256 seals incl. MASTER_PROTOCOL_HASH.
- `rejected_split_seeds.json` — seed governance record (empty: all
  predeclared seeds passed).
- `fold_statistics.json` — per fold/dataset/split/class statistics.
- `split_reproducibility.json` — git, environment, Part-1 hashes, seeds.

Regenerate (byte-identical): 
`.venv/bin/python scripts/methodology_v2/run_part2_splits.py`

Consume safely: call
`src.methodology_v2.part2_builder.verify_frozen_hashes()` first — it
fails closed if any sealed manifest changed.

Usage contract (binding for S0/S1 and later stages): train = optimization
and SSL signals; validation = model selection only; test = sealed until
final evaluation; identical folds for every compared arm. JNU guard
regions are symbolic until Part 3 freezes the effective window span
(guard G ≥ span; boundary b expands to [b−ceil(G/2), b+ceil(G/2))).

Status: **Part 2 complete — HARD STOP before Part 3.**
