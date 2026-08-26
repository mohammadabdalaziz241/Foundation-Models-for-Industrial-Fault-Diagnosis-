# methodology_v2 — redesigned dissertation experiment

Isolated namespace for the redesigned main experiment (self-supervised vs
supervised fault diagnosis over CWRU, JNU, HIT and MaFaulDa). Nothing in
here modifies or depends on the legacy experiment families; previous
results remain byte-reproducible.

Layout:

- `part1_audit/` — Part 1 deliverables (dataset census, recording-level
  manifest, grouping policy, candidate label mapping, integrity report,
  hashes, final audit report `PART1_DATASET_AUDIT_REPORT.md`).
- Code: `src/methodology_v2/` (audit-only package; imports no training or
  windowing code — enforced by tests).
- Runner: `scripts/methodology_v2/run_part1_audit.py`.
- Tests: `tests/methodology_v2/test_part1_audit.py`.

Status: **Part 1 complete — HARD STOP.** Part 2 (splits, preprocessing,
training) requires explicit approval of the open decisions in
`part1_audit/PART1_DATASET_AUDIT_REPORT.md` §7.

Raw data locations (all under gitignored `data/`): CWRU `data/raw` +
`data/raw_cwru_48k` (pre-existing), JNU `data/raw_jnu/JNU-Bearing-Dataset`
(git clone 75b33611), HIT `data/raw_hit/HIT-dataset` (git clone ef176559)
+ `data/raw_hit/gdrive_full/HIT-dataset` (official Google Drive full
release), MaFaulDa `data/raw_mafaulda/full` (official full.zip, sha256 in
`part1_audit/reproducibility.json`).

- `part6_compression/` — Part 6 (PC-STE lightweight study: Student-D
  distillation, Q8 PTQ, Stage-2 sensitivity, single sealed TEST session).
  Code `src/methodology_v2/compression/`, CLI
  `scripts/methodology_v2/part6_compression.py`, tests
  `tests/methodology_v2/test_part6_compression.py`, guide
  `docs/methodology_v2_part6_implementation.md`. Status 2026-08-18:
  implementation complete, TEMPLATE registry, final seal pending, no
  experiment started.
