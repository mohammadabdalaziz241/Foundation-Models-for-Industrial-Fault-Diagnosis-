# Repository structure

The public repository separates executable code, frozen scientific provenance, dissertation-facing configuration, and compact result evidence.

- `src/methodology_v2/` — executable PC-STE preprocessing, encoder, SSL, downstream and compression implementation.
- `scripts/methodology_v2/` — staged audit/build/train/compression entry points for PC-STE.
- `scripts/run_methodology_v2_1pct_extension.py`, `scripts/run_methodology_v2_5pct.py`, `scripts/run_methodology_v2_10pct.py` — recovered public execution paths for the reduced-label S0/S1 grids.
- `src/baselines/` — controlled four-domain InceptionTime baseline model, waveform access/sampling, objectives and metrics. The historical filename `three_domain.py` is retained for compatibility, but its active implementation is fail-closed to exactly CWRU, JNU, HIT and MaFaulDa and computes Macro-4.
- `scripts/baselines/` — InceptionTime training, launcher, aggregation and public evidence-analysis tools.
- `tests/methodology_v2/` — PC-STE pipeline tests.
- `tests/baselines/` — controlled-baseline tests.
- `methodology_v2/` — frozen scientific inputs, manifests, design records, registries and checksums; it is not a second implementation.
- `configs/dissertation/` — dissertation-facing PC-STE specifications, including the authoritative 1%/5%/10%/100% downstream fraction list.
- `results/tables/`, `results/figures/`, `results/summaries/` — curated PC-STE result outputs and historical summaries.
- `results/baselines/inceptiontime_four_domain/` — compact evidence for the completed controlled InceptionTime comparison: portable protocol, nine-cell results, paired comparisons, dataset summaries, aggregate statistics and training-cost provenance.
- `docs/` — methodology, reproducibility, reduced-label provenance and this repository map.
- `DATA.md` — dataset acquisition/layout information and integrity references.

Raw third-party datasets, trained checkpoints, teacher caches, prediction/probability caches, complete experiment run trees and machine-local logs are intentionally excluded. The files retained in the repository are the publication-safe implementation, frozen partition/preprocessing evidence, compact results and provenance needed to inspect or rerun the reported dissertation study.
