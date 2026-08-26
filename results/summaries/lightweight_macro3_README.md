# Part-6 Macro-3 reaggregation

The models were trained under the previously frozen four-domain Part-6 downstream protocol. Macro-3 is recalculated here from the already saved JNU, HIT, and MaFaulDa TEST metrics for consistency with the dissertation's three-dataset reporting. It is therefore an evaluation-only post-hoc reaggregation, not a retrained three-domain compression experiment.

## Completeness

All 27 authoritative reports passed identity and four-domain field checks. Macro-3 excludes CWRU.

## Nine cells

| Fold | Seed | Full S1 | K1 | Δ K1−S1 | Q8(K1) | Δ Q8K1−S1 | Δ Q8K1−K1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 42 | 0.890562 | 0.945387 | +0.054825 | 0.945294 | +0.054732 | -0.000093 |
| 1 | 1337 | 0.953758 | 0.959762 | +0.006005 | 0.959762 | +0.006005 | +0.000000 |
| 1 | 2026 | 0.921994 | 0.920008 | -0.001986 | 0.920008 | -0.001986 | +0.000000 |
| 2 | 42 | 0.933157 | 0.943690 | +0.010533 | 0.943431 | +0.010274 | -0.000259 |
| 2 | 1337 | 0.906860 | 0.916152 | +0.009292 | 0.916168 | +0.009308 | +0.000016 |
| 2 | 2026 | 0.910048 | 0.911868 | +0.001820 | 0.911967 | +0.001919 | +0.000099 |
| 3 | 42 | 0.915334 | 0.923532 | +0.008198 | 0.923668 | +0.008334 | +0.000136 |
| 3 | 1337 | 0.929120 | 0.965237 | +0.036117 | 0.965237 | +0.036117 | +0.000000 |
| 3 | 2026 | 0.918457 | 0.946577 | +0.028120 | 0.947564 | +0.029107 | +0.000987 |

## Analysis files

See `tables/` for machine-readable results and `provenance/` for sources, hashes, metric definition, and integrity checks.
