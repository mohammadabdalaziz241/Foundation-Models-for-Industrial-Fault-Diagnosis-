# Four-domain computational-efficiency audit

This directory contains the final inference-only efficiency rerun over CWRU,
JNU, HIT, and MaFaulDa. Each dataset receives equal weight. The benchmark uses
the same nine sealed fold/seed cells (folds 1–3; seeds 42, 1337, and 2026), 50
warm-ups and 500 timed batch-size-1 forwards per model/device/dataset/cell.
Nothing was trained and no TEST examples were opened.

The single session ran from `2026-08-28T19:31:56.533394781+01:00` to
`2026-08-28T19:51:08.237387074+01:00` (1151.703992293 seconds). It comprised
90,000 timed forwards and 9,000 warm-up forwards. CPU used four PyTorch intra-op
threads and one inter-op thread. GPU timings used synchronized CUDA events.

## Headline results

| Metric | Old 3-domain | New 4-domain | Change |
|---|---:|---:|---:|
| Full S1 FLOPs (GFLOP/window) | 2.952525099 | 3.058269056 | +0.105743957 |
| K1 FLOPs (GFLOP/window) | 1.621550379 | 1.679675264 | +0.058124885 |
| FLOPs reduction | 45.079201% | 45.077584% | -0.001617 pp |
| Full S1 CPU latency (ms) | 21.894279 | 22.239370 | +0.345090 |
| K1 CPU latency (ms) | 11.632264 | 11.729429 | +0.097166 |
| CPU speed-up | 1.882203× | 1.896032× | +0.013829× |
| CPU reduction | 46.870762% | 47.258265% | +0.387502 pp |
| Full S1 GPU latency (ms) | 7.797068 | 7.663848 | -0.133221 |
| K1 GPU latency (ms) | 4.220152 | 4.152942 | -0.067211 |
| GPU speed-up | 1.847580× | 1.845402× | -0.002178× |
| GPU reduction | 45.875137% | 45.811266% | -0.063871 pp |
| Q8(K1) CPU latency (ms) | 11.266842 | 11.391179 | +0.124336 |
| Q8 speed-up vs Full S1 | 1.943249× | 1.952333× | +0.009084× |
| Q8 reduction vs Full S1 | 48.539788% | 48.779220% | +0.239432 pp |

FLOPs were recalculated from the actual sealed Full S1 and final K1
(`half_4x1`) model definitions. The generic historical compression artifact's
`student_d` entry is not treated as final K1.

## Scan steps

| Dataset | Geometry | Full S1 | K1 | Reduction |
|---|---:|---:|---:|---:|
| CWRU | 513×184 | 184 | 92 | 50% |
| JNU | 513×192 | 192 | 96 | 50% |
| HIT | 257×192 | 192 | 96 | 50% |
| MaFaulDa | 513×192 | 192 | 96 | 50% |

## Parity and memory

The existing deterministic validation policy produced 576 windows
(4 datasets × 9 cells × 16). Maximum absolute packed-versus-simulated Q8 logit
deviation was 3.987194061279297 and mean cell-level prediction agreement was
99.13194444444444%. Predictions were not universally identical, consistent
with the original audit: packed dynamic activation quantization is a different
runtime representation from weight-only Q8 followed by FP32 compute. The audit
passed because all bindings, shapes, finite outputs, and original measurements
were valid; its tolerance/criterion was not weakened.

Peak allocated GPU memory is reported as a maximum, not an average. The
four-domain maxima were 33.5625 MiB for Full S1 and 28.564453125 MiB for K1.
Per-dataset values are in `per_dataset_efficiency.csv`.

Q8 GPU latency remains intentionally unreported: the repository has no genuine
packed INT8 CUDA implementation, and its registered CUDA Q8 path dequantizes to
FP32.

## Reproduction and provenance

- Run: `.venv/bin/python scripts/methodology_v2/benchmark_part6_latency_four_domain.py`
- Compact result: `four_domain_efficiency.json`
- Per-domain table: `per_dataset_efficiency.csv`
- Raw timing evidence: `latency_raw.csv`
- Cell summaries: `latency_by_cell.csv`

Full checkpoints, datasets, caches, and machine-specific absolute paths are not
redistributed in this compact result folder.

## Old-reference audit

The only exact old-efficiency references found outside this folder were the
historical three-domain benchmark implementation and its sealed output under
`results/methodology_v2/part6_compression/latency/`. They are intentionally
retained as provenance rather than rewritten. No dissertation LaTeX source is
present in this repository. Unrelated occurrences such as 46.875 Hz, other
experiments containing 432 runs/windows, and percentages near 33.43 were not
changed.
