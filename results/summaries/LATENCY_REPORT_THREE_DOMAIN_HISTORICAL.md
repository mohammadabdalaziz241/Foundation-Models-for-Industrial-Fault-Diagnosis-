# Historical three-domain Part-6 inference latency report

> **Historical provenance only.** Numerical content below is preserved unchanged. The final dissertation-facing four-domain summary is `LATENCY_REPORT.md`.

## Scope and protocol audit

Model forward-pass latency only was measured at batch size 1: encoder plus the
correct dataset-specific head through logits. Disk I/O, checkpoint loading,
dataset loading, STFT, N2 preprocessing, transfers, and serialization were
outside timing. There were 50 warm-ups and
500 individually timed forwards per cell. No significance
testing was performed on repeated timings.

Full details are in [lightweight_protocol_audit.md](lightweight_protocol_audit.md).
Full S1 and K1 were trained under the frozen four-domain CWRU+JNU+HIT+MaFaulDa
protocol. K1's CWRU windows contributed CE, KD, and relational loss. The final
dissertation Macro-3 values—and this equal-domain latency summary—exclude CWRU
post hoc and average JNU, HIT, and MaFaulDa equally.

Only deterministic N2-normalised **validation** representations were loaded.
Every selected ID was structurally checked as validation. **TEST data and sealed
TEST artifacts were not opened or modified by this benchmark.**

## Hardware and runtime

- Hostname: `worker1`
- CPU: Intel(R) Core(TM) i9-14900
- CPU cores: 24 physical / 32 logical
- PyTorch CPU threads: 4 intra-op / 1 inter-op
- Packed Q8 engine: `fbgemm`
- GPU: NVIDIA RTX 4000 Ada Generation
- CUDA: 13.0
- PyTorch: 2.12.0+cu130
- Python: 3.12.3
- Git commit: `df34c8cba1aa774b3ca2d9ff4d928cd34e9a7a48`

FP32 execution uses the repository's `build_encoder`/`build_heads`, strict
checkpoint loads, `PCSTE.forward`, and `DatasetHeads.forward`. No AMP/autocast
was used. Packed Q8 CPU uses the registered `cpu_dynamic_quantize` path on both
encoder and heads. Exact per-cell paths and hashes are recorded in metadata.

## Q8 runtime/parity audit

The dissertation predictive representation is per-output-channel weight-only
Q8 followed by FP32 compute. The packed deployment model additionally performs
dynamic activation quantization, so it is neither assumed nor described as
numerically identical. Comparison against the dissertation representation:

| Dataset | N validation windows | Prediction agreement % | Max absolute logit delta | Mean absolute logit delta |
|---|---|---|---|---|
| JNU | 144 | 100.000000 | 2.877818 | 0.463524 |
| HIT | 144 | 100.000000 | 2.219269 | 0.356789 |
| MAFAULDA | 144 | 99.305556 | 2.771443 | 0.292651 |

The 1.600486 MB result is the compact serializable tensor representation for
the f1/s42 example; it is not the packed `torch.ao` runtime state size.

## CPU latency by dataset

Values are means across the nine matched fold-seed cell means; SD is across
those nine model cells.

| Model | Dataset | Mean ms/window | Cell SD ms |
|---|---|---|---|
| Full S1 | JNU | 25.0901 | 0.7695 |
| Full S1 | HIT | 15.5414 | 0.3806 |
| Full S1 | MAFAULDA | 25.0513 | 0.7623 |
| K1 | JNU | 13.3031 | 0.3673 |
| K1 | HIT | 8.2874 | 0.2116 |
| K1 | MAFAULDA | 13.3063 | 0.3816 |
| Q8(K1) | JNU | 12.7197 | 0.3088 |
| Q8(K1) | HIT | 8.2538 | 0.1902 |
| Q8(K1) | MAFAULDA | 12.8270 | 0.0557 |

## Equal-domain CPU results and speedups

| Model | Equal-domain ms | Cell SD ms | windows/s | Speedup vs Full | Reduction vs Full % | Speedup vs K1 | Reduction vs K1 % |
|---|---|---|---|---|---|---|---|
| Full S1 | 21.8943 | 0.6329 | 45.6740 | 1.0000 | 0.0000 | N/A | N/A |
| K1 | 11.6323 | 0.3192 | 85.9678 | 1.8822 | 46.8708 | N/A | N/A |
| Q8(K1) | 11.2668 | 0.1675 | 88.7560 | 1.9432 | 48.5398 | 1.0324 | 3.1414 |

## GPU latency by dataset

| Model | Dataset | Mean ms/window | Cell SD ms | Peak allocated MiB |
|---|---|---|---|---|
| Full S1 | JNU | 7.8121 | 0.0236 | 33.4336 |
| Full S1 | HIT | 7.7830 | 0.0281 | 25.5864 |
| Full S1 | MAFAULDA | 7.7962 | 0.0283 | 33.4336 |
| K1 | JNU | 4.2262 | 0.0151 | 28.4355 |
| K1 | HIT | 4.2240 | 0.0176 | 21.1509 |
| K1 | MAFAULDA | 4.2103 | 0.0168 | 28.4355 |

## Equal-domain GPU results

| Model | Device | JNU ms | HIT ms | MaFaulDa ms | Equal-domain ms | Speedup vs Full | Reduction vs Full % |
|---|---|---|---|---|---|---|---|
| Full S1 | cuda | 7.8121 | 7.7830 | 7.7962 | 7.7971 | 1.0000 | 0.0000 |
| K1 | cuda | 4.2262 | 4.2240 | 4.2103 | 4.2202 | 1.8476 | 45.8751 |
| Q8(K1) | cuda | N/A | N/A | N/A | N/A | N/A | N/A |

Peak CUDA allocated memory is reported per dataset/cell in
`latency_by_cell.csv`; the table above reports the maximum over nine cells.

**A genuine INT8 Q8(K1) GPU latency is not reported because the current
implementation has no packed INT8 CUDA execution path.**

## Warnings and interpretation

- K1/Q8 are four-domain-trained artifacts; Macro-3 is a post-hoc reporting scope.
- Packed Q8 CPU and the compact predictive Q8 representation are distinct, as
  quantified by the validation parity audit.
- The pure-PyTorch reference selective scan is the active model backend.
- Timing iterations estimate runtime variability and are not independent
  experimental replicates. A later paired comparison should use matched
  fold-seed/domain model cells.
