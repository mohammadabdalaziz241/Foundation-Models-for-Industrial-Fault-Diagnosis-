# Results

This directory contains the curated result tables, figures and summaries used
for the dissertation.

## Primary aggregate

The dissertation reports **Macro-4** as the principal cross-dataset metric:

**Macro-4 = (F1_CWRU + F1_JNU + F1_HIT + F1_MaFaulDa) / 4**

This matches the executed four-domain downstream protocol, in which CWRU, JNU,
HIT and MaFaulDa all contribute supervised loss through dataset-specific heads.

| Model | Macro-4 | Historical Macro-3 |
|---|---:|---:|
| Full S0 | 0.7711 | 0.9214 |
| Full S1 | 0.7708 | 0.9199 |
| K1 | 0.793125 | 0.936913 |
| Q8(K1) | 0.793382 | 0.937011 |

The historical **Macro-3** values average JNU, HIT and MaFaulDa only. They are
retained as secondary results from an earlier reporting stage and do not
represent three-domain training; both aggregates are computed from models
trained under the same executed four-domain protocol.

For detailed provenance, see [`PROVENANCE.md`](PROVENANCE.md).
Curated numerical outputs are under [`tables/`](tables/), figures under
[`figures/`](figures/), and compact reports under [`summaries/`](summaries/).
