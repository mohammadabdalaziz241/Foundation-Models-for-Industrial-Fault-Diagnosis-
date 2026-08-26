# Executed methodology

The authoritative scientific record is the frozen material in `methodology_v2/` and implementation in `src/methodology_v2/`.

SSL sampled CWRU, JNU, HIT and MaFaulDa equally and selected checkpoints by four-dataset MacroDomain reconstruction MSE. Downstream S0/S1 sampled the same four datasets, constructed four heads, optimized the equal mean of per-dataset CE losses, selected maximum four-domain validation MacroDomainF1, then touched TEST once after sealing. Macro-3 is a later reporting aggregate over JNU, HIT and MaFaulDa.

K1 is the four-block unidirectional `half_4x1` student initialized from same-cell S1. It used same-fold three-seed S1 teacher ensembles, hard CE, temperature-scaled forward KL (`T=4`, `alpha=0.5`) and mixer-attention relational KL (weight 1.0). Q8 and packed CPU INT8 are distinct representations.
