# methodology_v2 — Part 5D: frozen experiment registry

The COMPLETE frozen S0/S1 experiment, sealed and NOT launched:
9 SSL runs + 90 downstream runs (2 arms x 5 label fractions x 3 folds x
3 seeds {42,1337,2026}), deterministic nested label subsets
(label_subsets/, 9 sealed manifests, 5c10c25c50c100 per class with
group-even round-robin), label-free SSL sampler (16x4=64), hierarchical
supervised sampler, one shared frozen AdamW recipe (3e-4, wd 0.05,
warm-up 5, cosine to 1e-6; SSL 60 epochs, downstream 50; steps/epoch
202/205/201 from FULL train counts for every fraction), checkpoint
criteria MacroDomainReconMSE (SSL, fixed validation masks) and
MacroDomainF1_val (downstream, tie -> earlier epoch), primary metric
MacroDomainF1_test, pre-registered paired sign-flip statistics.

MASTER HASH: a1470bbb2f3d9c324e1c2a1ce06f177a462f905a9b491a0dc422abcfc40fdbac
(part5d_hashes.csv; experiment.registry.verify_part5d_hash() fails
closed). Mechanical finding: effective batch 64 = micro 32 x grad-accum
2 on this GPU (exact-loss chunking; identical S0/S1). Honest compute:
full matrix ~30 GPU-days on the reference backend; primary subset ~8.5
days — launch scheduling is a pre-launch human decision.

Code: src/methodology_v2/experiment/ (heads, label_subsets, samplers,
metrics, trainers, registry). Smoke: smoke_test_report.json
(NOT_AN_EXPERIMENT; weights discarded). Full detail:
PART5D_EXPERIMENT_REGISTRY_REPORT.md.

Regenerate (byte-identical): .venv/bin/python scripts/methodology_v2/run_part5d.py

Status: **Part 5D complete — HARD STOP. Awaiting explicit authorization
to launch the frozen registry.**
