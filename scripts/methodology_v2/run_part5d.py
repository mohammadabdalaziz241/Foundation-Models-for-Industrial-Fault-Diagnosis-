#!/usr/bin/env python
"""methodology_v2 Part 5D — build and seal the frozen experiment
registry; bounded smoke tests only (NOT_AN_EXPERIMENT; weights
discarded). Does NOT launch the 99-run matrix.
"""
from __future__ import annotations

import datetime as dt
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.methodology_v2.integrity import sha256_file  # noqa: E402
from src.methodology_v2.part2_builder import verify_frozen_hashes  # noqa: E402
from src.methodology_v2.part3b_windows import (PART3B_DIR,  # noqa: E402
                                               verify_part3b_hashes)
from src.methodology_v2.part4c_normalizers import (  # noqa: E402
    verify_part4c_hashes)
from src.methodology_v2.part4c_reader import get_representation  # noqa: E402
from src.methodology_v2.experiment.heads import (CLASS_ORDERS,  # noqa: E402
                                                 head_seed)
from src.methodology_v2.experiment.label_subsets import (  # noqa: E402
    FOLDS, FRACTIONS, SEEDS, build_subset_table, realised_fractions)
from src.methodology_v2.experiment.registry import (  # noqa: E402
    PART5D_DIR, SUBSET_DIR, UPSTREAM, build_registries, master_hash,
    part5c_spec_hash, train_counts)
from src.methodology_v2.experiment.samplers import (  # noqa: E402
    SSLSampler, SupervisedSampler)
from src.methodology_v2.experiment.trainers import (  # noqa: E402
    DOWNSTREAM_EPOCHS, EFFECTIVE_BATCH, GRIDS, OPTIMIZER_SPEC, SSL_EPOCHS,
    SSLTrainer, SupervisedTrainer, steps_per_epoch)
from src.methodology_v2.registry import REPO_ROOT  # noqa: E402


def load_reps(fold: int, ids: list[str]):
    man = pd.read_csv(PART3B_DIR / f"window_manifest_fold_{fold}.csv"
                      ).set_index("window_id")
    reps, metas = [], []
    for wid in ids:
        row = man.loc[wid]
        assert row["split"] == "train", "smoke uses TRAIN only"
        x, meta = get_representation(wid, fold)
        reps.append((x, meta["frequency_hz"], meta["time_seconds"]))
        metas.append((row["dataset"], wid))
    return reps, metas


def smoke_tests(subset_f1_s42: pd.DataFrame) -> dict:
    """Bounded TRAIN-only smoke: micro-batch feasibility probe first,
    then 2 SSL steps, 2 supervised steps, S1 loading. All weights
    discarded afterwards."""
    report = {"label": "NOT_AN_EXPERIMENT — smoke verification only; "
                       "weights discarded; no metrics reported"}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    man = pd.read_csv(PART3B_DIR / "window_manifest_fold_1.csv")
    tr = man[man["split"] == "train"]
    ssl_view = tr[["dataset", "group_id", "window_id"]].copy()
    sampler = SSLSampler(ssl_view, seed=42)

    # ---- feasibility probe: effective 64 with micro 64 -> 32 -> 16 -------
    probe_batch = sampler.next_batch()
    reps, metas = load_reps(1, [w for _, w in probe_batch])
    feasible, probe = None, {}
    for micro in (64, 32, 16):
        t = SSLTrainer(seed=42, device=dev)
        try:
            if dev == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            t.train_step(reps, metas, epoch=0, micro_batch=micro)
            step_ms = 1000 * (time.perf_counter() - t0)
            feasible = micro
            probe = {"micro_batch": micro,
                     "gradient_accumulation": EFFECTIVE_BATCH // micro,
                     "train_step_ms": round(step_ms, 1),
                     "peak_mem_mb": (round(
                         torch.cuda.max_memory_allocated() / 2**20, 1)
                         if dev == "cuda" else None)}
            del t
            break
        except torch.OutOfMemoryError:
            del t
            if dev == "cuda":
                torch.cuda.empty_cache()
            probe[f"oom_at_micro_{micro}"] = True
    if feasible is None:
        raise AssertionError("no feasible micro-batch found — STOP")
    report["batch64_feasibility"] = {
        "device": dev, "effective_batch": EFFECTIVE_BATCH,
        **probe,
        "note": "mechanical gradient accumulation preserves effective "
                "batch 64 and the exact batch loss (dataset-aligned "
                "chunks); identical for S0 and S1",
    }

    # ---- SSL smoke (2 steps at the feasible micro-batch) ------------------
    trainer = SSLTrainer(seed=42, device=dev)
    losses = []
    for step in range(2):
        batch = sampler.next_batch()
        reps, metas = load_reps(1, [w for _, w in batch])
        losses.append(trainer.train_step(reps, metas, epoch=0,
                                         micro_batch=feasible))
    mixer_g = max(p.grad.abs().max().item()
                  for p in trainer.encoder.mixer.parameters()
                  if p.grad is not None)
    report["ssl_smoke"] = {"steps": 2, "losses": [round(x, 4)
                                                  for x in losses],
                           "micro_batch": feasible,
                           "mixer_grad_max": round(mixer_g, 6),
                           "mixer_grad_nonzero": mixer_g > 0}
    smoke_encoder_state = trainer.encoder_state()
    del trainer
    if dev == "cuda":
        torch.cuda.empty_cache()

    # ---- S0 supervised smoke ----------------------------------------------
    sup = SupervisedSampler(subset_f1_s42, fraction=1.0, seed=42)
    s0 = SupervisedTrainer(seed=42, head_init_seed=head_seed(1, 42),
                           device=dev)
    s0_heads_init = {k: v.detach().cpu().clone()
                     for k, v in s0.heads.state_dict().items()}
    sup_losses = []
    for step in range(2):
        batch = sup.next_batch()
        reps, _ = load_reps(1, [w for _, _, w in batch])
        sup_losses.append(s0.train_step(reps, batch,
                                        micro_batch=feasible))
    head_grads = {ds: any(p.grad is not None and p.grad.abs().sum() > 0
                          for p in s0.heads.heads[ds].parameters())
                  for ds in CLASS_ORDERS}
    report["s0_smoke"] = {"steps": 2,
                          "losses": [round(x, 4) for x in sup_losses],
                          "micro_batch": feasible,
                          "all_heads_received_gradients":
                              all(head_grads.values())}

    # ---- S1 loading smoke ---------------------------------------------------
    s1 = SupervisedTrainer(seed=42, head_init_seed=head_seed(1, 42),
                           encoder_state=smoke_encoder_state, device=dev)
    loaded_ok = all(torch.equal(a.cpu(), smoke_encoder_state[k])
                    for k, a in s1.encoder.state_dict().items())
    # compare INITIALIZATIONS (s0's snapshot before its smoke steps)
    heads_identical = all(
        torch.equal(s0_heads_init[k], b.detach().cpu())
        for k, b in s1.heads.state_dict().items())
    batch = sup.next_batch()
    reps, _ = load_reps(1, [w for _, _, w in batch])
    l1 = s1.train_step(reps, batch, micro_batch=feasible)
    report["s1_loading_smoke"] = {
        "encoder_state_loaded": loaded_ok,
        "paired_heads_identical_init": heads_identical,
        "supervised_step_after_load": round(l1, 4)}

    del s0, s1, smoke_encoder_state            # discard all smoke weights
    if dev == "cuda":
        torch.cuda.empty_cache()
    return report


def main() -> None:
    start = dt.datetime.now(dt.timezone.utc)
    verify_frozen_hashes(); verify_part3b_hashes(); verify_part4c_hashes()
    spec5c = part5c_spec_hash()
    print(f"upstream seals verified; part5c spec {spec5c[:12]}…")
    PART5D_DIR.mkdir(parents=True, exist_ok=True)
    SUBSET_DIR.mkdir(parents=True, exist_ok=True)

    # ---- nested label subsets (9 manifests) ------------------------------
    subset_hashes = {}
    realised_all = []
    subset_f1_s42 = None
    for f in FOLDS:
        parent_hash = sha256_file(
            PART3B_DIR / f"window_manifest_fold_{f}.csv")
        for s in SEEDS:
            df = build_subset_table(f, s)
            df["parent_train_manifest_sha256"] = parent_hash
            path = SUBSET_DIR / f"label_subset_f{f}_s{s}.csv"
            df.to_csv(path, index=False)
            subset_hashes[(f, s)] = sha256_file(path)
            r = realised_fractions(df)
            r.insert(0, "fold", f); r.insert(1, "seed", s)
            realised_all.append(r)
            if (f, s) == (1, 42):
                subset_f1_s42 = df
            print(f"  subset f{f} s{s}: {len(df)} train windows, "
                  f"hash {subset_hashes[(f, s)][:10]}…")
    pd.concat(realised_all).to_csv(
        PART5D_DIR / "label_fraction_registry.csv", index=False)
    pd.DataFrame([{"fold": f, "seed": s, "file":
                   f"label_subsets/label_subset_f{f}_s{s}.csv",
                   "sha256": h}
                  for (f, s), h in sorted(subset_hashes.items())]) \
        .to_csv(PART5D_DIR / "label_subset_hashes.csv", index=False)

    # ---- registries -------------------------------------------------------
    ssl_reg, main_reg = build_registries(subset_hashes)
    ssl_reg.to_csv(PART5D_DIR / "ssl_run_registry.csv", index=False)
    main_reg.to_csv(PART5D_DIR / "main_run_registry.csv", index=False)
    counts = train_counts()
    print(f"registries: {len(ssl_reg)} SSL + {len(main_reg)} downstream "
          f"runs; steps/epoch {dict((f, steps_per_epoch(c)) for f, c in counts.items())}")

    # ---- spec files --------------------------------------------------------
    specs = {
        "experiment_protocol.yaml": {
            "primary_hypothesis": "S1 improves MacroDomainF1_test over "
                "S0 at 100% labels",
            "primary_comparison": "S1@100% vs S0@100% on "
                                  "MacroDomainF1_test",
            "secondary": "label efficiency at 5/10/25/50%",
            "exploratory": "ablations A1-A4 (full-label only)",
            "arms": {"S0": "random PC-STE encoder + dataset heads",
                     "S1": "SSL-pretrained PC-STE encoder (best "
                           "MacroDomainReconMSE checkpoint) + SAME heads"},
            "seeds": list(SEEDS), "folds": list(FOLDS),
            "label_fractions": list(FRACTIONS),
            "run_matrix": {"ssl": 9, "downstream": 90, "total": 99},
            "test_access": "train -> validation checkpoint -> freeze -> "
                           "single TEST evaluation; no test-driven "
                           "decisions; failed runs restart only with "
                           "identical registry config",
            "run_states": ["REGISTERED", "RUNNING", "COMPLETE", "FAILED"],
            "upstream": UPSTREAM, "part5c_spec_hash": spec5c,
        },
        "optimizer_schedule_spec.yaml": {
            "shared_recipe": OPTIMIZER_SPEC,
            "ssl": {"max_epochs": SSL_EPOCHS, "no_early_stopping": True},
            "downstream": {"max_epochs": DOWNSTREAM_EPOCHS,
                           "no_early_stopping": True,
                           "single_lr_encoder_and_heads": True,
                           "identical_for_S0_and_S1": True},
            "steps_per_epoch_rule": "ceil(FULL fold TRAIN windows / 64) "
                "for SSL and for EVERY label fraction (fixed compute; "
                "low fractions revisit labelled windows via replacement)",
            "steps_per_epoch": {str(f): steps_per_epoch(c)
                                for f, c in counts.items()},
        },
        "head_registry.yaml": {
            ds: {"layer": "Linear(192,%d)" % len(c),
                 "classes_in_order": list(c)}
            for ds, c in CLASS_ORDERS.items()},
        "sampler_spec.yaml": {
            "ssl": {"hierarchy": "dataset->group->window", "labels": None,
                    "dataset_prob": 0.25, "batch": "16x4=64",
                    "replacement": True},
            "supervised": {"hierarchy": "dataset->class->group->window",
                           "source": "selected labelled subset only",
                           "class_balance": "deterministic class cycling",
                           "batch": "16x4=64", "replacement": True},
        },
        "metric_spec.yaml": {
            "primary_test_metric": "MacroDomainF1_test = mean of 4 "
                                   "dataset Macro-F1",
            "validation_checkpoint": "MacroDomainF1_val (max; tie -> "
                                     "earlier epoch)",
            "ssl_checkpoint": "MacroDomainReconMSE (min; per-dataset "
                              "window-mean masked-cell MSE)",
            "per_dataset": ["accuracy", "macro_f1", "per_class_precision",
                            "per_class_recall", "per_class_f1",
                            "confusion_matrix"],
            "zero_division": "0.0 explicit; absent classes never dropped",
            "class_orders": {ds: list(c)
                             for ds, c in CLASS_ORDERS.items()},
        },
        "statistical_analysis_spec.yaml": {
            "paired_unit": "fold x seed (9 pairs per fraction)",
            "delta": "MacroDomainF1(S1) - MacroDomainF1(S0)",
            "report": ["mean S0", "mean S1", "mean delta", "median delta",
                       "sd delta", "all 9 paired deltas", "effect size"],
            "primary_test": "exact two-sided paired sign-flip permutation "
                            "test (2^9 = 512 flips) at 100% labels",
            "secondary": "same test at 5/10/25/50% with Holm correction "
                         "if inferential p-values are reported across "
                         "fractions",
            "caveat": "fold x seed cells are paired but not fully "
                      "independent; conclusions stay cautious",
        },
        "ablation_registry.yaml": {
            "A1": {"change": "absolute coords -> index PE",
                   "question": "does physical calibration help",
                   "config": "PCSTEConfig(use_coordinates=False)"},
            "A2": {"change": "Hz-gated exchange -> masked mean fusion",
                   "question": "does cross-band exchange help",
                   "config": "PCSTEConfig(use_mixer=False)"},
            "A3": {"change": "BiMamba -> comparable Transformer",
                   "question": "backbone-specific?",
                   "config": "PCSTEConfig(backbone='transformer')"},
            "A4": {"change": "N2 -> N1 normalization",
                   "question": "does amplitude preservation matter",
                   "layer": "representation"},
            "policy": "full-label condition only; NOT launched in 5D",
        },
    }
    for name, content in specs.items():
        with open(PART5D_DIR / name, "w") as fh:
            json.dump(content, fh, indent=1, sort_keys=True)

    # ---- smoke tests -------------------------------------------------------
    smoke = smoke_tests(subset_f1_s42)
    with open(PART5D_DIR / "smoke_test_report.json", "w") as fh:
        json.dump(smoke, fh, indent=1)
    print("smoke:", {k: v for k, v in smoke.items() if k != "label"})

    # ---- master freeze hash -------------------------------------------------
    sealed = ([PART5D_DIR / n for n in
               ["experiment_protocol.yaml", "optimizer_schedule_spec.yaml",
                "head_registry.yaml", "sampler_spec.yaml",
                "metric_spec.yaml", "statistical_analysis_spec.yaml",
                "ablation_registry.yaml", "main_run_registry.csv",
                "ssl_run_registry.csv", "label_subset_hashes.csv",
                "label_fraction_registry.csv"]]
              + sorted(SUBSET_DIR.glob("*.csv")))
    rows = [{"file": str(p.relative_to(PART5D_DIR)),
             "sha256": sha256_file(p)} for p in sealed]
    m = master_hash(sealed)
    rows.append({"file": "PART5D_MASTER_HASH", "sha256": m})
    pd.DataFrame(rows).to_csv(PART5D_DIR / "part5d_hashes.csv",
                              index=False)
    print(f"PART5D master hash: {m}")

    def git(*a):
        return subprocess.run(["git", *a], cwd=REPO_ROOT, text=True,
                              capture_output=True).stdout.strip()
    repro = {
        "stage": "methodology_v2 Part 5D experiment registry freeze",
        "started_utc": start.isoformat(),
        "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "python": sys.version, "platform": platform.platform(),
        "torch": torch.__version__,
        "git": {"commit": git("rev-parse", "HEAD"),
                "branch": git("rev-parse", "--abbrev-ref", "HEAD")},
        "part5c_checkpoint_commit":
            "c54c702de41f0c15f26117e1e57a31c31567abe2",
        "upstream": UPSTREAM, "part5c_spec_hash": spec5c,
        "part5d_master_hash": m,
        "smoke_label": "NOT_AN_EXPERIMENT",
    }
    with open(PART5D_DIR / "part5d_reproducibility.json", "w") as fh:
        json.dump(repro, fh, indent=1)

    verify_frozen_hashes(); verify_part3b_hashes(); verify_part4c_hashes()
    from src.methodology_v2.experiment.registry import verify_part5d_hash
    verify_part5d_hash()
    print("all seals verified (post); Part 5D registry frozen")


if __name__ == "__main__":
    main()
