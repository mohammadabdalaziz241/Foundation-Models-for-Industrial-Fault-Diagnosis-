#!/usr/bin/env python
"""Isolated launcher and strict pre-launch audit for the REGISTERED Methodology V2 5% experiment.

The 5% fraction and all 18 S0/S1 runs are present in the sealed Part-5D registry.  It imports the
frozen model/trainer/sampler/reader and the authorized executor's mechanical
helpers, while writing only below results/methodology_v2_5pct.
The registered 5% selection uses the already-frozen class_rank ordering and the
same ceil/minimum-one rule as the registered fractions.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.methodology_v2.encoder import PCSTE, collate_representations
from src.methodology_v2.experiment.heads import (CLASS_ORDERS, DatasetHeads,
                                                 head_seed)
from src.methodology_v2.experiment.label_subsets import (FOLDS, SEEDS,
                                                         build_subset_table)
from src.methodology_v2.experiment.registry import verify_part5d_hash
from src.methodology_v2.experiment.samplers import SupervisedSampler
from src.methodology_v2.experiment.trainers import (DOWNSTREAM_EPOCHS,
                                                    EFFECTIVE_BATCH,
                                                    OPTIMIZER_SPEC,
                                                    SupervisedTrainer,
                                                    lr_lambda,
                                                    steps_per_epoch)
from src.methodology_v2.integrity import sha256_file
from src.methodology_v2.part2_builder import verify_frozen_hashes
from src.methodology_v2.part3b_reader import read_window
from src.methodology_v2.part3b_windows import PART3B_DIR, verify_part3b_hashes
from src.methodology_v2.part4c_normalizers import verify_part4c_hashes
from src.methodology_v2.part4c_reader import get_representation

_spec = importlib.util.spec_from_file_location(
    "methodology_v2_primary_executor",
    REPO / "scripts/methodology_v2/experiment_executor.py")
ex = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(ex)

NOMINAL_FRACTION = 0.05
FRACTION_COLUMN = "frac_5"
RESULTS_ROOT = Path(os.environ.get("PCSTE_RESULTS_ROOT", REPO / "results")).expanduser()
RESULTS = RESULTS_ROOT / "methodology_v2_5pct"
LOGS = REPO / "logs/methodology_v2_5pct"
MANIFESTS = RESULTS / "manifests"
AUDIT = RESULTS / "prelaunch_audit"
RUNS = RESULTS / "downstream"
MICRO_BATCH = ex.MICRO_BATCH
DATASETS = ex.DATASETS
EXPECTED_SHAPES = {"CWRU": (513, 184), "JNU": (513, 192),
                   "HIT": (257, 192), "MAFAULDA": (513, 192)}


def canonical_selected_bytes(df: pd.DataFrame) -> bytes:
    cols = ["fold", "seed", "dataset", "class", "group_id", "window_id",
            "class_rank", "n_class"]
    return df.loc[df[FRACTION_COLUMN], cols].sort_values(cols).to_csv(
        index=False, lineterminator="\n").encode()


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def build_fivepct(fold: int, seed: int) -> pd.DataFrame:
    df = build_subset_table(fold, seed)
    assert FRACTION_COLUMN in df.columns
    return df


def generate_manifests() -> tuple[pd.DataFrame, pd.DataFrame]:
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    count_rows, pair_rows = [], []
    for fold in FOLDS:
        for seed in SEEDS:
            df = build_fivepct(fold, seed)
            path = MANIFESTS / f"label_subset_f{fold}_s{seed}_registered5pct.csv"
            df.to_csv(path, index=False)
            selected = canonical_selected_bytes(df)
            selected_hash = sha_bytes(selected)
            # Two arm-specific byte streams are generated independently from
            # the shared manifest and asserted identical before being hashed.
            s0_bytes = canonical_selected_bytes(df.copy(deep=True))
            s1_bytes = canonical_selected_bytes(df.copy(deep=True))
            assert s0_bytes == s1_bytes == selected
            man = pd.read_csv(PART3B_DIR / f"window_manifest_fold_{fold}.csv")
            train_ids = set(man.loc[man.split == "train", "window_id"])
            assert set(df.window_id) == train_ids
            for (ds, cls), grp in df.groupby(["dataset", "class"], sort=True):
                sel = grp[grp[FRACTION_COLUMN]]
                assert len(sel) == math.ceil(NOMINAL_FRACTION * len(grp))
                count_rows.append({
                    "fold": fold, "seed": seed, "dataset": ds, "class": cls,
                    "full_train_windows": len(grp), "selected_windows": len(sel),
                    "nominal_fraction": NOMINAL_FRACTION,
                    "realized_fraction": len(sel) / len(grp),
                    "distinct_groups": sel.group_id.astype(str).nunique(),
                    "selected_group_ids": "|".join(sorted(
                        sel.group_id.astype(str).unique())),
                })
            # Split identities are pairing quantities, not evaluations.
            split_hashes = {}
            for split in ("train", "validation", "test"):
                ids = sorted(man.loc[man.split == split, "window_id"].astype(str))
                split_hashes[split] = sha_bytes(("\n".join(ids) + "\n").encode())
            hseed = head_seed(fold, seed)
            hh = ex.state_dict_hash(DatasetHeads(init_seed=hseed).state_dict())
            spe = steps_per_epoch(len(train_ids))
            stream0 = ex.build_sup_stream(df, NOMINAL_FRACTION, seed,
                                          DOWNSTREAM_EPOCHS * spe)
            stream1 = ex.build_sup_stream(df, NOMINAL_FRACTION, seed,
                                          DOWNSTREAM_EPOCHS * spe)
            sh0, sh1 = ex.stream_hash(stream0), ex.stream_hash(stream1)
            assert sh0 == sh1
            pair_rows.append({
                "fold": fold, "seed": seed,
                "s0_subset_sha256": sha_bytes(s0_bytes),
                "s1_subset_sha256": sha_bytes(s1_bytes),
                "subset_pairing_pass": sha_bytes(s0_bytes) == sha_bytes(s1_bytes),
                "train_rows_sha256": split_hashes["train"],
                "validation_rows_sha256": split_hashes["validation"],
                "test_rows_sha256": split_hashes["test"],
                "head_init_seed": hseed, "head_init_state_sha256": hh,
                "s0_batch_stream_sha256": sh0,
                "s1_batch_stream_sha256": sh1,
                "batch_stream_pairing_pass": sh0 == sh1,
                "steps_per_epoch": spe, "epochs": DOWNSTREAM_EPOCHS,
            })
    counts, pairs = pd.DataFrame(count_rows), pd.DataFrame(pair_rows)
    counts.to_csv(AUDIT / "subset_counts.csv", index=False)
    pairs.to_csv(AUDIT / "pairing_proofs.csv", index=False)
    return counts, pairs


def checkpoint_inventory() -> list[dict]:
    rows = []
    for fold in FOLDS:
        for seed in SEEDS:
            rid = f"ssl_f{fold}_s{seed}"
            p = ex.RESULTS / "ssl" / rid / "best.pt"
            st = ex.RESULTS / "ssl" / rid / "state.json"
            state = json.loads(st.read_text())
            assert p.is_file() and state["status"] == "COMPLETE"
            ck = torch.load(p, map_location="cpu", weights_only=False)
            assert ck["run_id"] == rid and "encoder" in ck
            rows.append({"run_id": rid, "path": str(Path("results") / p.relative_to(RESULTS_ROOT)),
                         "sha256": sha256_file(p), "status": "PASS"})
    return rows


def primary_fingerprint() -> dict:
    roots = [REPO / "methodology_v2/part5_experiment_registry",
             REPO / "scripts/methodology_v2/experiment_executor.py",
             ex.RESULTS / "downstream",
             ex.RESULTS / "ssl"]
    rows = []
    for root in roots:
        paths = [root] if root.is_file() else (
            sorted(p for p in root.rglob("*") if p.is_file()) if root.exists() else [])
        for p in paths:
            display = (str(p.relative_to(REPO)) if p.is_relative_to(REPO)
                       else str(Path("results") / p.relative_to(RESULTS_ROOT)))
            rows.append((display, sha256_file(p)))
    payload = "".join(f"{p}:{h}\n" for p, h in rows).encode()
    return {"sha256": sha_bytes(payload), "files": len(rows)}


def dry_run() -> None:
    AUDIT.mkdir(parents=True, exist_ok=True)
    before = primary_fingerprint()
    verify_frozen_hashes(); verify_part3b_hashes(); verify_part4c_hashes()
    verify_part5d_hash(); ex.verify_part5b_hash()
    registry = pd.read_csv(ex.PART5D_DIR / "main_run_registry.csv")
    registered = registry[registry["label_fraction"] == NOMINAL_FRACTION]
    assert len(registered) == 18 and set(registered["arm"]) == {"S0", "S1"}
    ckpts = checkpoint_inventory()
    counts, pairs = generate_manifests()
    assert counts.selected_windows.min() >= 1
    assert pairs.subset_pairing_pass.all() and pairs.batch_stream_pairing_pass.all()
    assert torch.cuda.is_available()
    device = "cuda"
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

    # Read one raw TRAIN window and its frozen representation per dataset.
    man = pd.read_csv(PART3B_DIR / "window_manifest_fold_1.csv")
    reps, triples, raw_checks, shape_checks = [], [], [], {}
    subset = build_fivepct(1, 42)
    for ds in DATASETS:
        row = man[(man.dataset == ds) & (man.split == "train")].iloc[0]
        raw = read_window(row)
        assert np.isfinite(raw).all() and raw.size > 0
        raw_checks.append({"dataset": ds, "window_id": row.window_id,
                           "samples": int(raw.size)})
        x, meta = get_representation(row.window_id, 1)
        assert tuple(x.shape) == EXPECTED_SHAPES[ds]
        shape_checks[ds] = list(x.shape)
    sampler = SupervisedSampler(subset, NOMINAL_FRACTION, 42)
    batch = sampler.next_batch()
    reps = [get_representation(w, 1)[0:2] for _, _, w in []]  # type guard
    reps = []
    for ds, cls, wid in batch:
        x, meta = get_representation(wid, 1)
        reps.append((x, np.asarray(meta["frequency_hz"], np.float32),
                     np.asarray(meta["time_seconds"], np.float32)))
    collated = collate_representations(reps)
    assert collated["spec"].shape[0] == EFFECTIVE_BATCH

    hseed = head_seed(1, 42)
    timings, losses = {}, {}
    head_hashes = []
    for arm in ("S0", "S1"):
        enc = None
        if arm == "S1":
            bp = ex.RESULTS / "ssl/ssl_f1_s42/best.pt"
            enc = torch.load(bp, map_location="cpu", weights_only=False)["encoder"]
        trainer = SupervisedTrainer(42, hseed, encoder_state=enc, device=device)
        assert sum(p.numel() for p in trainer.encoder.parameters()) == 2_382_033
        head_hashes.append(ex.state_dict_hash(trainer.heads.state_dict()))
        # Shape/head audit is inference-only; do not retain a batch-64 graph
        # before the prescribed micro-batch-32 backward feasibility check.
        with torch.no_grad():
            z = trainer.encoder(**{k: v[:MICRO_BATCH].to(device) for k, v in collated.items()})[
                "global_embedding"]
            assert tuple(z.shape) == (MICRO_BATCH, 192) and torch.isfinite(z).all()
            dims = {ds: trainer.heads(z[:1], ds).shape[-1] for ds in DATASETS}
        assert dims == {"CWRU": 3, "JNU": 4, "HIT": 3, "MAFAULDA": 10}
        del z
        torch.cuda.empty_cache()
        t0 = time.perf_counter()
        loss = trainer.train_step(reps, batch, micro_batch=MICRO_BATCH)
        torch.cuda.synchronize()
        timings[arm] = time.perf_counter() - t0
        losses[arm] = loss
        assert math.isfinite(loss)
        del trainer
        torch.cuda.empty_cache()
    assert head_hashes[0] == head_hashes[1]
    peak = torch.cuda.max_memory_allocated()
    total = torch.cuda.get_device_properties(0).total_memory
    assert peak < total
    after = primary_fingerprint()
    assert before == after, "primary Methodology V2 files changed during dry-run"
    spe_mean = float(pairs.steps_per_epoch.mean())
    sec_step = max(timings.values())
    runtime_hours = sec_step * spe_mean * DOWNSTREAM_EPOCHS / 3600
    total_runtime_hours = runtime_hours * 18
    report = {
        "scientific_status": "REGISTERED Methodology V2 5% label-efficiency experiment",
        "label_status": "registered 5% subset; frozen class-aware group-even nested selection",
        "host": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(0),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO,
                                               text=True).strip(),
        "datasets": {ds: "PASS" for ds in DATASETS},
        "raw_train_windows": raw_checks, "representation_shapes": shape_checks,
        "ssl_checkpoints": ckpts, "subset_manifests": "9/9 PASS",
        "subset_pairing": "9/9 PASS", "representation_pipeline": "PASS",
        "s0_forward_backward": "PASS", "s1_checkpoint_load": "PASS",
        "s1_forward_backward": "PASS", "losses": losses,
        "gpu_peak_allocated_bytes": peak, "gpu_total_bytes": total,
        "gpu_memory": "PASS", "test_metric_evaluated": False,
        "test_representations_loaded": False,
        "primary_methodology_v2_files_modified": False,
        "primary_fingerprint": before, "all_original_seals": "PASS",
        "dry_step_seconds": timings,
        "estimated_runtime_per_run_hours_compute_only": runtime_hours,
        "estimated_total_18_sequential_hours_compute_only": total_runtime_hours,
        "estimate_note": "one-step linear extrapolation; excludes validation, preload, checkpoint I/O",
        "launch_command": "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python scripts/run_methodology_v2_5pct.py drive",
    }
    (AUDIT / "prelaunch_audit.json").write_text(json.dumps(report, indent=2,
                                                            sort_keys=True))
    pd.DataFrame(ckpts).to_csv(AUDIT / "ssl_checkpoint_inventory.csv", index=False)
    print("=== REGISTERED 5% PRE-LAUNCH AUDIT ===")
    print(f"Host: {report['host']}\nGPU: {report['gpu']}\nGit commit: {report['git_commit']}")
    print("\nRegistered fraction 0.05:\nPASS")
    print("\nDatasets:\nCWRU: PASS\nJNU: PASS\nHIT: PASS\nMAFAULDA: PASS")
    print("\nSSL checkpoints:\n9/9 PASS\n\n5% subset manifests:\n9/9 PASS")
    print("\nS0/S1 pairing:\n9/9 PASS\n\nRepresentation shapes:\nPASS")
    print("\nPC-STE parameter count:\n2,382,033 PASS")
    print("\nS0 forward/backward:\nPASS\n\nS1 checkpoint load:\nPASS")
    print("\nS1 forward/backward:\nPASS\n\nNaN/Inf:\nNONE\n\nGPU memory:\nPASS")
    print("\nTEST touched:\nNO\n\nProtected primary fingerprint:\nUNCHANGED")
    print("\n============================================================")
    print("       REGISTERED 5% EXPERIMENT READY TO LAUNCH ✅")
    print("============================================================")


def run_one(run_id: str, resume: bool = False) -> None:
    """Execute one isolated run using the frozen trainer and executor helpers."""
    parts = run_id.split("_")
    arm, fold, seed = parts[0].upper(), int(parts[1][1:]), int(parts[2][1:])
    assert arm in ("S0", "S1") and fold in FOLDS and seed in SEEDS
    ex.verify_all_seals()
    subset = pd.read_csv(MANIFESTS / f"label_subset_f{fold}_s{seed}_registered5pct.csv")
    man = pd.read_csv(PART3B_DIR / f"window_manifest_fold_{fold}.csv").set_index("window_id")
    train_ids, val_ids = ex.split_ids(man, "train"), ex.split_ids(man, "validation")
    spe, epochs = steps_per_epoch(len(train_ids)), DOWNSTREAM_EPOCHS
    stream = ex.build_sup_stream(subset, NOMINAL_FRACTION, seed, epochs * spe)
    stream_sha = ex.stream_hash(stream)
    hseed = head_seed(fold, seed)
    enc_state, ssl_prov = None, None
    if arm == "S1":
        dep = f"ssl_f{fold}_s{seed}"
        bp = ex.RESULTS / "ssl" / dep / "best.pt"
        ck = torch.load(bp, map_location="cpu", weights_only=False)
        enc_state = ck["encoder"]
        ssl_prov = {"run_id": dep, "sha256": sha256_file(bp), "epoch": ck["epoch"]}
    d = RUNS / run_id
    d.mkdir(parents=True, exist_ok=True)
    state_path = d / "state.json"
    if state_path.exists() and not resume:
        raise SystemExit(f"{run_id} already has state; use --resume")
    trainer = SupervisedTrainer(seed, hseed, encoder_state=enc_state, device="cuda")
    sched = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer,
                                               lr_lambda(epochs, spe))
    store = ex.RepStore(fold, man)
    store.preload(train_ids + val_ids)
    val_by_ds = ex.ids_by_dataset(man, val_ids)
    best = {"metric": float("-inf"), "epoch": None, "reports": None}
    start_epoch = 0
    cfg = sha_bytes(json.dumps({"status": "REGISTERED", "fraction": NOMINAL_FRACTION,
        "arm": arm, "fold": fold, "seed": seed, "optimizer": OPTIMIZER_SPEC,
        "epochs": epochs, "spe": spe, "micro_batch": MICRO_BATCH},
        sort_keys=True).encode())
    if resume and (d / "last.pt").exists():
        ck = torch.load(d / "last.pt", map_location="cuda", weights_only=False)
        assert ck["config_hash"] == cfg and ck["stream_hash"] == stream_sha
        trainer.encoder.load_state_dict(ck["encoder"]); trainer.heads.load_state_dict(ck["heads"])
        trainer.optimizer.load_state_dict(ck["optimizer"]); sched.load_state_dict(ck["scheduler"])
        start_epoch, best = ck["epoch"] + 1, ck["best"]
    state = {"run_id": run_id, "status": "RUNNING", "scientific_status": "REGISTERED",
             "nominal_label_fraction": NOMINAL_FRACTION, "fold": fold, "seed": seed,
             "arm": arm, "config_hash": cfg, "stream_sha256": stream_sha,
             "ssl_provenance": ssl_prov, "host": socket.gethostname()}
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True))
    hist = open(d / "epoch_metrics.jsonl", "a")
    try:
        for epoch in range(start_epoch, epochs):
            losses = []
            for k in range(spe):
                batch = stream[epoch * spe + k]
                losses.append(trainer.train_step([store.rep(w) for _, _, w in batch],
                                                  batch, scheduler=sched,
                                                  micro_batch=MICRO_BATCH))
            reports, _ = ex.supervised_eval(trainer, store, val_by_ds, man)
            macro = ex.macro_domain_f1(reports)
            if macro > best["metric"]:  # exact ties retain earlier epoch
                best = {"metric": macro, "epoch": epoch, "reports": reports}
                torch.save({"run_id": run_id, "config_hash": cfg, "epoch": epoch,
                            "macro_f1_val": macro, "val_reports": reports,
                            "encoder": ex.to_cpu_state(trainer.encoder.state_dict()),
                            "heads": ex.to_cpu_state(trainer.heads.state_dict())}, d / "best.pt")
            torch.save({"config_hash": cfg, "stream_hash": stream_sha, "epoch": epoch,
                        "best": best, "encoder": ex.to_cpu_state(trainer.encoder.state_dict()),
                        "heads": ex.to_cpu_state(trainer.heads.state_dict()),
                        "optimizer": trainer.optimizer.state_dict(),
                        "scheduler": sched.state_dict()}, d / "last.pt")
            hist.write(json.dumps({"epoch": epoch, "train_loss_mean": float(np.mean(losses)),
                "val_macro_domain_f1": macro, "best_epoch": best["epoch"]}) + "\n"); hist.flush()
        # TEST remains sealed until the validation-selected checkpoint is immutable.
        ckpt_hash = sha256_file(d / "best.pt")
        (d / "test_seal.json").write_text(json.dumps({"run_id": run_id,
            "best_epoch": best["epoch"], "best_val_macro_f1": best["metric"],
            "best_checkpoint_sha256": ckpt_hash}, indent=2, sort_keys=True))
        ck = torch.load(d / "best.pt", map_location="cuda", weights_only=False)
        trainer.encoder.load_state_dict(ck["encoder"]); trainer.heads.load_state_dict(ck["heads"])
        test_ids = ex.split_ids(man, "test"); store.preload(test_ids)
        reports, rows = ex.supervised_eval(trainer, store, ex.ids_by_dataset(man, test_ids), man)
        test_macro = ex.macro_domain_f1(reports)
        pd.DataFrame(rows).to_csv(d / "test_predictions.csv", index=False)
        (d / "test_report.json").write_text(json.dumps({"run_id": run_id,
            "macro_domain_f1_test": test_macro, "per_dataset_reports": reports,
            "best_checkpoint_sha256": ckpt_hash}, indent=2, sort_keys=True))
        state.update({"status": "COMPLETE", "best_epoch": best["epoch"],
                      "macro_domain_f1_test": test_macro})
    except BaseException:
        state.update({"status": "FAILED", "error": traceback.format_exc()})
        raise
    finally:
        state_path.write_text(json.dumps(state, indent=2, sort_keys=True)); hist.close()


def drive() -> None:
    assert (AUDIT / "prelaunch_audit.json").exists(), "run dry-run first"
    for fold in FOLDS:
        for seed in SEEDS:
            for arm in ("s0", "s1"):
                run_one(f"{arm}_f{fold}_s{seed}_l005")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("dry-run")
    p = sub.add_parser("run"); p.add_argument("--run-id", required=True); p.add_argument("--resume", action="store_true")
    sub.add_parser("drive")
    a = ap.parse_args()
    if a.cmd == "dry-run": dry_run()
    elif a.cmd == "run": run_one(a.run_id, a.resume)
    else: drive()


if __name__ == "__main__":
    main()
