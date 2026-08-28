#!/usr/bin/env python3
"""Train/evaluate the isolated four-domain raw-waveform InceptionTime baseline."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.baselines.inceptiontime import (ARCHITECTURE, CLASS_ORDERS, DATASETS,
                                         FourDomainInceptionTime)
from src.baselines.three_domain import (EXPECTED_LENGTHS, LABEL_FIELD, GuardedWindowAccess,
                                        FourDomainSampler,
                                        classification_metrics, macro4,
                                        validate_manifest)
from src.methodology_v2.part3b_reader import read_window

STEPS = {1: 202, 2: 205, 3: 201}
SEEDS = (42, 1337, 2026)
EPOCHS = 50
RESULT_ROOT = REPO / "results/baselines/inceptiontime_four_domain"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=True) + "\n")
    os.replace(tmp, path)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                          text=True, capture_output=True, check=True).stdout.strip()


def make_config(args, manifest_path: Path, model: torch.nn.Module) -> dict:
    return {
        "baseline": "external_supervised_inceptiontime", "fold": args.fold,
        "seed": args.seed, "labels_percent": 100, "smoke": args.smoke,
        "status_label": "SMOKE / NOT A RESULT" if args.smoke else "REAL",
        "manifest": str(manifest_path.relative_to(REPO)),
        "manifest_sha256": sha256(manifest_path), "git_commit": git_commit(),
        "host": socket.gethostname(), "python": platform.python_version(),
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "architecture": {**ARCHITECTURE,
                         "exact_parameter_count": sum(p.numel() for p in model.parameters())},
        "class_orders": {k: list(v) for k, v in CLASS_ORDERS.items()},
        "training": {"epochs": EPOCHS, "steps_per_epoch": STEPS[args.fold],
                     "effective_batch": 64, "micro_batch": args.micro_batch,
                     "optimizer": "AdamW", "lr": 3e-4,
                     "betas": [0.9, 0.95], "eps": 1e-8,
                     "weight_decay": 0.05, "gradient_clip_global_norm": 1.0,
                     "lr_schedule": "5-epoch linear warmup then cosine to 1e-6",
                     "checkpoint": "strict maximum validation MacroDomainF1 over four equal domains"},
        "input_policy": {"source": "part3b_reader.read_window",
                         "raw_one_second": True, "float32_immediately_before_model": True,
                         "stft": False, "n2": False,
                         "architecture_native_raw_input_exception":
                         "InceptionTime consumes frozen one-second raw windows; PC-STE-only STFT and N2 are not applied", "normalisation": False,
                         "resampling": False, "interpolation": False,
                         "cropping": False, "padding": False,
                         "mixed_length_tensor": False, "test_derived_preprocessing": False},
        "datasets": list(DATASETS),
        "expected_lengths": EXPECTED_LENGTHS,
    }


def lr_factor(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup - 1)
    min_factor = 1e-6 / 3e-4
    return min_factor + (1 - min_factor) * 0.5 * (1 + np.cos(np.pi * progress))


def rows_by_ids(view: pd.DataFrame) -> dict[str, pd.Series]:
    return {str(r.window_id): r for _, r in view.iterrows()}


def train_step(model, optimizer, scheduler, sampler, access, rows, device,
               micro_batch: int) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    batch = sampler.next_batch()
    total_loss = 0.0
    for ds in DATASETS:
        items = [(label, wid) for d, label, wid in batch if d == ds]
        n = len(items)
        for start in range(0, n, micro_batch):
            chunk = items[start:start + micro_batch]
            x = np.stack([access.read(wid, "train") for _, wid in chunk])
            y = np.asarray([CLASS_ORDERS[ds].index(str(label)) for label, _ in chunk])
            xt = torch.from_numpy(x).unsqueeze(1).to(device)
            yt = torch.from_numpy(y).long().to(device)
            loss = F.cross_entropy(model(xt, ds), yt, reduction="sum") / (4.0 * n)
            loss.backward()
            total_loss += float(loss.detach())
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    return total_loss


@torch.no_grad()
def evaluate(model, access, view, split: str, device, batch_size: int,
             cap: int | None = None):
    if split == "test" and set(view.dataset) != set(DATASETS):
        raise AssertionError("TEST evaluator requires exactly four domains")
    model.eval()
    per, records = {}, []
    for ds in DATASETS:
        drows = view[(view.dataset == ds) & (view.split.str.lower() == split)].copy()
        drows = drows.sort_values("window_id")
        if cap is not None:
            drows = drows.iloc[:cap]
        probs_all, truth_all = [], []
        for start in range(0, len(drows), batch_size):
            chunk = drows.iloc[start:start + batch_size]
            x = np.stack([access.read(str(w), split) for w in chunk.window_id])
            truth = np.asarray([CLASS_ORDERS[ds].index(str(row[LABEL_FIELD[ds]]))
                                for _, row in chunk.iterrows()], dtype=np.int64)
            logits = model(torch.from_numpy(x).unsqueeze(1).to(device), ds)
            probs = logits.softmax(1).cpu().numpy()
            probs_all.append(probs); truth_all.append(truth)
            if split == "test":
                for (_, row), yy, pp in zip(chunk.iterrows(), truth, probs):
                    pred = int(pp.argmax())
                    rec = {"window_id": row.window_id, "dataset": ds,
                           "y_true": CLASS_ORDERS[ds][int(yy)],
                           "y_pred": CLASS_ORDERS[ds][pred],
                           "correct": bool(pred == yy)}
                    rec.update({f"prob_{label}": float(pp[i])
                                for i, label in enumerate(CLASS_ORDERS[ds])})
                    records.append(rec)
        y = np.concatenate(truth_all); probs = np.concatenate(probs_all)
        per[ds] = classification_metrics(y, probs, len(CLASS_ORDERS[ds]))
        per[ds]["n_windows"] = int(len(y))
    return {"per_dataset": per, "macro4_f1": macro4(per, "macro_f1"),
            "macro4_auc": macro4(per, "macro_roc_auc_ovr")}, records


def checkpoint(model, optimizer, scheduler, sampler, epoch, global_step, best,
               config_hash):
    return {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "sampler": sampler.state_dict(),
            "epoch": epoch, "global_step": global_step, "best": best,
            "config_hash": config_hash, "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
            "numpy_rng": np.random.get_state()}


def run(args) -> None:
    if args.fold not in STEPS or args.seed not in SEEDS:
        raise SystemExit("fold must be 1/2/3 and seed must be 42/1337/2026")
    if args.micro_batch < 1 or args.eval_batch < 1:
        raise SystemExit("batch sizes must be positive")
    if not args.smoke and not torch.cuda.is_available():
        raise SystemExit("CUDA is mandatory for real training")
    if not torch.cuda.is_available():
        raise SystemExit("smoke benchmark requires CUDA for meaningful feasibility data")
    run_id = ("smoke" if args.smoke else
              f"inceptiontime_f{args.fold}_s{args.seed}_l100")
    out = RESULT_ROOT / run_id
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / "state.json"
    if state_path.exists():
        old = json.loads(state_path.read_text())
        if old.get("status") == "COMPLETE":
            raise SystemExit(f"refusing to overwrite COMPLETE run: {run_id}")
        if old.get("status") == "TEST_EVALUATION_IN_PROGRESS":
            raise SystemExit("refusing restart after TEST access boundary; audit required")
        if not args.resume and not args.smoke:
            raise SystemExit("incomplete run exists; pass --resume")
    manifest_path = REPO / f"methodology_v2/part3_windows/window_manifest_fold_{args.fold}.csv"
    view = validate_manifest(pd.read_csv(manifest_path))
    seed_all(args.seed)
    model = FourDomainInceptionTime().cuda()
    config = make_config(args, manifest_path, model)
    config_text = json.dumps(config, sort_keys=True)
    config_hash = hashlib.sha256(config_text.encode()).hexdigest()
    config["config_sha256"] = config_hash
    atomic_json(out / "config.json", config)
    train = view[view.split.str.lower() == "train"].copy()
    sampler = FourDomainSampler(train, args.seed)
    access = GuardedWindowAccess(view, out / "test_seal.json", read_window,
                                 smoke=args.smoke)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4,
                                  betas=(0.9, 0.95), eps=1e-8,
                                  weight_decay=0.05)
    real_total = EPOCHS * STEPS[args.fold]
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: lr_factor(s, real_total, 5 * STEPS[args.fold]))
    best = {"macro4_f1": -1.0, "epoch": None}
    start_epoch = 0; global_step = 0
    if args.resume:
        ck = torch.load(out / "last.pt", map_location="cuda", weights_only=False)
        if ck["config_hash"] != config_hash:
            raise AssertionError("resume configuration differs")
        model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"]); sampler.load_state_dict(ck["sampler"])
        torch.set_rng_state(ck["torch_rng"]); torch.cuda.set_rng_state_all(ck["cuda_rng"])
        np.random.set_state(ck["numpy_rng"])
        start_epoch = ck["epoch"] + 1; global_step = ck["global_step"]; best = ck["best"]
    state = {"run_id": run_id, "status": "SMOKE / NOT A RESULT" if args.smoke else "RUNNING",
             "fold": args.fold, "seed": args.seed, "started_at": now(),
             "config_sha256": config_hash, "test_waveforms_read": False,
             "test_evaluations": 0}
    atomic_json(state_path, state)
    hist_path = out / "epoch_metrics.jsonl"
    if args.smoke:
        hist_path.write_text("")
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        losses = []
        for _ in range(3):
            losses.append(train_step(model, optimizer, scheduler, sampler, access,
                                     None, "cuda", args.micro_batch))
        torch.cuda.synchronize(); elapsed = time.perf_counter() - t0
        vt = time.perf_counter()
        val, _ = evaluate(model, access, view, "validation", "cuda",
                          args.eval_batch, cap=32)
        torch.cuda.synchronize(); val_seconds = time.perf_counter() - vt
        seconds_step = elapsed / 3
        report = {"marker": "SMOKE / NOT A RESULT", "optimizer_steps": 3,
                  "mean_train_loss": float(np.mean(losses)),
                  "seconds_per_optimizer_step": seconds_step,
                  "windows_per_second": 64 * 3 / elapsed,
                  "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                  "model_parameter_count": sum(p.numel() for p in model.parameters()),
                  "estimated_train_seconds_per_full_epoch_202_steps": seconds_step * 202,
                  "rough_estimated_train_seconds_50_epochs": seconds_step * 202 * 50,
                  "validation_cap_per_dataset": 32,
                  "validation_seconds": val_seconds, "validation": val,
                  "test_accessed": False, "completed_at": now()}
        with hist_path.open("a") as f: f.write(json.dumps(report, allow_nan=True) + "\n")
        state.update({"status": "SMOKE_COMPLETE / NOT A RESULT", "smoke": report})
        atomic_json(state_path, state)
        print(json.dumps(report, indent=2, allow_nan=True))
        return
    rows = rows_by_ids(view)
    with hist_path.open("a") as hist:
        for epoch in range(start_epoch, EPOCHS):
            t0 = time.perf_counter(); losses = []
            for _ in range(STEPS[args.fold]):
                losses.append(train_step(model, optimizer, scheduler, sampler,
                                         access, rows, "cuda", args.micro_batch))
                global_step += 1
            val, _ = evaluate(model, access, view, "validation", "cuda", args.eval_batch)
            rec = {"epoch": epoch + 1, "train_loss_mean": float(np.mean(losses)),
                   "validation": val, "lr": scheduler.get_last_lr()[0],
                   "seconds": time.perf_counter() - t0, "at": now()}
            if val["macro4_f1"] > best["macro4_f1"]:
                best = {"macro4_f1": val["macro4_f1"], "epoch": epoch + 1,
                        "validation": val}
                torch.save({"model": model.state_dict(), "best": best,
                            "config_hash": config_hash}, out / "best.pt")
            torch.save(checkpoint(model, optimizer, scheduler, sampler, epoch,
                                  global_step, best, config_hash), out / "last.pt")
            hist.write(json.dumps(rec, allow_nan=True) + "\n"); hist.flush()
    best_hash = sha256(out / "best.pt")
    atomic_json(out / "test_seal.json", {"sealed_at": now(),
                "best_checkpoint": "best.pt", "best_checkpoint_sha256": best_hash,
                "best_epoch": best["epoch"], "selection_metric": "validation_macro4_f1",
                "selection_value": best["macro4_f1"], "test_evaluations_before_seal": 0})
    # Persist the irreversible boundary before the first TEST read.
    # A crash after this point must fail closed rather than evaluate twice.
    state.update({"status": "TEST_EVALUATION_IN_PROGRESS",
                  "test_waveforms_read": False, "test_evaluations": 1,
                  "test_evaluation_started_at": now()})
    atomic_json(state_path, state)
    frozen = torch.load(out / "best.pt", map_location="cuda", weights_only=False)
    if sha256(out / "best.pt") != best_hash: raise AssertionError("best checkpoint changed")
    model.load_state_dict(frozen["model"])
    report, records = evaluate(model, access, view, "test", "cuda", args.eval_batch)
    pd.DataFrame(records).to_csv(out / "test_predictions.csv", index=False)
    report.update({"evaluated_at": now(), "best_checkpoint_sha256": best_hash,
                   "test_evaluation_count": 1})
    atomic_json(out / "test_report.json", report)
    state.update({"status": "COMPLETE", "completed_at": now(),
                  "test_waveforms_read": True,
                  "best": best, "test": report})
    atomic_json(state_path, state)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--micro-batch", type=int, default=2)
    p.add_argument("--eval-batch", type=int, default=2)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
