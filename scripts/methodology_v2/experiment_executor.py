#!/usr/bin/env python
"""methodology_v2 PRIMARY EXPERIMENT EXECUTOR — launch-authorized 2026-08-12.

Consumes the SEALED Part-5D registries and executes registered runs
exactly as frozen. Epoch loops live HERE, outside
src/methodology_v2/experiment/ (which is guard-tested to contain none).

AUTHORIZED SCOPE: Phase A = the 9 registered SSL runs; Phase B = the 18
registered full-label (100%) downstream runs. Few-shot fractions
(5/10/25/50%) and ablations A1-A4 are NOT launched by this executor
(fail-closed guard on label_fraction).

Run-state model: the sealed registry is immutable (its status column is
frozen at REGISTERED and verify_part5d_hash() would fail closed if it
were edited). Live run state therefore lives in
results/methodology_v2/{ssl,downstream}/<run_id>/state.json with status
RUNNING -> COMPLETE | FAILED. No state file == still REGISTERED.
Failed runs are preserved untouched; a restart is permitted only with
the identical frozen configuration (config-hash checked on resume).

Representation handling (disclosed mechanical decision, profiled at the
gate): each run preloads its fold's TRAIN+VALIDATION representations
into process RAM by calling the SEALED Part-4C reader once per window
in source-file order, then trains from RAM. No disk cache is created
and no value is altered: bit-equality of preloaded vs freshly computed
representations is asserted on a random sample every run and recorded
in state.json (suite test: tests/methodology_v2/test_executor.py).
TEST representations are loaded only AFTER the checkpoint is frozen and
the test seal is written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from src.methodology_v2.integrity import sha256_file  # noqa: E402
from src.methodology_v2.part2_builder import verify_frozen_hashes  # noqa: E402
from src.methodology_v2.part3b_windows import (PART3B_DIR,  # noqa: E402
                                               verify_part3b_hashes)
from src.methodology_v2.part4c_normalizers import verify_part4c_hashes  # noqa: E402
from src.methodology_v2.part4c_reader import get_representation  # noqa: E402
from src.methodology_v2.encoder import collate_representations  # noqa: E402
from src.methodology_v2.encoder.patchify import PATCH_F, PATCH_T  # noqa: E402
from src.methodology_v2.encoder.ssl_design import (baseline_mses,  # noqa: E402
                                                   generate_mask, window_rng)
from src.methodology_v2.experiment.heads import (CLASS_ORDERS,  # noqa: E402
                                                 LABEL_FIELD, DatasetHeads,
                                                 head_seed)
from src.methodology_v2.experiment.metrics import (classification_report,  # noqa: E402
                                                   macro_domain_f1,
                                                   macro_domain_recon_mse)
from src.methodology_v2.experiment.registry import (PART5D_DIR,  # noqa: E402
                                                    SUBSET_DIR, UPSTREAM,
                                                    part5c_spec_hash,
                                                    verify_part5d_hash)
from src.methodology_v2.experiment.samplers import (SSLSampler,  # noqa: E402
                                                    SupervisedSampler)
from src.methodology_v2.experiment.trainers import (DOWNSTREAM_EPOCHS,  # noqa: E402
                                                    GRIDS, MASK_GEOMETRY,
                                                    MASK_RATIO,
                                                    OPTIMIZER_SPEC,
                                                    SSL_EPOCHS, SSLTrainer,
                                                    SupervisedTrainer,
                                                    build_patch_mask,
                                                    lr_lambda,
                                                    validation_mask_seed)

RESULTS = Path(os.environ.get("PCSTE_RESULTS_ROOT", REPO / "results")).expanduser() / "methodology_v2"
MICRO_BATCH = 32              # frozen mechanical adjustment (Part 5D §21)
EXECUTOR_VERSION = "1.0"
DATASETS = ("CWRU", "JNU", "HIT", "MAFAULDA")
AUTHORIZED_FRACTION = 1.0     # Phase-B scope guard: 100% labels only
PART5B_DIR = REPO / "methodology_v2" / "part5_encoder"

SSL_ORDER = [f"ssl_f{f}_s{s}" for f in (1, 2, 3) for s in (42, 1337, 2026)]
DS_ORDER = [f"{arm}_f{f}_s{s}_l100" for f in (1, 2, 3)
            for s in (42, 1337, 2026) for arm in ("s0", "s1")]


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] "
          f"{msg}", flush=True)


# ---------------------------------------------------------------------------
# seals / hashing / state
# ---------------------------------------------------------------------------

def verify_part5b_hash() -> str:
    spec = json.load(open(PART5B_DIR / "pcste_encoder_spec.yaml"))
    h = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    stored = (PART5B_DIR / "part5b_architecture_hash.txt").read_text().strip()
    if h != stored:
        raise AssertionError("Part-5B architecture hash mismatch (fail closed)")
    return stored


def verify_all_seals() -> dict:
    """Every upstream seal, fail closed, plus registry cross-checks."""
    verify_frozen_hashes()
    verify_part3b_hashes()
    verify_part4c_hashes()
    arch = verify_part5b_hash()
    assert arch == UPSTREAM["part5b_architecture"], "5B hash != registry value"
    verify_part5d_hash()
    spec5c = part5c_spec_hash()
    checks = {"part5c_spec_hash_full": spec5c}
    col_map = {"part2_hash": "part2", "part3b_hash": "part3b",
               "part4c_hash": "part4c",
               "architecture_hash": "part5b_architecture"}
    for name in ("ssl_run_registry.csv", "main_run_registry.csv"):
        reg = pd.read_csv(PART5D_DIR / name)
        assert set(reg["part5c_spec_hash"]) == {spec5c}, \
            f"{name}: stored Part-5C spec hash != recomputed (fail closed)"
        for col, key in col_map.items():
            assert set(reg[col]) == {UPSTREAM[key]}, \
                f"{name}: {col} mismatch (fail closed)"
    checks.update({k: v for k, v in UPSTREAM.items()})
    checks["verified_at"] = datetime.now(timezone.utc).isoformat()
    return checks


def _jsonable(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def config_hash(row: pd.Series) -> str:
    """Frozen-run configuration hash: the full registry row (minus the
    immutable REGISTERED status) + executor mechanical constants."""
    d = {k: (v.item() if hasattr(v, "item") else v)
         for k, v in row.to_dict().items() if k != "status"}
    d["micro_batch"] = MICRO_BATCH
    d["gradient_accumulation"] = 2
    return hashlib.sha256(
        json.dumps(d, sort_keys=True, default=_jsonable).encode()).hexdigest()


def run_dir(kind: str, run_id: str) -> Path:
    return RESULTS / kind / run_id


def read_state(d: Path) -> dict | None:
    p = d / "state.json"
    return json.loads(p.read_text()) if p.exists() else None


def write_state(d: Path, state: dict) -> None:
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / "state.json.tmp"
    tmp.write_text(json.dumps(state, indent=1, default=_jsonable,
                              sort_keys=True))
    os.replace(tmp, d / "state.json")


def state_dict_hash(sd: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(sd):
        t = sd[k].detach().cpu().contiguous()
        h.update(k.encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(str(t.dtype).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def to_cpu_state(sd: dict) -> dict:
    return {k: v.detach().cpu().clone() for k, v in sd.items()}


# ---------------------------------------------------------------------------
# representations (in-RAM, sealed-reader values, bit-equality checked)
# ---------------------------------------------------------------------------

class RepStore:
    """Per-run in-process store of sealed Part-4C representations.
    Values come from get_representation() itself — nothing is recomputed
    or persisted; bit-equality vs fresh calls is asserted per run."""

    def __init__(self, fold: int, manifest: pd.DataFrame):
        self.fold = fold
        self.man = manifest              # indexed by window_id
        self.store: dict[str, tuple] = {}

    def preload(self, window_ids: list[str]) -> float:
        todo = [w for w in window_ids if w not in self.store]
        if not todo:
            return 0.0
        sub = self.man.loc[todo]
        order = sub.sort_values(
            ["dataset", "source_file", "start_sample"]).index
        t0 = time.time()
        for wid in order:
            x, meta = get_representation(wid, self.fold)
            self.store[wid] = (
                x,
                np.asarray(meta["frequency_hz"], dtype=np.float32),
                np.asarray(meta["time_seconds"], dtype=np.float32),
                meta["dataset"])
        return time.time() - t0

    def rep(self, wid: str) -> tuple:
        x, f, t, _ = self.store[wid]
        return (x, f, t)

    def equivalence_check(self, k: int = 16, seed: int = 0) -> dict:
        """Bit-equality of stored vs freshly computed representations."""
        wids = sorted(self.store)
        rng = np.random.default_rng(seed)
        sample = [wids[i] for i in
                  rng.choice(len(wids), size=min(k, len(wids)),
                             replace=False)]
        for wid in sample:
            fresh, meta = get_representation(wid, self.fold)
            x, f, t, _ = self.store[wid]
            assert np.array_equal(fresh, x), f"{wid}: preload != fresh"
            assert np.array_equal(
                np.asarray(meta["frequency_hz"], dtype=np.float32), f)
            assert np.array_equal(
                np.asarray(meta["time_seconds"], dtype=np.float32), t)
        return {"checked": len(sample), "bit_equal": True,
                "sample": sample}


def split_ids(man: pd.DataFrame, split: str) -> list[str]:
    return list(man.index[man["split"] == split])


def ids_by_dataset(man: pd.DataFrame, ids: list[str]) -> dict:
    sub = man.loc[ids]
    return {ds: sorted(sub.index[sub["dataset"] == ds]) for ds in DATASETS}


# ---------------------------------------------------------------------------
# deterministic batch streams (pairing-provable)
# ---------------------------------------------------------------------------

def build_ssl_stream(train_view: pd.DataFrame, seed: int,
                     n_steps: int) -> list:
    s = SSLSampler(train_view, seed)
    return [s.next_batch() for _ in range(n_steps)]


def build_sup_stream(subset: pd.DataFrame, fraction: float, seed: int,
                     n_steps: int) -> list:
    s = SupervisedSampler(subset, fraction, seed)
    return [s.next_batch() for _ in range(n_steps)]


def stream_hash(stream: list) -> str:
    h = hashlib.sha256()
    for batch in stream:
        for item in batch:
            h.update(("|".join(str(v) for v in item) + "\n").encode())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# SSL run
# ---------------------------------------------------------------------------

@torch.no_grad()
def ssl_validation(trainer: SSLTrainer, store: RepStore,
                   val_by_ds: dict) -> dict:
    """Per-dataset window-mean masked-cell MSE with FIXED per-seed masks
    (frozen criterion), computed in bounded per-dataset chunks.

    build_patch_mask always emits the GLOBAL-max patch grid (33, 24);
    a single-dataset chunk collates to that dataset's own grid (e.g.
    CWRU 33x23, HIT 17x24), so the mask is sliced to the chunk's grid.
    Sliced cells are padding (all False by construction) — every
    window's mask content is bit-identical to the frozen definition
    (regression-tested in tests/methodology_v2/test_executor.py)."""
    per_ds = {}
    for ds in DATASETS:
        wids = val_by_ds[ds]
        vals: list[float] = []
        for lo in range(0, len(wids), 64):
            chunk = wids[lo:lo + 64]
            reps = [store.rep(w) for w in chunk]
            metas = [(ds, w) for w in chunk]
            batch = collate_representations(reps)
            batch = {k: v.to(trainer.device) for k, v in batch.items()}
            fb = math.ceil(batch["spec"].shape[1] / PATCH_F)
            tp = math.ceil(batch["spec"].shape[2] / PATCH_T)
            pm = build_patch_mask(metas, GRIDS, trainer.seed, 0,
                                  fixed_validation=True)
            assert int(pm.sum()) == int(pm[:, :fb, :tp].sum()), \
                "mask slicing dropped non-padding cells (fail closed)"
            pm = pm[:, :fb, :tp].to(trainer.device)
            out = trainer.model(**batch, patch_mask=pm)
            vals.extend(float(v) for v in out["per_window_mse"])
            del out, batch, pm
        per_ds[ds] = float(np.mean(vals))
    return per_ds


def val_trivial_baselines(store: RepStore, val_by_ds: dict,
                          seed: int) -> dict:
    """P0/P1/P2 trivial-baseline MSEs on the VALIDATION set under the
    SAME fixed per-seed masks the model is scored with (descriptive)."""
    vseed = validation_mask_seed(seed)
    out = {}
    for ds in DATASETS:
        accs: dict[str, list] = {}
        for wid in val_by_ds[ds]:
            x = store.store[wid][0]
            mask = generate_mask(np.ones(GRIDS[ds], dtype=bool),
                                 MASK_RATIO, MASK_GEOMETRY,
                                 window_rng(vseed, 0, wid))
            for k, v in baseline_mses(x, mask).items():
                accs.setdefault(k, []).append(v)
        out[ds] = {k: float(np.mean(v)) for k, v in accs.items()}
    return out


def run_ssl(run_id: str, resume: bool = False) -> None:
    reg = pd.read_csv(PART5D_DIR / "ssl_run_registry.csv"
                      ).set_index("run_id")
    row = reg.loc[run_id]
    fold, seed = int(row["fold"]), int(row["seed"])
    spe, epochs = int(row["steps_per_epoch"]), int(row["max_epochs"])
    assert epochs == SSL_EPOCHS
    d = run_dir("ssl", run_id)
    st = read_state(d)
    if st and st["status"] == "COMPLETE":
        log(f"{run_id}: already COMPLETE — skipping")
        return
    if st and not resume:
        raise SystemExit(f"{run_id}: state={st['status']}; restart requires "
                         f"--resume (identical frozen config only)")
    seals = verify_all_seals()
    cfg = config_hash(row)
    log(f"{run_id}: START fold={fold} seed={seed} spe={spe} "
        f"epochs={epochs} cfg={cfg[:12]}")

    man = pd.read_csv(PART3B_DIR / f"window_manifest_fold_{fold}.csv"
                      ).set_index("window_id")
    train_ids = split_ids(man, "train")
    val_ids = split_ids(man, "validation")
    assert math.ceil(len(train_ids) / 64) == spe, \
        "steps_per_epoch != ceil(fold TRAIN / 64) (fail closed)"

    view = (man.loc[train_ids].reset_index()
            [["dataset", "group_id", "window_id"]])
    stream = build_ssl_stream(view, seed, epochs * spe)
    sh = stream_hash(stream)

    assert torch.cuda.is_available(), "GPU required (frozen environment)"
    trainer = SSLTrainer(seed, device="cuda")
    sched = torch.optim.lr_scheduler.LambdaLR(
        trainer.optimizer, lr_lambda(epochs, spe))

    store = RepStore(fold, man)
    pre_s = store.preload(train_ids + val_ids)
    eq = store.equivalence_check(seed=seed)
    val_by_ds = ids_by_dataset(man, val_ids)
    log(f"{run_id}: preloaded {len(store.store)} representations in "
        f"{pre_s:.1f}s; bit-equality {eq['checked']}/{eq['checked']}")

    start_epoch = 0
    best = {"metric": float("inf"), "epoch": None, "per_dataset": None}
    if resume and (d / "last.pt").exists():
        ck = torch.load(d / "last.pt", map_location="cuda",
                        weights_only=False)
        assert ck["config_hash"] == cfg, \
            "resume config differs from frozen config (fail closed)"
        assert ck["stream_hash"] == sh, "batch stream mismatch on resume"
        trainer.model.load_state_dict(ck["model"])
        trainer.optimizer.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        torch.set_rng_state(ck["torch_rng"].cpu())
        torch.cuda.set_rng_state_all([s.cpu() for s in ck["cuda_rng"]])
        start_epoch = ck["epoch"] + 1
        best = ck["best"]
        log(f"{run_id}: RESUMED at epoch {start_epoch} "
            f"(best={best['metric']:.6f}@{best['epoch']})")

    state = {"run_id": run_id, "kind": "ssl", "status": "RUNNING",
             "fold": fold, "seed": seed, "config_hash": cfg,
             "stream_sha256": sh, "seals": seals,
             "preload_seconds": round(pre_s, 2),
             "preload_windows": len(store.store),
             "preload_equivalence": eq["checked"],
             "started_at": datetime.now(timezone.utc).isoformat(),
             "resumed_from_epoch": start_epoch if resume else None,
             "pid": os.getpid(), "host": socket.gethostname(),
             "encoder_params": sum(p.numel()
                                   for p in trainer.encoder.parameters()),
             "model_params": sum(p.numel()
                                 for p in trainer.model.parameters())}
    write_state(d, state)

    hist = open(d / "epoch_metrics.jsonl", "a")
    try:
        for epoch in range(start_epoch, epochs):
            t0 = time.time()
            losses = []
            for k in range(spe):
                batch = stream[epoch * spe + k]
                reps = [store.rep(w) for _, w in batch]
                losses.append(trainer.train_step(
                    reps, batch, epoch, sched, micro_batch=MICRO_BATCH))
            t_tr = time.time() - t0
            t0 = time.time()
            per_ds = ssl_validation(trainer, store, val_by_ds)
            macro = macro_domain_recon_mse(per_ds)
            t_val = time.time() - t0
            if macro < best["metric"]:          # strict: ties keep earlier
                best = {"metric": macro, "epoch": epoch,
                        "per_dataset": per_ds}
                torch.save({"run_id": run_id, "config_hash": cfg,
                            "epoch": epoch, "macro": macro,
                            "per_dataset": per_ds,
                            "encoder": trainer.encoder_state(),
                            "model": to_cpu_state(
                                trainer.model.state_dict())},
                           d / "best.pt")
            torch.save({"run_id": run_id, "config_hash": cfg,
                        "stream_hash": sh, "epoch": epoch, "best": best,
                        "model": to_cpu_state(trainer.model.state_dict()),
                        "optimizer": trainer.optimizer.state_dict(),
                        "scheduler": sched.state_dict(),
                        "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state_all()},
                       d / "last.pt")
            rec = {"epoch": epoch, "train_loss_mean": float(np.mean(losses)),
                   "val_per_dataset_mse": per_ds, "val_macro_mse": macro,
                   "best_epoch": best["epoch"],
                   "lr": sched.get_last_lr()[0],
                   "seconds_train": round(t_tr, 1),
                   "seconds_val": round(t_val, 1),
                   "at": datetime.now(timezone.utc).isoformat()}
            hist.write(json.dumps(rec, default=_jsonable) + "\n")
            hist.flush()
            log(f"{run_id}: epoch {epoch + 1}/{epochs} "
                f"loss={rec['train_loss_mean']:.4f} val={macro:.6f} "
                f"best={best['metric']:.6f}@{best['epoch']} "
                f"({t_tr:.0f}s+{t_val:.0f}s)")
        log(f"{run_id}: computing trivial-baseline comparison on "
            f"validation (fixed masks)")
        baselines = val_trivial_baselines(store, val_by_ds, seed)
        ckpt_hash = sha256_file(d / "best.pt")
        completion = {"run_id": run_id, "fold": fold, "seed": seed,
                      "best_epoch": best["epoch"],
                      "best_val_macro_mse": best["metric"],
                      "best_val_per_dataset_mse": best["per_dataset"],
                      "val_trivial_baselines": baselines,
                      "best_checkpoint_sha256": ckpt_hash,
                      "config_hash": cfg, "stream_sha256": sh,
                      "epochs_completed": epochs}
        (d / "completion.json").write_text(
            json.dumps(completion, indent=1, default=_jsonable,
                       sort_keys=True))
        state.update({"status": "COMPLETE", "best_epoch": best["epoch"],
                      "best_val_macro_mse": best["metric"],
                      "best_checkpoint_sha256": ckpt_hash,
                      "finished_at": datetime.now(timezone.utc).isoformat()})
        write_state(d, state)
        log(f"{run_id}: COMPLETE best={best['metric']:.6f} "
            f"@epoch {best['epoch']}")
    except BaseException:
        state.update({"status": "FAILED",
                      "error": traceback.format_exc(),
                      "failed_at": datetime.now(timezone.utc).isoformat()})
        write_state(d, state)
        log(f"{run_id}: FAILED (state preserved)")
        raise
    finally:
        hist.close()


# ---------------------------------------------------------------------------
# downstream run (S0/S1, 100% labels only)
# ---------------------------------------------------------------------------

@torch.no_grad()
def supervised_eval(trainer: SupervisedTrainer, store: RepStore,
                    by_ds: dict, man: pd.DataFrame) -> tuple[dict, list]:
    reports, rows = {}, []
    for ds in DATASETS:
        wids = by_ds[ds]
        y_true = [str(man.loc[w][LABEL_FIELD[ds]]) for w in wids]
        y_pred: list[str] = []
        for lo in range(0, len(wids), 64):
            chunk = wids[lo:lo + 64]
            reps = [store.rep(w) for w in chunk]
            y_pred.extend(trainer.predict(reps, [ds] * len(chunk)))
        reports[ds] = classification_report(y_true, y_pred, ds)
        rows.extend({"window_id": w, "dataset": ds, "y_true": t,
                     "y_pred": p, "correct": t == p}
                    for w, t, p in zip(wids, y_true, y_pred))
    return reports, rows


def run_downstream(run_id: str, resume: bool = False) -> None:
    reg = pd.read_csv(PART5D_DIR / "main_run_registry.csv"
                      ).set_index("run_id")
    row = reg.loc[run_id]
    frac = float(row["label_fraction"])
    assert frac == AUTHORIZED_FRACTION, (
        f"{run_id}: label_fraction={frac} is OUTSIDE the launch "
        f"authorization (100% only) — fail closed")
    arm = str(row["arm"])
    fold, seed = int(row["fold"]), int(row["seed"])
    spe, epochs = int(row["steps_per_epoch"]), int(row["max_epochs"])
    assert epochs == DOWNSTREAM_EPOCHS
    d = run_dir("downstream", run_id)
    st = read_state(d)
    if st and st["status"] == "COMPLETE":
        log(f"{run_id}: already COMPLETE — skipping")
        return
    if st and not resume:
        raise SystemExit(f"{run_id}: state={st['status']}; restart requires "
                         f"--resume (identical frozen config only)")
    seals = verify_all_seals()
    cfg = config_hash(row)
    log(f"{run_id}: START arm={arm} fold={fold} seed={seed} "
        f"frac={frac} cfg={cfg[:12]}")

    spath = SUBSET_DIR / f"label_subset_f{fold}_s{seed}.csv"
    assert sha256_file(spath) == row["label_subset_hash"], \
        "label subset hash mismatch (fail closed)"
    subset = pd.read_csv(spath)

    man = pd.read_csv(PART3B_DIR / f"window_manifest_fold_{fold}.csv"
                      ).set_index("window_id")
    train_ids = split_ids(man, "train")
    val_ids = split_ids(man, "validation")
    assert set(subset.loc[subset["frac_100"], "window_id"]) \
        == set(train_ids), "frac_100 subset != full TRAIN (fail closed)"

    hseed = head_seed(fold, seed)
    heads_hash = state_dict_hash(DatasetHeads(init_seed=hseed).state_dict())
    stream = build_sup_stream(subset, frac, seed, epochs * spe)
    sh = stream_hash(stream)
    pairing = {"run_id": run_id, "arm": arm, "fold": fold, "seed": seed,
               "label_fraction": frac,
               "label_subset_sha256": str(row["label_subset_hash"]),
               "head_init_seed": hseed,
               "head_init_state_sha256": heads_hash,
               "batch_stream_sha256": sh,
               "steps_per_epoch": spe, "epochs": epochs,
               "optimizer_spec": OPTIMIZER_SPEC,
               "micro_batch": MICRO_BATCH,
               "paired_quantities": ["label_subset", "head_init",
                                     "batch_stream", "optimizer",
                                     "schedule", "steps"],
               "note": "identical values for the paired S0/S1 run prove "
                       "the arms differ ONLY in encoder initialization"}
    (d / "pairing_proof.json").parent.mkdir(parents=True, exist_ok=True)
    (d / "pairing_proof.json").write_text(
        json.dumps(pairing, indent=1, default=_jsonable, sort_keys=True))

    enc_state, ssl_prov = None, None
    if arm == "S1":
        dep = str(row["ssl_checkpoint_dependency"])
        ssl_st = read_state(run_dir("ssl", dep))
        assert ssl_st and ssl_st["status"] == "COMPLETE", \
            f"SSL dependency {dep} not COMPLETE (fail closed)"
        bp = run_dir("ssl", dep) / "best.pt"
        ck = torch.load(bp, map_location="cpu", weights_only=False)
        assert ck["run_id"] == dep
        enc_state = ck["encoder"]
        ssl_prov = {"ssl_run": dep, "ssl_best_epoch": ck["epoch"],
                    "ssl_val_macro_mse": ck["macro"],
                    "ssl_checkpoint_sha256": sha256_file(bp)}
        log(f"{run_id}: loaded SSL encoder from {dep} "
            f"(epoch {ck['epoch']}, {ssl_prov['ssl_checkpoint_sha256'][:12]})")

    assert torch.cuda.is_available(), "GPU required (frozen environment)"
    trainer = SupervisedTrainer(seed, hseed, encoder_state=enc_state,
                                device="cuda")
    assert state_dict_hash(trainer.heads.state_dict()) == heads_hash, \
        "trainer head init != recorded paired head init (fail closed)"
    sched = torch.optim.lr_scheduler.LambdaLR(
        trainer.optimizer, lr_lambda(epochs, spe))

    store = RepStore(fold, man)
    pre_s = store.preload(train_ids + val_ids)
    eq = store.equivalence_check(seed=seed)
    val_by_ds = ids_by_dataset(man, val_ids)
    log(f"{run_id}: preloaded {len(store.store)} representations in "
        f"{pre_s:.1f}s; bit-equality {eq['checked']}/{eq['checked']}")

    start_epoch = 0
    best = {"metric": float("-inf"), "epoch": None, "reports": None}
    if resume and (d / "last.pt").exists():
        ck = torch.load(d / "last.pt", map_location="cuda",
                        weights_only=False)
        assert ck["config_hash"] == cfg, \
            "resume config differs from frozen config (fail closed)"
        assert ck["stream_hash"] == sh, "batch stream mismatch on resume"
        trainer.encoder.load_state_dict(ck["encoder"])
        trainer.heads.load_state_dict(ck["heads"])
        trainer.optimizer.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        torch.set_rng_state(ck["torch_rng"].cpu())
        torch.cuda.set_rng_state_all([s.cpu() for s in ck["cuda_rng"]])
        start_epoch = ck["epoch"] + 1
        best = ck["best"]
        log(f"{run_id}: RESUMED at epoch {start_epoch}")

    state = {"run_id": run_id, "kind": "downstream", "status": "RUNNING",
             "arm": arm, "fold": fold, "seed": seed,
             "label_fraction": frac, "config_hash": cfg,
             "stream_sha256": sh, "seals": seals,
             "head_init_state_sha256": heads_hash,
             "ssl_provenance": ssl_prov,
             "preload_seconds": round(pre_s, 2),
             "preload_windows": len(store.store),
             "preload_equivalence": eq["checked"],
             "started_at": datetime.now(timezone.utc).isoformat(),
             "resumed_from_epoch": start_epoch if resume else None,
             "pid": os.getpid(), "host": socket.gethostname(),
             "encoder_params": sum(p.numel()
                                   for p in trainer.encoder.parameters()),
             "head_params": sum(p.numel()
                                for p in trainer.heads.parameters())}
    write_state(d, state)

    hist = open(d / "epoch_metrics.jsonl", "a")
    try:
        for epoch in range(start_epoch, epochs):
            t0 = time.time()
            losses = []
            for k in range(spe):
                batch = stream[epoch * spe + k]
                reps = [store.rep(w) for _, _, w in batch]
                losses.append(trainer.train_step(
                    reps, batch, scheduler=sched,
                    micro_batch=MICRO_BATCH))
            t_tr = time.time() - t0
            t0 = time.time()
            reports, _ = supervised_eval(trainer, store, val_by_ds, man)
            macro = macro_domain_f1(reports)
            t_val = time.time() - t0
            if macro > best["metric"]:          # strict: ties keep earlier
                best = {"metric": macro, "epoch": epoch,
                        "reports": reports}
                torch.save({"run_id": run_id, "config_hash": cfg,
                            "epoch": epoch, "macro_f1_val": macro,
                            "val_reports": reports,
                            "encoder": to_cpu_state(
                                trainer.encoder.state_dict()),
                            "heads": to_cpu_state(
                                trainer.heads.state_dict())},
                           d / "best.pt")
            torch.save({"run_id": run_id, "config_hash": cfg,
                        "stream_hash": sh, "epoch": epoch, "best": best,
                        "encoder": to_cpu_state(
                            trainer.encoder.state_dict()),
                        "heads": to_cpu_state(trainer.heads.state_dict()),
                        "optimizer": trainer.optimizer.state_dict(),
                        "scheduler": sched.state_dict(),
                        "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state_all()},
                       d / "last.pt")
            per_ds_f1 = {ds: reports[ds]["macro_f1"] for ds in DATASETS}
            rec = {"epoch": epoch, "train_loss_mean": float(np.mean(losses)),
                   "val_per_dataset_macro_f1": per_ds_f1,
                   "val_macro_domain_f1": macro,
                   "best_epoch": best["epoch"],
                   "lr": sched.get_last_lr()[0],
                   "seconds_train": round(t_tr, 1),
                   "seconds_val": round(t_val, 1),
                   "at": datetime.now(timezone.utc).isoformat()}
            hist.write(json.dumps(rec, default=_jsonable) + "\n")
            hist.flush()
            log(f"{run_id}: epoch {epoch + 1}/{epochs} "
                f"loss={rec['train_loss_mean']:.4f} valF1={macro:.4f} "
                f"best={best['metric']:.4f}@{best['epoch']} "
                f"({t_tr:.0f}s+{t_val:.0f}s)")

        # ---- checkpoint freeze + SINGLE sealed TEST evaluation ----
        if not (d / "test_report.json").exists():
            ckpt_hash = sha256_file(d / "best.pt")
            seal = {"run_id": run_id,
                    "best_epoch": best["epoch"],
                    "best_val_macro_f1": best["metric"],
                    "best_checkpoint_sha256": ckpt_hash,
                    "sealed_at": datetime.now(timezone.utc).isoformat(),
                    "rule": "TEST evaluated exactly once after this seal; "
                            "no training influence possible"}
            (d / "test_seal.json").write_text(
                json.dumps(seal, indent=1, default=_jsonable,
                           sort_keys=True))
            ck = torch.load(d / "best.pt", map_location="cuda",
                            weights_only=False)
            trainer.encoder.load_state_dict(ck["encoder"])
            trainer.heads.load_state_dict(ck["heads"])
            test_ids = split_ids(man, "test")
            store.preload(test_ids)             # only AFTER the seal
            test_by_ds = ids_by_dataset(man, test_ids)
            t0 = time.time()
            test_reports, pred_rows = supervised_eval(
                trainer, store, test_by_ds, man)
            test_macro = macro_domain_f1(test_reports)
            pd.DataFrame(pred_rows).to_csv(
                d / "test_predictions.csv", index=False)
            test_out = {"run_id": run_id, "arm": arm, "fold": fold,
                        "seed": seed, "label_fraction": frac,
                        "best_epoch": best["epoch"],
                        "best_val_macro_f1": best["metric"],
                        "best_checkpoint_sha256": ckpt_hash,
                        "macro_domain_f1_test": test_macro,
                        "per_dataset_reports": test_reports,
                        "ssl_provenance": ssl_prov,
                        "config_hash": cfg,
                        "eval_seconds": round(time.time() - t0, 1),
                        "evaluated_at": datetime.now(
                            timezone.utc).isoformat()}
            (d / "test_report.json").write_text(
                json.dumps(test_out, indent=1, default=_jsonable,
                           sort_keys=True))
            log(f"{run_id}: TEST MacroDomainF1 = {test_macro:.4f} "
                f"(single sealed evaluation)")
        test_macro = json.loads(
            (d / "test_report.json").read_text())["macro_domain_f1_test"]
        state.update({"status": "COMPLETE", "best_epoch": best["epoch"],
                      "best_val_macro_f1": best["metric"],
                      "macro_domain_f1_test": test_macro,
                      "finished_at": datetime.now(timezone.utc).isoformat()})
        write_state(d, state)
        log(f"{run_id}: COMPLETE")
    except BaseException:
        state.update({"status": "FAILED",
                      "error": traceback.format_exc(),
                      "failed_at": datetime.now(timezone.utc).isoformat()})
        write_state(d, state)
        log(f"{run_id}: FAILED (state preserved)")
        raise
    finally:
        hist.close()


# ---------------------------------------------------------------------------
# aggregation + pre-registered statistics
# ---------------------------------------------------------------------------

def exact_sign_flip_p(deltas: list[float]) -> float:
    """Exact two-sided paired sign-flip permutation test on the mean.
    p = #(|mean(sign-flipped)| >= |mean(observed)|) / 2^n."""
    n = len(deltas)
    obs = abs(float(np.mean(deltas)))
    arr = np.asarray(deltas, dtype=np.float64)
    count = 0
    for pattern in range(2 ** n):
        signs = np.array([1.0 if (pattern >> i) & 1 else -1.0
                          for i in range(n)])
        if abs(float(np.mean(arr * signs))) >= obs - 1e-12:
            count += 1
    return count / (2 ** n)


def aggregate_ssl() -> None:
    rows = []
    for rid in SSL_ORDER:
        c = json.loads((run_dir("ssl", rid) / "completion.json").read_text())
        r = {"run_id": rid, "fold": c["fold"], "seed": c["seed"],
             "best_epoch": c["best_epoch"],
             "val_macro_mse": c["best_val_macro_mse"],
             "checkpoint_sha256": c["best_checkpoint_sha256"]}
        for ds in DATASETS:
            r[f"val_mse_{ds}"] = c["best_val_per_dataset_mse"][ds]
            for k, v in c["val_trivial_baselines"][ds].items():
                r[f"{k}_{ds}"] = v
        rows.append(r)
    df = pd.DataFrame(rows)
    RESULTS.joinpath("ssl").mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS / "ssl" / "SSL_COMPLETION_TABLE.csv", index=False)

    lines = ["# Phase A — SSL completion table (9/9)", "",
             "| run | fold | seed | best epoch | val MacroDomainReconMSE |",
             "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['run_id']} | {r['fold']} | {r['seed']} | "
                     f"{r['best_epoch']} | {r['val_macro_mse']:.6f} |")
    lines += ["", "## Descriptive comparison vs trivial baselines "
              "(validation, same fixed masks)", "",
              "| dataset | model (mean over 9 runs) | P0 zero | "
              "P1 temporal | P2 frequency |", "|---|---|---|---|---|"]
    for ds in DATASETS:
        lines.append(
            f"| {ds} | {df[f'val_mse_{ds}'].mean():.4f} | "
            f"{df[f'P0_zero_{ds}'].mean():.4f} | "
            f"{df[f'P1_temporal_neighbour_{ds}'].mean():.4f} | "
            f"{df[f'P2_frequency_neighbour_{ds}'].mean():.4f} |")
    (RESULTS / "ssl" / "SSL_COMPLETION_TABLE.md").write_text(
        "\n".join(lines) + "\n")
    log("Phase A aggregation written (SSL_COMPLETION_TABLE.csv/.md)")


def aggregate_primary() -> None:
    out_dir = RESULTS / "primary_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    cells = []
    for f in (1, 2, 3):
        for s in (42, 1337, 2026):
            pair = {}
            for arm in ("s0", "s1"):
                rid = f"{arm}_f{f}_s{s}_l100"
                rep = json.loads((run_dir("downstream", rid)
                                  / "test_report.json").read_text())
                pair[arm] = rep
            p0 = json.loads((run_dir("downstream", f"s0_f{f}_s{s}_l100")
                             / "pairing_proof.json").read_text())
            p1 = json.loads((run_dir("downstream", f"s1_f{f}_s{s}_l100")
                             / "pairing_proof.json").read_text())
            for key in ("label_subset_sha256", "head_init_state_sha256",
                        "batch_stream_sha256", "steps_per_epoch", "epochs"):
                assert p0[key] == p1[key], \
                    f"pairing proof mismatch f{f}s{s}: {key}"
            cell = {"fold": f, "seed": s,
                    "s0_macro_f1_test": pair["s0"]["macro_domain_f1_test"],
                    "s1_macro_f1_test": pair["s1"]["macro_domain_f1_test"],
                    "s0_best_epoch": pair["s0"]["best_epoch"],
                    "s1_best_epoch": pair["s1"]["best_epoch"],
                    "pairing_verified": True}
            cell["delta"] = (cell["s1_macro_f1_test"]
                             - cell["s0_macro_f1_test"])
            for ds in DATASETS:
                cell[f"s0_f1_{ds}"] = \
                    pair["s0"]["per_dataset_reports"][ds]["macro_f1"]
                cell[f"s1_f1_{ds}"] = \
                    pair["s1"]["per_dataset_reports"][ds]["macro_f1"]
            cells.append(cell)
    df = pd.DataFrame(cells)
    df.to_csv(out_dir / "PRIMARY_RESULT_TABLE.csv", index=False)

    deltas = list(df["delta"])
    sd = float(np.std(deltas, ddof=1))
    analysis = {
        "primary_metric": "MacroDomainF1_test",
        "n_pairs": len(deltas),
        "mean_s0": float(df["s0_macro_f1_test"].mean()),
        "mean_s1": float(df["s1_macro_f1_test"].mean()),
        "deltas_s1_minus_s0": deltas,
        "mean_delta": float(np.mean(deltas)),
        "median_delta": float(np.median(deltas)),
        "sd_delta": sd,
        "cohen_dz": float(np.mean(deltas) / sd) if sd > 0 else None,
        "wins_s1": int((df["delta"] > 0).sum()),
        "losses_s1": int((df["delta"] < 0).sum()),
        "ties": int((df["delta"] == 0).sum()),
        "exact_sign_flip_p_two_sided": exact_sign_flip_p(deltas),
        "n_permutations": 2 ** len(deltas),
        "per_dataset_mean_delta": {
            ds: float((df[f"s1_f1_{ds}"] - df[f"s0_f1_{ds}"]).mean())
            for ds in DATASETS},
        "caveat": "fold x seed cells are paired but not fully independent; "
                  "interpretation must not reduce to significance alone",
    }
    (out_dir / "primary_analysis.json").write_text(
        json.dumps(analysis, indent=1, default=_jsonable, sort_keys=True))

    lines = ["# PRIMARY RESULT — S1 vs S0 at 100% labels "
             "(MacroDomainF1_test)", "",
             "| fold | seed | S0 | S1 | Δ = S1−S0 |", "|---|---|---|---|---|"]
    for c in cells:
        lines.append(f"| {c['fold']} | {c['seed']} | "
                     f"{c['s0_macro_f1_test']:.4f} | "
                     f"{c['s1_macro_f1_test']:.4f} | "
                     f"{c['delta']:+.4f} |")
    lines += ["",
              f"Mean S0 = {analysis['mean_s0']:.4f}; "
              f"mean S1 = {analysis['mean_s1']:.4f}; "
              f"mean Δ = {analysis['mean_delta']:+.4f} "
              f"(median {analysis['median_delta']:+.4f}, "
              f"SD {analysis['sd_delta']:.4f}).",
              f"S1 wins {analysis['wins_s1']}/9, "
              f"losses {analysis['losses_s1']}/9, ties {analysis['ties']}.",
              f"Exact two-sided paired sign-flip test (512 permutations): "
              f"p = {analysis['exact_sign_flip_p_two_sided']:.6f}.", "",
              analysis["caveat"]]
    (out_dir / "PRIMARY_ANALYSIS.md").write_text("\n".join(lines) + "\n")
    log(f"Primary analysis written: mean_delta="
        f"{analysis['mean_delta']:+.4f} "
        f"p={analysis['exact_sign_flip_p_two_sided']:.6f}")


# ---------------------------------------------------------------------------
# pre-launch gate (ACTION 2)
# ---------------------------------------------------------------------------

def profile_representation_loading() -> dict:
    """Measured comparison: lazy random-order access (per-step training
    pattern) vs source-file-ordered access (preload pattern). Required
    disclosure BEFORE any deviation from pure lazy loading."""
    man = pd.read_csv(PART3B_DIR / "window_manifest_fold_1.csv"
                      ).set_index("window_id")
    train = man[man["split"] == "train"]
    rng = np.random.default_rng(0)
    sample = []
    for ds, k in (("CWRU", 8), ("JNU", 8), ("HIT", 8), ("MAFAULDA", 12)):
        wids = list(train.index[train["dataset"] == ds])
        sample += [wids[i] for i in rng.choice(len(wids), size=k,
                                               replace=False)]
    order = list(rng.permutation(sample))          # lazy access pattern
    t0 = time.time()
    for w in order:
        get_representation(w, 1)
    lazy_s = time.time() - t0
    ordered = list(man.loc[sample].sort_values(
        ["dataset", "source_file", "start_sample"]).index)
    t0 = time.time()
    for w in ordered:
        get_representation(w, 1)
    ordered_s = time.time() - t0
    per_lazy = lazy_s / len(sample)
    return {"sample_windows": len(sample),
            "lazy_random_order_seconds": round(lazy_s, 2),
            "file_ordered_seconds_warm": round(ordered_s, 2),
            "lazy_seconds_per_window": round(per_lazy, 4),
            "est_lazy_seconds_per_64_window_step": round(per_lazy * 64, 2),
            "measured_compute_seconds_per_step": 2.5,
            "note": "lazy per-step loading would multiply wall time; "
                    "executor therefore preloads per run via the sealed "
                    "reader (bit-equality asserted per run + suite test); "
                    "no disk cache is created"}


def gate() -> None:
    log("PRE-LAUNCH GATE (ACTION 2)")
    seals = verify_all_seals()
    log("  all seals verified fail-closed (2/3B/4C/5B/5C-full/5D)")
    r = subprocess.run(
        [sys.executable, "-m", "pytest",
         str(REPO / "tests" / "methodology_v2" / "test_part5b_encoder.py"),
         "-q"], capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, f"Part-5B/parity tests failed:\n{r.stdout}"
    parity = json.loads((PART5B_DIR / "mamba_reference_parity.json")
                        .read_text())
    assert parity["all_pass"] is True
    log("  Mamba reference parity: all_pass=True; encoder tests green")

    assert torch.cuda.is_available(), "no CUDA device visible"
    gpu = torch.cuda.get_device_name(0)
    free_b, total_b = torch.cuda.mem_get_info()
    assert free_b / 2**30 > 18, f"GPU not free: {free_b / 2**30:.1f} GiB"
    disk = shutil.disk_usage(REPO)
    assert disk.free / 2**30 > 50, "insufficient disk"

    stale = list(RESULTS.glob("*/*/state.json"))
    completes = [p for p in stale
                 if json.loads(p.read_text()).get("status") == "COMPLETE"]
    assert not completes, f"stale COMPLETE states exist: {completes}"
    RESULTS.mkdir(parents=True, exist_ok=True)
    for sub in ("ssl", "downstream", "primary_analysis"):
        (RESULTS / sub).mkdir(exist_ok=True)
        probe = RESULTS / sub / ".write_probe"
        probe.write_text("ok")
        probe.unlink()
    log(f"  GPU: {gpu} ({free_b / 2**30:.1f} GiB free); "
        f"disk {disk.free / 2**30:.0f} GiB free; output dirs writable; "
        f"no stale COMPLETE state")

    log("  profiling representation loading (required disclosure)...")
    profile = profile_representation_loading()
    log(f"  lazy per-step estimate: "
        f"{profile['est_lazy_seconds_per_64_window_step']}s/step vs "
        f"{profile['measured_compute_seconds_per_step']}s compute")

    smi = subprocess.run(["nvidia-smi", "--query-gpu=driver_version",
                          "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    git_sha = subprocess.run(["git", "rev-parse", "HEAD"],
                             capture_output=True, text=True,
                             cwd=REPO).stdout.strip()
    import einops
    import scipy
    env = {"seals": seals,
           "python": sys.version.split()[0],
           "torch": torch.__version__,
           "cuda": torch.version.cuda,
           "cudnn": torch.backends.cudnn.version(),
           "gpu": gpu, "driver": smi,
           "numpy": np.__version__, "pandas": pd.__version__,
           "scipy": scipy.__version__, "einops": einops.__version__,
           "hostname": socket.gethostname(),
           "platform": platform.platform(),
           "git_head": git_sha,
           "executor_sha256": sha256_file(Path(__file__)),
           "micro_batch": MICRO_BATCH, "effective_batch": 64,
           "mamba_backend": "pure-PyTorch reference selective scan "
                            "(parity-verified; fused kernels unbuildable "
                            "on this stack — frozen for the whole matrix)",
           "representation_loading_profile": profile,
           "authorized_runs": {"phase_A_ssl": SSL_ORDER,
                               "phase_B_downstream": DS_ORDER},
           "gate_passed_at": datetime.now(timezone.utc).isoformat()}
    (RESULTS / "environment_record.json").write_text(
        json.dumps(env, indent=1, default=_jsonable, sort_keys=True))
    log("GATE PASSED — environment_record.json written")


# ---------------------------------------------------------------------------
# sequential driver
# ---------------------------------------------------------------------------

def _drive_one(kind: str, rid: str) -> None:
    d = run_dir(kind, rid)
    d.mkdir(parents=True, exist_ok=True)
    for attempt in (1, 2):
        st = read_state(d)
        if st and st["status"] == "COMPLETE":
            return
        cmd = [sys.executable, str(Path(__file__).resolve()), "run",
               "--run-id", rid]
        if st is not None:
            cmd.append("--resume")       # identical-config restart only
        log(f"driver: launching {rid} (attempt {attempt})")
        with open(d / "run_log.txt", "a") as f:
            f.write(f"\n===== attempt {attempt} "
                    f"{datetime.now(timezone.utc).isoformat()} =====\n")
            f.flush()
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=REPO)
        st = read_state(d)
        if st and st["status"] == "COMPLETE":
            return
    raise SystemExit(
        f"driver: {rid} FAILED twice — stopping (state preserved for "
        f"review; no silent alteration)")


def drive(phase: str) -> None:
    assert (RESULTS / "environment_record.json").exists(), \
        "run the gate first"
    dstate = {"phase": phase, "started_at":
              datetime.now(timezone.utc).isoformat()}

    def heartbeat(msg):
        dstate["last"] = msg
        dstate["at"] = datetime.now(timezone.utc).isoformat()
        (RESULTS / "driver_state.json").write_text(
            json.dumps(dstate, indent=1))
        log(f"driver: {msg}")

    if phase in ("A", "primary"):
        for rid in SSL_ORDER:
            _drive_one("ssl", rid)
            heartbeat(f"phase A: {rid} complete")
        n = sum(1 for rid in SSL_ORDER
                if read_state(run_dir("ssl", rid))["status"] == "COMPLETE")
        assert n == 9, f"phase A incomplete: {n}/9"
        aggregate_ssl()
        heartbeat("phase A complete (9/9) + aggregation written")
    if phase in ("B", "primary"):
        for rid in SSL_ORDER:
            st = read_state(run_dir("ssl", rid))
            assert st and st["status"] == "COMPLETE", \
                f"phase B blocked: {rid} not COMPLETE"
        for rid in DS_ORDER:
            _drive_one("downstream", rid)
            heartbeat(f"phase B: {rid} complete")
        aggregate_primary()
        heartbeat("phase B complete (18/18) + primary analysis written")
    heartbeat(f"DRIVER DONE phase={phase}")


def status() -> None:
    rows = []
    for kind, order in (("ssl", SSL_ORDER), ("downstream", DS_ORDER)):
        for rid in order:
            st = read_state(run_dir(kind, rid))
            rows.append({"kind": kind, "run_id": rid,
                         "status": st["status"] if st else "REGISTERED",
                         "best_epoch": (st or {}).get("best_epoch")})
    print(pd.DataFrame(rows).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("gate")
    p = sub.add_parser("run")
    p.add_argument("--run-id", required=True)
    p.add_argument("--resume", action="store_true")
    p = sub.add_parser("drive")
    p.add_argument("--phase", choices=("A", "B", "primary"),
                   required=True)
    sub.add_parser("aggregate-ssl")
    sub.add_parser("aggregate-primary")
    sub.add_parser("status")
    a = ap.parse_args()
    if a.cmd == "gate":
        gate()
    elif a.cmd == "run":
        if a.run_id.startswith("ssl_"):
            run_ssl(a.run_id, resume=a.resume)
        else:
            run_downstream(a.run_id, resume=a.resume)
    elif a.cmd == "drive":
        drive(a.phase)
    elif a.cmd == "aggregate-ssl":
        aggregate_ssl()
    elif a.cmd == "aggregate-primary":
        aggregate_primary()
    elif a.cmd == "status":
        status()


if __name__ == "__main__":
    main()
