"""
smoke_stage0_d.py — end-to-end smoke test for the Stage 0 + Stage D changes.

Runs in seconds on CPU against tiny SYNTHETIC datasets written to a temp dir.
It trains nothing meaningful; it checks that every new code path executes, that
the leakage gate fires when it should, and that the aggregator no longer
fabricates an n.

Run from the project root:
    python3 tests/smoke_stage0_d.py

Every check prints PASS or raises.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PY = sys.executable
WINDOW_LEN = 1024


def _run(cmd: list[str], expect_fail: bool = False) -> subprocess.CompletedProcess:
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    ok = (r.returncode != 0) if expect_fail else (r.returncode == 0)
    if not ok:
        print("─" * 70)
        print("COMMAND:", " ".join(cmd))
        print("STDOUT:\n", r.stdout[-3000:])
        print("STDERR:\n", r.stderr[-3000:])
        raise AssertionError(
            f"expected {'failure' if expect_fail else 'success'}, "
            f"got returncode={r.returncode}"
        )
    return r


# ---------------------------------------------------------------------------
# Synthetic datasets
# ---------------------------------------------------------------------------

def make_cwru(root: Path, rng) -> None:
    """Tiny CWRU-shaped dataset: windows + labels + metadata.csv (recording ids)."""
    import pandas as pd

    for split, recs in {
        "train": ["Normal_0HP", "IR007_0_DE12k", "B007_0_DE12k", "OR007@6_0_DE12k"],
        "val":   ["Normal_2HP", "IR007_2_DE12k", "B007_2_DE12k", "OR007@6_2_DE12k"],
        "test":  ["Normal_3HP", "IR007_3_DE12k", "B007_3_DE12k", "OR007@6_3_DE12k"],
    }.items():
        d = root / split
        d.mkdir(parents=True, exist_ok=True)
        n_per = 24
        X = rng.standard_normal((n_per * len(recs), WINDOW_LEN)).astype(np.float32)
        y = np.repeat(np.arange(len(recs)), n_per).astype(np.int64)
        rows = []
        for lbl, rec in enumerate(recs):
            for i in range(n_per):
                rows.append({"recording_id": rec, "fault_type": rec, "label": lbl,
                             "load_hp": 0, "rpm": 1797.0, "window_idx": i})
        np.save(d / "windows.npy", X)
        np.save(d / "labels.npy", y)
        pd.DataFrame(rows).to_csv(d / "metadata.csv", index=False)
    (root / "pipeline_config.json").write_text(json.dumps({
        "channel": "DE_time", "window_len": WINDOW_LEN, "overlap": 0.5,
        "strategy": "by_load",
    }, indent=2))


def make_paderborn(root: Path, rng, leak: bool = False) -> None:
    """
    Tiny Paderborn-shaped dataset: windows + labels + bearings.npy + config.

    leak=True deliberately writes a TEST bearing into the train split, to prove
    the SSL leakage gate fires.
    """
    split_map = {
        "train": [("K001", 0), ("K002", 0), ("KI01", 1), ("KI03", 1),
                  ("KA01", 2), ("KA03", 2)],
        "val":   [("K004", 0), ("KI07", 1), ("KA07", 2)],
        "test":  [("K005", 0), ("KI08", 1), ("KA08", 2)],
    }
    if leak:
        split_map["train"] = split_map["train"] + [("KA08", 2)]   # a TEST bearing!

    for split, bearings in split_map.items():
        d = root / split
        d.mkdir(parents=True, exist_ok=True)
        n_per = 20
        X, y, g = [], [], []
        for code, lbl in bearings:
            X.append(rng.standard_normal((n_per, WINDOW_LEN)).astype(np.float32))
            y.append(np.full(n_per, lbl, dtype=np.int64))
            g.append(np.full(n_per, code, dtype="<U6"))
        np.save(d / "windows.npy", np.concatenate(X))
        np.save(d / "labels.npy", np.concatenate(y))
        np.save(d / "bearings.npy", np.concatenate(g))

    # The config always records the TRUE split — so a leaked window in train/ is
    # detectable by comparing bearings.npy against it.
    true_map = {
        "train": ["K001", "K002", "KI01", "KI03", "KA01", "KA03"],
        "val":   ["K004", "KI07", "KA07"],
        "test":  ["K005", "KI08", "KA08"],
    }
    (root / "pipeline_config.json").write_text(json.dumps({
        "dataset": "paderborn", "condition": "N15_M07_F10", "strategy": "by_bearing",
        "window_len": WINDOW_LEN, "overlap": 0.5, "target_fs": 12_000,
        "n_classes": 3, "class_names": ["Normal", "Inner Race", "Outer Race"],
        "splits": {s: {"bearings": b} for s, b in true_map.items()},
    }, indent=2))


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def main() -> None:
    rng = np.random.default_rng(0)
    tmp = Path(tempfile.mkdtemp(prefix="smoke_"))
    print(f"scratch dir: {tmp}\n")

    cwru = tmp / "processed"
    pu = tmp / "processed_paderborn"
    pu_leak = tmp / "processed_paderborn_LEAKY"
    make_cwru(cwru, rng)
    make_paderborn(pu, rng, leak=False)
    make_paderborn(pu_leak, rng, leak=True)

    models = tmp / "models"
    res_ssl = tmp / "results_ssl"
    res_tr = tmp / "results_transfer"

    # -- 1. repro manifest ---------------------------------------------------
    from src.repro import manifest_digest, provenance, source_manifest
    m = source_manifest(ROOT)
    assert len(m) > 10, "manifest suspiciously small"
    assert manifest_digest(m) == manifest_digest(source_manifest(ROOT)), "digest unstable"
    p = provenance(ROOT, ["x"])
    assert p["git_commit"] is None or isinstance(p["git_commit"], str)
    print(f"PASS  1. repro: {len(m)} files, digest {p['source_digest'][:12]}…, "
          f"git_commit={p['git_commit']}")

    # -- 2. SSL pre-training, source-only, with group-aware val selection -----
    _run([PY, "scripts/pretrain_ssl.py", "--data-roots", str(cwru),
          "--epochs", "2", "--batch-size", "16", "--seed", "1",
          "--ssl-val-frac", "0.25",
          "--save-dir", str(models), "--results-dir", str(res_ssl)])
    recs = sorted(res_ssl.glob("ssl_pretrain_*.json"))
    assert recs, "no SSL run record written"
    rec = json.loads(recs[-1].read_text())
    assert rec["selection_criterion"] == "val_recon_loss"
    assert rec["pretrain_source"] == "source_only"
    assert rec["n_heldout_windows"] > 0 and rec["heldout_groups"]
    assert rec["provenance"]["source_digest"]
    ssl_ckpt = sorted(models.glob("ssl_encoder_*.pt"))[-1]
    print(f"PASS  2. SSL source-only: record written, selection="
          f"{rec['selection_criterion']}, held out {rec['heldout_groups']}")

    # -- 3. SSL target-aware: CWRU + unlabelled PU train bearings ------------
    _run([PY, "scripts/pretrain_ssl.py",
          "--data-roots", f"{cwru},{pu}", "--pretrain-source", "target_aware",
          "--epochs", "2", "--batch-size", "16", "--seed", "1",
          "--ssl-val-frac", "0.2",
          "--save-dir", str(models), "--results-dir", str(res_ssl)])
    rec = json.loads(sorted(res_ssl.glob("ssl_pretrain_*.json"))[-1].read_text())
    assert rec["pretrain_source"] == "target_aware"
    assert {s["dataset"] for s in rec["sources"]} == {"cwru", "paderborn"}
    assert all(s["labels_used"] is False for s in rec["sources"])
    print("PASS  3. SSL target-aware: CWRU + PU train bearings, no labels used")

    # -- 4. THE LEAKAGE GATE MUST FIRE ---------------------------------------
    r = _run([PY, "scripts/pretrain_ssl.py", "--data-roots", str(pu_leak),
              "--epochs", "1", "--batch-size", "16",
              "--save-dir", str(models), "--results-dir", str(res_ssl)],
             expect_fail=True)
    combined = r.stdout + r.stderr
    assert "SSL LEAKAGE" in combined, "leakage gate did not fire!"
    assert "KA08" in combined, "leakage message did not name the offending bearing"
    print("PASS  4. leakage gate fires on a test bearing in the SSL corpus (KA08)")

    # -- 5. a fake supervised CWRU checkpoint for the transfer control --------
    from src.models.cnn1d import CNN1D
    import torch
    sup_ckpt = models / "cnn1d_supervised_smoke.pt"
    torch.save(CNN1D(num_classes=4).state_dict(), sup_ckpt)

    # -- 6. transfer: OLD defaults must still run (backward compatibility) ----
    _run([PY, "scripts/finetune_transfer.py", "--condition", "pu_supervised",
          "--data-root", str(pu), "--epochs", "2", "--batch-size", "16",
          "--seed", "1", "--results-dir", str(res_tr), "--save-dir", str(models)])
    j = json.loads(sorted(res_tr.glob("pu_supervised_*.json"))[-1].read_text())
    tm = j["transfer_mechanics"]
    assert tm["encoder_lr"] == tm["head_lr"] == 1e-3, "defaults changed!"
    assert tm["bn_train_mode"] == "update" and tm["head_warmup_epochs"] == 0
    assert j["per_bearing"]["group_accuracy"]["n_groups"] == 3
    assert j["split_bearings"]["test"] == ["K005", "KI08", "KA08"]
    print("PASS  6. transfer defaults unchanged; per-bearing + split IDs recorded")

    # -- 7. Stage D mechanics, all at once, on the critical control -----------
    _run([PY, "scripts/finetune_transfer.py", "--condition", "pu_transfer_sup_full",
          "--sup-checkpoint", str(sup_ckpt), "--data-root", str(pu),
          "--epochs", "8", "--batch-size", "16", "--seed", "1",
          "--encoder-lr", "1e-4", "--head-lr", "1e-3",
          "--head-warmup-epochs", "2", "--gradual-unfreeze", "--unfreeze-every", "2",
          "--bn-policy", "recalibrate", "--bn-train-mode", "freeze",
          "--results-dir", str(res_tr), "--save-dir", str(models)])
    j = json.loads(sorted(res_tr.glob("pu_transfer_sup_full_*.json"))[-1].read_text())
    tm = j["transfer_mechanics"]
    assert tm["encoder_lr"] == 1e-4 and tm["head_lr"] == 1e-3
    assert tm["bn_policy"] == "recalibrate" and tm["bn_calib_batches"] > 0
    assert tm["bn_train_mode"] == "freeze"
    # warm-up: encoder frozen for epochs 1-2, then blocks appear one at a time
    hist = {h["epoch"]: h["encoder_blocks_trainable"] for h in j["history"]}
    assert hist[1] == 0 and hist[2] == 0, f"warm-up did not freeze the encoder: {hist}"
    assert hist[3] == 1, f"first block should unfreeze at epoch 3: {hist}"
    assert hist[5] == 2 and hist[7] == 3, f"gradual unfreeze schedule wrong: {hist}"
    assert hist[3] < hist[7], "encoder never fully unfroze"
    print(f"PASS  7. Stage D: warm-up + gradual unfreeze {[hist[e] for e in sorted(hist)]}, "
          f"BN recalibrated on {tm['bn_calib_batches']} PU train batches")

    # -- 8. frozen probe + BN policy 'reset' ---------------------------------
    _run([PY, "scripts/finetune_transfer.py", "--condition", "pu_transfer_ssl_frozen",
          "--ssl-encoder", str(ssl_ckpt), "--data-root", str(pu),
          "--epochs", "2", "--batch-size", "16", "--seed", "1",
          "--bn-policy", "reset", "--head-lr", "1e-3",
          "--results-dir", str(res_tr), "--save-dir", str(models)])
    j = json.loads(sorted(res_tr.glob("pu_transfer_ssl_frozen_*.json"))[-1].read_text())
    assert j["encoder_trainable"] is False
    assert j["transfer_mechanics"]["bn_policy"] == "reset"
    assert all(h["encoder_blocks_trainable"] == 0 for h in j["history"])
    print("PASS  8. frozen probe stays frozen; bn-policy=reset applied")

    # -- 9. aggregator must NOT fabricate n from a duplicated seed ------------
    dupes = sorted(res_tr.glob("pu_supervised_*.json"))
    shutil.copy(dupes[-1], res_tr / "pu_supervised_20000101_000000.json")  # older ts
    r = _run([PY, "scripts/aggregate_transfer.py", "--results-dir", str(res_tr)])
    assert "WARNING" in r.stdout and "duplicate" in r.stdout.lower(), \
        "duplicate not reported"
    import csv as _csv
    with open(res_tr / "transfer_summary.csv") as f:
        rows = list(_csv.DictReader(f))
    sup = [x for x in rows if x["config"] == "pu_supervised"]
    assert len(sup) == 1 and sup[0]["n_seeds"] == "1" and sup[0]["seeds"] == "1", \
        f"duplicate seed still inflating n: {sup}"
    print("PASS  9. aggregator drops the duplicate; n_seeds=1, not the fabricated 2")

    # -- 10. --strict refuses instead --------------------------------------
    r = _run([PY, "scripts/aggregate_transfer.py", "--results-dir", str(res_tr),
              "--strict"], expect_fail=True)
    assert "DUPLICATE RUNS" in (r.stdout + r.stderr)
    print("PASS 10. --strict refuses to aggregate duplicated runs")

    shutil.rmtree(tmp)
    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
