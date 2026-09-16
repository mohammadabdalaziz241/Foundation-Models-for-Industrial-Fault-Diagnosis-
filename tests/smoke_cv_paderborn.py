"""
smoke_cv_paderborn.py — Smoke tests A–M for the Paderborn GroupCV framework.

Each smoke test runs 2 epochs / 2 batches using --smoke.  Tests verify that
the pipeline executes without error and produces the expected output files.
I–K cover the E0/E3/E4 additions: seeded reproducibility, per-window
normalisation, and LP warm-up with discriminative learning rates.
L covers the large-RF TCN variant.  M covers the held-out SSL validation
split (--ssl-val-fraction).

Run:
  python tests/smoke_cv_paderborn.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

GREEN = "\033[32m"
RED   = "\033[31m"
RESET = "\033[0m"
PASS  = f"{GREEN}PASS{RESET}"
FAIL  = f"{RED}FAIL{RESET}"

SCRIPT = str(Path(__file__).resolve().parents[1] / "scripts" / "train_cv_paderborn.py")
PYTHON = sys.executable

n_pass = n_fail = 0


def _run(label: str, argv: list[str], extra_checks=None) -> None:
    global n_pass, n_fail
    out_dir = Path(tempfile.mkdtemp(prefix="smoke_cv_"))
    cmd = [PYTHON, SCRIPT, "--smoke", "--out-dir", str(out_dir)] + argv
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        ok = proc.returncode == 0

        # Check expected output files exist
        req_files = ["config.json", "fold_manifest.json", "aggregate.json"]
        if ok and not argv[:1] == ["--mode"]:  # skip for final mode
            for rf in req_files:
                if not (out_dir / rf).exists():
                    ok = False
                    break

        # Custom checks
        if ok and extra_checks:
            for check_fn in extra_checks:
                if not check_fn(out_dir, proc):
                    ok = False
                    break

        if ok:
            n_pass += 1
            print(f"  {label:55s} {PASS}")
        else:
            n_fail += 1
            tail = (proc.stdout + proc.stderr)[-600:]
            print(f"  {label:55s} {FAIL}")
            if tail:
                print(f"      {tail[-300:]}")
    except subprocess.TimeoutExpired:
        n_fail += 1
        print(f"  {label:55s} {FAIL}  TIMEOUT")
    except Exception as e:
        n_fail += 1
        print(f"  {label:55s} {FAIL}  {e}")
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


def _check_aggregate_keys(expected_keys: list[str]):
    def _fn(out_dir, proc):
        agg_path = out_dir / "aggregate.json"
        if not agg_path.exists():
            return False
        agg = json.loads(agg_path.read_text())
        return all(k in agg for k in expected_keys)
    return _fn


def _check_fold_outputs(out_dir: Path, proc) -> bool:
    """Verify that all 4 fold directories contain required outputs."""
    for r in range(1):
        for f in range(4):
            fold_dir = out_dir / f"repeat_{r}" / f"fold_{f}"
            for fname in ["fold_config.json", "epoch_history.json",
                          "fold_summary.json", "checkpoint.pt"]:
                if not (fold_dir / fname).exists():
                    return False
    return True


def _check_bb_f1_in_range(out_dir: Path, proc) -> bool:
    """Verify bb_f1_mean is in [0, 1]."""
    agg_path = out_dir / "aggregate.json"
    if not agg_path.exists():
        return False
    agg = json.loads(agg_path.read_text())
    v = agg.get("bb_f1_mean", -1)
    return 0.0 <= v <= 1.0


def _check_no_test_access(out_dir: Path, proc) -> bool:
    """In final mode, verify test result file is present."""
    return (out_dir / "final_result.json").exists()


def _check_final_epoch_used(final_epochs: int):
    def _fn(out_dir: Path, proc) -> bool:
        fr = out_dir / "final_result.json"
        if not fr.exists():
            return False
        d = json.loads(fr.read_text())
        return d.get("final_epochs") == final_epochs
    return _fn


def _check_fold_norm_stats(policy: str):
    """
    Verify fold_summary.json records the normalisation policy and stats.

    fold_mean / fold_std are the *fitted* statistics in stored space.  For the
    fold_train policy they are genuine fold-train values (not the identity
    sentinel); for historical_fixed they are the (0.0, 1.0) identity sentinel.
    """
    def _fn(out_dir: Path, proc) -> bool:
        for r in range(1):
            for f in range(4):
                fs = out_dir / f"repeat_{r}" / f"fold_{f}" / "fold_summary.json"
                if not fs.exists():
                    return False
                d = json.loads(fs.read_text())
                if d.get("normalisation_policy") != policy:
                    return False
                if "fold_mean" not in d or "fold_std" not in d:
                    return False
                if "n_fold_train_samples" not in d or d["n_fold_train_samples"] <= 0:
                    return False
                if policy == "historical_fixed":
                    if d["fold_mean"] != 0.0 or d["fold_std"] != 1.0:
                        return False
                else:  # fold_train — a genuine fitted std must be positive
                    if not (d["fold_std"] > 0.0):
                        return False
        return True
    return _fn


def _check_metadata_and_balance(out_dir: Path, proc) -> bool:
    """Verify fold_metadata.json and balance_report.json exist with note fields."""
    meta = out_dir / "fold_metadata.json"
    bal  = out_dir / "balance_report.json"
    if not meta.exists() or not bal.exists():
        return False
    rows = json.loads(meta.read_text())
    if not rows:
        return False
    required = {"fold_role", "bearing_id", "class_name", "damage_origin",
                "fault_type", "note", "original_split"}
    if not required <= set(rows[0].keys()):
        return False
    b = json.loads(bal.read_text())
    return "repeats" in b and "warnings" in b


def _check_pred_csv(out_dir: Path, proc) -> bool:
    """Verify val_predictions.csv exists and has required columns."""
    for r in range(1):
        for f in range(4):
            pred_csv = out_dir / f"repeat_{r}" / f"fold_{f}" / "val_predictions.csv"
            if not pred_csv.exists():
                return False
            header = pred_csv.read_text().splitlines()[0] if pred_csv.stat().st_size > 0 else ""
            if "bearing_id" not in header or "true_label" not in header:
                return False
    return True


# ---------------------------------------------------------------------------
# Print header
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("  Paderborn GroupCV smoke tests  (A–M)")
print("=" * 60)

# ---------------------------------------------------------------------------
# A — LSTM supervised four-fold CV
# ---------------------------------------------------------------------------
_run(
    "A  LSTM supervised 4-fold CV",
    ["--model", "lstm", "--regime", "supervised"],
    extra_checks=[
        _check_aggregate_keys(["bb_f1_mean", "recommended_final_epochs",
                               "fold_summaries", "per_class_f1_means"]),
        _check_fold_outputs,
        _check_bb_f1_in_range,
        _check_pred_csv,
        _check_fold_norm_stats("fold_train"),
        _check_metadata_and_balance,
    ],
)

# ---------------------------------------------------------------------------
# B — LSTM SSL-full four-fold CV
# ---------------------------------------------------------------------------
_run(
    "B  LSTM SSL-full 4-fold CV",
    ["--model", "lstm", "--regime", "ssl_full"],
    extra_checks=[
        _check_fold_outputs,
        _check_bb_f1_in_range,
    ],
)

# ---------------------------------------------------------------------------
# C — TCN supervised four-fold CV
# ---------------------------------------------------------------------------
_run(
    "C  TCN supervised 4-fold CV",
    ["--model", "tcn", "--regime", "supervised"],
    extra_checks=[
        _check_fold_outputs,
        _check_bb_f1_in_range,
    ],
)

# ---------------------------------------------------------------------------
# D — TCN SSL-full four-fold CV
# ---------------------------------------------------------------------------
_run(
    "D  TCN SSL-full 4-fold CV",
    ["--model", "tcn", "--regime", "ssl_full"],
    extra_checks=[
        _check_fold_outputs,
        _check_bb_f1_in_range,
    ],
)

# ---------------------------------------------------------------------------
# E — Adam with gradient clipping
# ---------------------------------------------------------------------------
_run(
    "E  Adam with gradient clipping (--grad-clip 1.0)",
    ["--model", "lstm", "--regime", "supervised",
     "--optimizer", "adam", "--grad-clip", "1.0"],
    extra_checks=[
        _check_fold_outputs,
        lambda d, p: json.loads((d / "config.json").read_text()).get("grad_clip") == 1.0,
    ],
)

# ---------------------------------------------------------------------------
# F — AdamW with weight decay and clipping
# ---------------------------------------------------------------------------
_run(
    "F  AdamW weight-decay=0.01 + grad-clip=1.0",
    ["--model", "lstm", "--regime", "supervised",
     "--optimizer", "adamw", "--weight-decay", "0.01", "--grad-clip", "1.0"],
    extra_checks=[
        _check_fold_outputs,
        lambda d, p: (
            json.loads((d / "config.json").read_text()).get("optimizer") == "adamw" and
            abs(json.loads((d / "config.json").read_text()).get("weight_decay", -1) - 0.01) < 1e-9
        ),
    ],
)

# ---------------------------------------------------------------------------
# G — Final-training dry run: test access only in final mode
# ---------------------------------------------------------------------------
_run(
    "G  Final-mode dry run (test access only in --mode final)",
    ["--mode", "final", "--model", "lstm", "--regime", "supervised",
     "--final-epochs", "2"],
    extra_checks=[
        _check_no_test_access,
        _check_final_epoch_used(2),
    ],
)

# ---------------------------------------------------------------------------
# H — historical_fixed normalisation + separate SSL/adapt LRs
# ---------------------------------------------------------------------------
_run(
    "H  historical_fixed norm + --ssl-lr/--adapt-lr",
    ["--model", "lstm", "--regime", "ssl_full",
     "--normalisation-policy", "historical_fixed",
     "--ssl-lr", "5e-4", "--adapt-lr", "3e-4"],
    extra_checks=[
        _check_fold_outputs,
        _check_bb_f1_in_range,
        _check_fold_norm_stats("historical_fixed"),
        lambda d, p: (
            json.loads((d / "config.json").read_text()).get("normalisation_policy")
            == "historical_fixed" and
            abs(json.loads((d / "config.json").read_text()).get("ssl_lr") - 5e-4) < 1e-12 and
            abs(json.loads((d / "config.json").read_text()).get("adapt_lr") - 3e-4) < 1e-12
        ),
    ],
)

# ---------------------------------------------------------------------------
# I — Seeded reproducibility: two identical runs give identical histories
# ---------------------------------------------------------------------------

def _run_repro_pair(label: str) -> None:
    global n_pass, n_fail
    dirs = [Path(tempfile.mkdtemp(prefix="smoke_cv_")) for _ in range(2)]
    try:
        hists, seeds = [], []
        for d in dirs:
            cmd = [PYTHON, SCRIPT, "--smoke", "--out-dir", str(d),
                   "--model", "lstm", "--regime", "supervised",
                   "--train-seed", "7"]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if proc.returncode != 0:
                raise RuntimeError((proc.stdout + proc.stderr)[-400:])
            run_h, run_s = [], []
            for f in range(4):
                fd = d / "repeat_0" / f"fold_{f}"
                h = json.loads((fd / "epoch_history.json").read_text())
                run_h.append([(e["train_loss"], e["val_bb_f1"]) for e in h])
                run_s.append(json.loads(
                    (fd / "fold_summary.json").read_text()).get("fold_seed"))
            hists.append(run_h)
            seeds.append(run_s)
        prov_ok = all((d / "provenance.json").exists() for d in dirs)
        seeds_ok = (seeds[0] == seeds[1]
                    and len(set(seeds[0])) == 4
                    and all(s is not None for s in seeds[0]))
        ok = hists[0] == hists[1] and seeds_ok and prov_ok
        if ok:
            n_pass += 1
            print(f"  {label:55s} {PASS}")
        else:
            n_fail += 1
            print(f"  {label:55s} {FAIL}")
            print(f"      identical={hists[0] == hists[1]} seeds_ok={seeds_ok} "
                  f"provenance={prov_ok}")
    except Exception as e:
        n_fail += 1
        print(f"  {label:55s} {FAIL}  {e}")
    finally:
        for d in dirs:
            shutil.rmtree(d, ignore_errors=True)


_run_repro_pair("I  Same --train-seed → bit-identical fold histories")

# ---------------------------------------------------------------------------
# J — per_window normalisation policy
# ---------------------------------------------------------------------------
_run(
    "J  per_window normalisation policy",
    ["--model", "lstm", "--regime", "supervised",
     "--normalisation-policy", "per_window"],
    extra_checks=[
        _check_fold_outputs,
        _check_bb_f1_in_range,
        _check_fold_norm_stats("per_window"),
    ],
)

# ---------------------------------------------------------------------------
# K — SSL-full with LP warm-up + discriminative encoder LR
# ---------------------------------------------------------------------------

def _check_lp_warmup(out_dir: Path, proc) -> bool:
    for f in range(4):
        fd = out_dir / "repeat_0" / f"fold_{f}"
        fc = json.loads((fd / "fold_config.json").read_text())
        if fc.get("lp_warmup_epochs") != 1:
            return False
        if abs(fc.get("encoder_lr_scale", -1) - 0.1) > 1e-12:
            return False
        h = json.loads((fd / "epoch_history.json").read_text())
        if not (h[0].get("encoder_frozen") is True
                and h[1].get("encoder_frozen") is False):
            return False
    return True


_run(
    "K  ssl_full + --lp-warmup-epochs 1 + --encoder-lr-scale 0.1",
    ["--model", "lstm", "--regime", "ssl_full",
     "--lp-warmup-epochs", "1", "--encoder-lr-scale", "0.1"],
    extra_checks=[
        _check_fold_outputs,
        _check_bb_f1_in_range,
        _check_lp_warmup,
    ],
)

# ---------------------------------------------------------------------------
# L — Large-receptive-field TCN (tcn_rf1021) supervised CV
# ---------------------------------------------------------------------------
_run(
    "L  tcn_rf1021 (RF 1021) supervised 4-fold CV",
    ["--model", "tcn_rf1021", "--regime", "supervised"],
    extra_checks=[
        _check_fold_outputs,
        _check_bb_f1_in_range,
        lambda d, p: json.loads((d / "config.json").read_text()).get("model")
        == "tcn_rf1021",
    ],
)

# ---------------------------------------------------------------------------
# M — ssl_linear with a genuinely held-out SSL validation split
# ---------------------------------------------------------------------------

def _check_ssl_val_split(out_dir: Path, proc) -> bool:
    for f in range(4):
        fd = out_dir / "repeat_0" / f"fold_{f}"
        fc = json.loads((fd / "fold_config.json").read_text())
        if abs(fc.get("ssl_val_fraction", -1) - 0.3) > 1e-12:
            return False
        tr, va = fc.get("ssl_train_bearings"), fc.get("ssl_val_bearings")
        if not tr or not va:
            return False
        if set(tr) & set(va):
            return False
        sh = json.loads((fd / "ssl_history.json").read_text())
        if sh.get("ssl_val_bearings") != va:
            return False
        if "val_recon_smoothed" not in sh["history"][0]:
            return False
    return True


_run(
    "M  ssl_linear + --ssl-val-fraction 0.3 (held-out SSL checkpoint selection)",
    ["--model", "lstm", "--regime", "ssl_linear", "--ssl-val-fraction", "0.3"],
    extra_checks=[
        _check_fold_outputs,
        _check_bb_f1_in_range,
        _check_ssl_val_split,
    ],
)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print()
sep = "=" * 60
if n_fail == 0:
    print(f"{sep}\n  {GREEN}ALL {n_pass} SMOKE TESTS PASSED{RESET}\n{sep}")
else:
    print(f"{sep}\n  {RED}{n_fail} SMOKE TEST(S) FAILED{RESET}  ({n_pass} passed)\n{sep}")

sys.exit(0 if n_fail == 0 else 1)
