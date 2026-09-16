"""
smoke_joint_ssl_paderborn.py — smoke tests A–D for the joint CWRU+Paderborn
SSL pretraining script (scripts/train_joint_ssl_paderborn.py).

Each smoke test runs 2 epochs / 2 batches using --smoke.  Tests verify that
each of the three conditions (cwru, joint_balanced, joint_natural) executes
end-to-end, produces the expected outputs, and that condition-specific
invariants hold (e.g. the CWRU-only encoder is shared across a repeat's
folds; joint runs record correct window counts and sampling proportions).

Run:
  python tests/smoke_joint_ssl_paderborn.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

GREEN, RED, RESET = "\033[32m", "\033[31m", "\033[0m"
PASS, FAIL = f"{GREEN}PASS{RESET}", f"{RED}FAIL{RESET}"

SCRIPT = str(Path(__file__).resolve().parents[1] / "scripts" / "train_joint_ssl_paderborn.py")
PYTHON = sys.executable

n_pass = n_fail = 0


def _run(label: str, argv: list[str], extra_checks=None) -> None:
    global n_pass, n_fail
    out_dir = Path(tempfile.mkdtemp(prefix="smoke_joint_"))
    cmd = [PYTHON, SCRIPT, "--smoke", "--out-dir", str(out_dir)] + argv
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        ok = proc.returncode == 0
        if ok:
            for rf in ("config.json", "provenance.json", "fold_manifest.json", "aggregate.json"):
                if not (out_dir / rf).exists():
                    ok = False
                    break
        if ok and extra_checks:
            for check_fn in extra_checks:
                if not check_fn(out_dir, proc):
                    ok = False
                    break
        if ok:
            n_pass += 1
            print(f"  {label:65s} {PASS}")
        else:
            n_fail += 1
            tail = (proc.stdout + proc.stderr)[-600:]
            print(f"  {label:65s} {FAIL}")
            if tail:
                print(f"      {tail[-400:]}")
    except subprocess.TimeoutExpired:
        n_fail += 1
        print(f"  {label:65s} {FAIL}  TIMEOUT")
    except Exception as e:
        n_fail += 1
        print(f"  {label:65s} {FAIL}  {e}")
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


def _check_fold_outputs(out_dir: Path, proc) -> bool:
    for f in range(4):
        fold_dir = out_dir / "repeat_0" / f"fold_{f}"
        for fname in ("fold_config.json", "epoch_history.json", "fold_summary.json", "checkpoint.pt"):
            if not (fold_dir / fname).exists():
                return False
    return True


def _check_bb_f1_in_range(out_dir: Path, proc) -> bool:
    agg = json.loads((out_dir / "aggregate.json").read_text())
    return 0.0 <= agg.get("bb_f1_mean", -1) <= 1.0


def _check_cwru_encoder_shared_across_folds(out_dir: Path, proc) -> bool:
    """CWRU-only condition: exactly one cwru_ssl_encoder.pt per repeat, and
    every fold's fold_config records the SAME ssl_meta seed/best_epoch
    (proving the encoder was pretrained once, not once per fold)."""
    rep_dir = out_dir / "repeat_0"
    if not (rep_dir / "cwru_ssl_encoder.pt").exists():
        return False
    if not (rep_dir / "cwru_ssl_history.json").exists():
        return False
    seeds = set()
    for f in range(4):
        fc = json.loads((rep_dir / f"fold_{f}" / "fold_config.json").read_text())
        meta = fc.get("ssl_meta", {})
        if meta.get("condition") != "cwru":
            return False
        seeds.add(meta.get("seed"))
    return len(seeds) == 1   # identical across all 4 folds -> shared encoder


def _check_joint_meta(condition: str, expect_balanced_fraction: bool):
    def _fn(out_dir: Path, proc) -> bool:
        for f in range(4):
            fold_dir = out_dir / "repeat_0" / f"fold_{f}"
            jh = fold_dir / "joint_ssl_history.json"
            if not jh.exists():
                return False
            d = json.loads(jh.read_text())
            if d.get("condition") != condition:
                return False
            if d.get("n_cwru_train_windows", 0) <= 0:
                return False
            if d.get("n_paderborn_ssl_train_windows", 0) <= 0:
                return False
            if not d.get("paderborn_ssl_val_bearings"):
                return False
            if expect_balanced_fraction and abs(d.get("realised_cwru_fraction", -1) - 0.5) > 1e-9:
                return False
            fc = json.loads((fold_dir / "fold_config.json").read_text())
            if fc.get("ssl_meta", {}).get("condition") != condition:
                return False
        return True
    return _fn


print("\n" + "=" * 68)
print("  Joint SSL smoke tests  (A–D)")
print("=" * 68)

_run(
    "A  cwru-only SSL linear probe (encoder shared across folds)",
    ["--condition", "cwru", "--regime", "ssl_linear"],
    extra_checks=[_check_fold_outputs, _check_bb_f1_in_range,
                  _check_cwru_encoder_shared_across_folds],
)

_run(
    "B  joint_balanced SSL full fine-tune (encoder-lr-scale 0.1)",
    ["--condition", "joint_balanced", "--regime", "ssl_full", "--encoder-lr-scale", "0.1"],
    extra_checks=[_check_fold_outputs, _check_bb_f1_in_range,
                  _check_joint_meta("joint_balanced", expect_balanced_fraction=True)],
)

_run(
    "C  joint_natural SSL full fine-tune (encoder-lr-scale 0.1)",
    ["--condition", "joint_natural", "--regime", "ssl_full", "--encoder-lr-scale", "0.1"],
    extra_checks=[_check_fold_outputs, _check_bb_f1_in_range,
                  _check_joint_meta("joint_natural", expect_balanced_fraction=False)],
)

_run(
    "D  cwru-only SSL partial fine-tune (lower-priority regime, sanity only)",
    ["--condition", "cwru", "--regime", "ssl_partial"],
    extra_checks=[_check_fold_outputs, _check_bb_f1_in_range],
)

print()
sep = "=" * 68
if n_fail == 0:
    print(f"{sep}\n  {GREEN}ALL {n_pass} SMOKE TESTS PASSED{RESET}\n{sep}")
else:
    print(f"{sep}\n  {RED}{n_fail} SMOKE TEST(S) FAILED{RESET}  ({n_pass} passed)\n{sep}")
sys.exit(0 if n_fail == 0 else 1)
