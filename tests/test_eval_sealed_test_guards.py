"""Guard tests for scripts/eval_sealed_test.py (sealed-test completion session).

Two properties added on 2026-08-02, on top of the frozen-config resolution
remediated at commit 7355d3cf:

1. --dry-run must never read the sealed split: it prints the file paths and
   bearing IDs it WOULD load and exits before load_sealed().
2. Arm-B encoders are resolved from the frozen configuration's pinned_encoders
   and sha256-verified BEFORE any sealed data is read; operator-supplied
   --encoder-checkpoint paths are accepted only if they hash-match the pins.

Nothing here trains, evaluates, or reads a sealed file: load_sealed is
monkeypatched to raise, and the development loader is stubbed where main()
is exercised.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import eval_sealed_test as est  # noqa: E402

SEALED = ["K005", "K006", "KA08", "KA09", "KA30", "KI08", "KI18", "KI21"]


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# resolve_arm_b_encoders — pinned-encoder enforcement (unit)
# ---------------------------------------------------------------------------

@pytest.fixture()
def pinned(tmp_path):
    out = {}
    for seed in (42, 43):
        fp = tmp_path / f"enc_{seed}.pt"
        fp.write_bytes(f"encoder-{seed}".encode())
        out[str(seed)] = {"path": str(fp), "sha256": _sha(fp)}
    return out


def test_default_resolution_uses_pinned_paths(pinned, capsys):
    enc, protocol = est.resolve_arm_b_encoders(pinned, None, [42, 43])
    assert "pinned_encoders" in protocol
    assert enc[42].endswith("enc_42.pt") and enc[43].endswith("enc_43.pt")
    assert "sha256 matches the pinned freeze" in capsys.readouterr().out


def test_tampered_pinned_file_fails_fast(pinned):
    Path(pinned["43"]["path"]).write_bytes(b"tampered")
    with pytest.raises(SystemExit, match="sha256 mismatch for seed 43"):
        est.resolve_arm_b_encoders(pinned, None, [42, 43])


def test_supplied_path_not_matching_pin_refuses(pinned, tmp_path):
    rogue = tmp_path / "rogue.pt"
    rogue.write_bytes(b"not the pinned encoder")
    with pytest.raises(SystemExit, match="sha256 mismatch"):
        est.resolve_arm_b_encoders(
            pinned, [str(rogue), pinned["43"]["path"]], [42, 43])


def test_supplied_paths_matching_pins_are_accepted(pinned):
    enc, protocol = est.resolve_arm_b_encoders(
        pinned, [pinned["42"]["path"], pinned["43"]["path"]], [42, 43])
    assert protocol == "one_encoder_per_seed (pre-registered)"
    assert set(enc) == {42, 43}


def test_single_encoder_reuse_cannot_defeat_pins(pinned):
    with pytest.raises(SystemExit, match="sha256 mismatch"):
        est.resolve_arm_b_encoders(pinned, [pinned["42"]["path"]], [42, 43])


def test_seed_without_pin_refuses(pinned):
    with pytest.raises(SystemExit, match="pins no encoder for seed"):
        est.resolve_arm_b_encoders(pinned, None, [42, 43, 44])


def test_no_pins_and_no_supplied_refuses():
    with pytest.raises(SystemExit, match="--encoder-checkpoint"):
        est.resolve_arm_b_encoders(None, None, [42])


def test_supplied_count_mismatch_refuses(pinned):
    with pytest.raises(SystemExit, match="pass 1 or 2 paths"):
        est.resolve_arm_b_encoders(pinned, ["a", "b", "c"], [42, 43])


def test_missing_pinned_file_refuses(pinned):
    Path(pinned["42"]["path"]).unlink()
    with pytest.raises(SystemExit, match="not found"):
        est.resolve_arm_b_encoders(pinned, None, [42, 43])


def test_repo_frozen_pins_resolve_and_verify():
    """The ACTUAL freeze must resolve exactly as the real run will at check 1b."""
    sel = json.loads(
        (REPO_ROOT / "docs/project_review/sealed_test_selected_configs.json")
        .read_text())
    pins = sel["arms"]["B"]["pinned_encoders"]
    enc, protocol = est.resolve_arm_b_encoders(pins, None, [42, 43, 44])
    assert len(enc) == 3 and "pinned_encoders" in protocol


# ---------------------------------------------------------------------------
# main() — sealed-data isolation and guard ordering
# ---------------------------------------------------------------------------

def _forbid_sealed(monkeypatch, called):
    def _boom():
        called["sealed"] = True
        raise AssertionError("sealed split was read")
    monkeypatch.setattr(est, "load_sealed", _boom)


def test_dry_run_never_reads_sealed_split(monkeypatch, capsys):
    called = {"sealed": False}
    _forbid_sealed(monkeypatch, called)
    monkeypatch.setattr(
        est, "load_development_data",
        lambda root: (np.zeros((4, 16), dtype=np.float32),
                      np.zeros(4, dtype=np.int64),
                      np.array(["K001", "K002", "KA04", "KI04"])))
    monkeypatch.setattr(sys, "argv", ["eval_sealed_test.py", "--dry-run"])
    est.main()
    out = capsys.readouterr().out
    assert called["sealed"] is False
    assert "[dry-run] sealed files that WOULD be loaded" in out
    assert "[dry-run] sealed bearings that WOULD be evaluated" in out
    assert "DRY RUN — nothing was evaluated" in out
    assert "sha256_match=True" in out          # pinned encoders shown per seed
    for b in SEALED:
        assert b in out


def test_real_run_verifies_encoders_before_sealed_read(monkeypatch, tmp_path):
    """A wrong encoder must abort BEFORE load_sealed() is reached."""
    called = {"sealed": False}
    _forbid_sealed(monkeypatch, called)
    rogue = tmp_path / "rogue.pt"
    rogue.write_bytes(b"wrong encoder")
    monkeypatch.setattr(sys, "argv", [
        "eval_sealed_test.py", "--arm", "B", "--prereg-committed",
        "--encoder-checkpoint", str(rogue)])
    with pytest.raises(SystemExit, match="sha256 mismatch"):
        est.main()
    assert called["sealed"] is False


# ---------------------------------------------------------------------------
# Training-loop / dataset interface (the arm-A crash of 2026-08-02)
# ---------------------------------------------------------------------------

def test_loader_yields_two_tuples_matching_training_loop():
    """make_cv_loader_arrays batches are (x, y) pairs — the interface the
    runner's training loop must unpack (it crashed unpacking three)."""
    from src.cv_paderborn import make_cv_loader_arrays
    X = np.zeros((8, 32), dtype=np.float32)
    y = np.zeros(8, dtype=np.int64)
    b = np.array(["K001"] * 8)
    batch = next(iter(make_cv_loader_arrays(X, y, b, batch_size=4)))
    assert len(batch) == 2
    xb, yb = batch                       # the runner's unpack pattern
    assert xb.shape[0] == 4 and yb.shape[0] == 4


# ---------------------------------------------------------------------------
# assert_no_significance_language — restored after being lost at 7355d3cf
# ---------------------------------------------------------------------------

def test_significance_guard_accepts_clean_payload():
    est.assert_no_significance_language(
        {"bb_f1_mean": 0.5, "per_seed": [{"seed": 42, "bb_f1": 0.5}],
         "statistical_rule": "reported descriptively; no significance test is "
                             "performed or implied."})


def test_significance_guard_refuses_forbidden_language():
    with pytest.raises(SystemExit, match="significance language"):
        est.assert_no_significance_language(
            {"conclusion": "arm A is statistically better (Wilcoxon p=0.03)"})


def test_significance_guard_exempts_quoted_frozen_provenance():
    """The frozen justification/conflict text (committed pre-run) legitimately
    names Wilcoxon; the guard must allow it there but nowhere else."""
    quoted = "Wilcoxon signed-rank floor is p=0.25; statistically indistinguishable"
    est.assert_no_significance_language(
        {"frozen_selection": {"conflict_with_preregistration":
                              {"explanation": quoted}},
         "statistical_rule_justification": quoted})
    with pytest.raises(SystemExit, match="significance language"):
        est.assert_no_significance_language({"per_seed_note": quoted})


# ---------------------------------------------------------------------------
# rehearsal mode — roster properties
# ---------------------------------------------------------------------------

def test_rehearsal_roster_is_deterministic_dev_only():
    from src.cv_paderborn import DEVELOPMENT_POOL
    r1, r2 = est.rehearsal_roster(), est.rehearsal_roster()
    assert r1 == r2 and len(r1) > 0
    assert set(r1) <= set(DEVELOPMENT_POOL)
    assert not set(r1) & set(SEALED)


def test_rehearsal_rejects_non_rehearsal_out_dir(monkeypatch):
    """A NON-default, non-rehearsal out-dir must be refused outright.

    (Passing the real default explicitly is indistinguishable from omitting it,
    so that case is silently redirected to the rehearsal default instead.)"""
    monkeypatch.setattr(sys, "argv", [
        "eval_sealed_test.py", "--arm", "A", "--prereg-committed",
        "--rehearsal", "--out-dir", "results/some_other_dir"])
    with pytest.raises(SystemExit, match="containing 'rehearsal'"):
        est.main()
