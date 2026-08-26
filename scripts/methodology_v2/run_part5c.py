#!/usr/bin/env python
"""methodology_v2 Part 5C — SSL design study runner (no training).

Fail-closed on all upstream seals + the Part-5B architecture hash.
Fold-1 TRAIN dev windows only for signal content. Produces the mask
redundancy/geometry/ratio studies, non-learned baselines, decoder/loss
tables, compute estimates, gradient-path proof, and the proposed SSL
specification.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.methodology_v2.part2_builder import verify_frozen_hashes  # noqa: E402
from src.methodology_v2.part3b_windows import (PART3B_DIR,  # noqa: E402
                                               verify_part3b_hashes)
from src.methodology_v2.part4c_normalizers import (  # noqa: E402
    verify_part4c_hashes)
from src.methodology_v2.part4c_reader import get_representation  # noqa: E402
from src.methodology_v2.part4a_repdesign import (  # noqa: E402
    assert_fold1_train, load_fold1_train)
from src.methodology_v2.encoder import PCSTE, collate_representations  # noqa: E402
from src.methodology_v2.encoder.ssl_design import (  # noqa: E402
    ReconstructionProbe, baseline_mses, generate_mask,
    redundancy_metrics, window_rng)
from src.methodology_v2.registry import REPO_ROOT  # noqa: E402

PART5C_DIR = REPO_ROOT / "methodology_v2" / "part5_ssl_design"
PART5B_DIR = REPO_ROOT / "methodology_v2" / "part5_encoder"
DATASETS = ("CWRU", "JNU", "HIT", "MAFAULDA")
VALID_PATCHES = {"CWRU": 759, "JNU": 792, "HIT": 408, "MAFAULDA": 792}
RATIOS = (0.40, 0.50, 0.60, 0.70, 0.75)
RECOMMENDED = {"geometry": "M1_random", "ratio": 0.60}  # data-driven: see report


def verify_part5b_hash() -> str:
    spec = json.load(open(PART5B_DIR / "pcste_encoder_spec.yaml"))
    h = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    stored = (PART5B_DIR / "part5b_architecture_hash.txt").read_text().strip()
    if h != stored:
        raise AssertionError("Part-5B architecture hash mismatch "
                             "(fail closed)")
    return stored


def dev_windows():
    """Part-4A qualitative dev subset (66 Fold-1 TRAIN windows)."""
    dev = pd.read_csv(REPO_ROOT / "methodology_v2" / "part3_windows"
                      / "../part4_stft_design"
                      / "part4a_development_windows.csv")
    man = load_fold1_train().set_index("window_id")
    out = []
    for wid in dev["window_id"]:
        row = man.loc[wid]
        assert_fold1_train(row)
        x, meta = get_representation(wid, 1)
        out.append({"window_id": wid, "dataset": row["dataset"],
                    "x": x, "meta": meta})
    return out


def token_valid_grid(ds: str) -> np.ndarray:
    fb = {"CWRU": 33, "JNU": 33, "HIT": 17, "MAFAULDA": 33}[ds]
    tp = {"CWRU": 23, "JNU": 24, "HIT": 24, "MAFAULDA": 24}[ds]
    return np.ones((fb, tp), dtype=bool)


def main() -> None:
    start = dt.datetime.now(dt.timezone.utc)
    verify_frozen_hashes(); verify_part3b_hashes(); verify_part4c_hashes()
    arch_hash = verify_part5b_hash()
    parity = json.load(open(PART5B_DIR / "mamba_reference_parity.json"))
    assert parity["all_pass"], "parity gate must pass before Part 5C"
    print(f"seals + Part-5B hash verified ({arch_hash[:12]}…); parity OK")
    PART5C_DIR.mkdir(parents=True, exist_ok=True)

    wins = dev_windows()
    print(f"dev windows: {len(wins)}")

    # ---- 5C.1 redundancy audit -----------------------------------------
    red_rows = []
    for w in wins:
        red_rows.append({"window_id": w["window_id"],
                         "dataset": w["dataset"],
                         **redundancy_metrics(w["x"])})
    red = pd.DataFrame(red_rows)
    red.to_csv(PART5C_DIR / "mask_redundancy_study.csv", index=False)
    print("redundancy (means):",
          red.groupby("dataset")[["temporal_corr_lag1",
                                  "freq_corr_dist1",
                                  "neighbour_interp_residual_ratio"]]
          .mean().round(3).to_dict())

    # ---- 5C.2 geometry study -------------------------------------------
    geo_rows = []
    for geom in ("M1_random", "M2_block", "M3_time_span",
                 "M4_band_span", "M5_mixed"):
        for ds in DATASETS:
            tv = token_valid_grid(ds)
            masked, full_band, full_time = [], 0, 0
            for s in range(20):
                rng = window_rng(1234, s, f"probe_{ds}")
                m = generate_mask(tv, RECOMMENDED["ratio"], geom, rng)
                masked.append(int(m.sum()))
                full_band += int((m.all(axis=1)).sum())
                full_time += int((m.all(axis=0)).sum())
            geo_rows.append({
                "geometry": geom, "dataset": ds,
                "valid_patches": int(tv.sum()),
                "masked_at_60pct": int(np.mean(masked)),
                "block_bands_x_time": str(
                    {"M1_random": (1, 1), "M2_block": (2, 3),
                     "M3_time_span": (1, 4), "M4_band_span": (3, 1),
                     "M5_mixed": "mix"}[geom]),
                "physical_extent": {
                    "M1_random": "0.75-0.78 kHz x 41-43 ms",
                    "M2_block": "1.5-1.56 kHz x 123-128 ms",
                    "M3_time_span": "0.75-0.78 kHz x 164-171 ms",
                    "M4_band_span": "2.25-2.34 kHz x 41-43 ms",
                    "M5_mixed": "mixture"}[geom],
                "fully_masked_bands_per20seeds": full_band,
                "fully_masked_times_per20seeds": full_time,
            })
    pd.DataFrame(geo_rows).to_csv(
        PART5C_DIR / "mask_geometry_options.csv", index=False)

    # ---- 5C.3 ratio study ----------------------------------------------
    ratio_rows = []
    for r in RATIOS:
        for ds in DATASETS:
            v = VALID_PATCHES[ds]
            ratio_rows.append({
                "ratio": r, "dataset": ds, "valid_patches": v,
                "masked_patches": int(round(r * v)),
                "visible_patches": v - int(round(r * v)),
                "masked_physical_area_pct": round(100 * r, 1),
                "relative_compute_cost": 1.0,   # full-sequence encoding
            })
    pd.DataFrame(ratio_rows).to_csv(PART5C_DIR / "mask_ratio_study.csv",
                                    index=False)

    # ---- 5C.14 baselines on recommended config -------------------------
    base_rows = []
    for w in wins:
        tv = token_valid_grid(w["dataset"])
        rng = window_rng(1234, 0, w["window_id"])
        m = generate_mask(tv, RECOMMENDED["ratio"],
                          RECOMMENDED["geometry"], rng)
        base_rows.append({"window_id": w["window_id"],
                          "dataset": w["dataset"],
                          **baseline_mses(w["x"], m)})
    base = pd.DataFrame(base_rows)
    base.to_csv(PART5C_DIR / "baseline_reconstruction_metrics.csv",
                index=False)
    print("baseline MSE (per-dataset means):")
    print(base.groupby("dataset")[["P0_zero", "P1_temporal_neighbour",
                                   "P2_frequency_neighbour"]]
          .mean().round(4).to_string())

    # ---- 5C.7 decoder + loss option tables -----------------------------
    torch.manual_seed(0)
    enc = PCSTE()
    probe = ReconstructionProbe(enc)
    dec_params = probe.decoder_parameter_count()
    enc_params = enc.parameter_breakdown()["total"]
    pd.DataFrame([
        {"option": "D1_per_token_mlp+X1", "params": dec_params,
         "pct_of_encoder": round(100 * dec_params / enc_params, 1),
         "verdict": "RECOMMENDED (simplest adequate; cross-time context "
                    "already provided by encoder + X1 band context)"},
        {"option": "D2_shallow_transformer_1blk_d128",
         "params": 128 * 128 * 12 + 128 * 192 * 2 + 128,
         "pct_of_encoder": round(100 * (128 * 128 * 12 + 128 * 192 * 2)
                                 / enc_params, 1),
         "verdict": "alternative if D1 cannot beat P1 baseline"},
        {"option": "D3_temporal_conv", "params": 5 * 192 * 192 * 2,
         "pct_of_encoder": round(100 * (5 * 192 * 192 * 2) / enc_params,
                                 1),
         "verdict": "not preferred (local RF duplicates encoder work)"},
    ]).to_csv(PART5C_DIR / "decoder_options.csv", index=False)
    pd.DataFrame([
        {"loss": "L1_masked_cell_MSE", "verdict": "RECOMMENDED training "
         "loss AND reporting metric (TA requirement)"},
        {"loss": "L2_masked_cell_SmoothL1", "verdict": "fallback if "
         "outlier residuals destabilise training (decide in 5D from "
         "training curves, not labels)"},
        {"loss": "L3_frequency_balanced_MSE", "verdict": "not needed: N2 "
         "already normalizes every bin to unit TRAIN variance"},
    ]).to_csv(PART5C_DIR / "loss_options.csv", index=False)

    # ---- gradient-path proof (bounded synthetic backward) ---------------
    probe_wins = [next(w for w in wins if w["dataset"] == ds)
                  for ds in DATASETS]      # one per dataset -> (33,24) grid
    items = [(w["x"], w["meta"]["frequency_hz"],
              w["meta"]["time_seconds"])
             for w in probe_wins]
    batch = collate_representations(items)
    pm = torch.zeros(batch["spec"].shape[0], 33, 24, dtype=torch.bool)
    for i, w in enumerate(probe_wins):
        tv = token_valid_grid(w["dataset"])
        m = generate_mask(tv, 0.6, "M2_block",
                          window_rng(1, 0, w["window_id"]))
        pm[i, :m.shape[0], :m.shape[1]] = torch.from_numpy(m)
    probe.zero_grad()
    out = probe(**batch, patch_mask=pm)
    out["loss"].backward()
    mixer_g = max(p.grad.abs().max().item()
                  for p in enc.mixer.parameters() if p.grad is not None)
    grads = {name: max(p.grad.abs().max().item()
                       for p in mod.parameters() if p.grad is not None)
             for name, mod in (("stem", enc.stem), ("coords", enc.coords),
                               ("temporal", enc.temporal),
                               ("mixer", enc.mixer))}
    print("gradient-path proof:", {k: round(v, 4) for k, v in
                                   grads.items()})
    assert mixer_g > 0, "NON-NEGOTIABLE: mixer must receive gradients"

    # ---- 5C.20 compute estimate ----------------------------------------
    man_counts = {}
    for fold in (1, 2, 3):
        man = pd.read_csv(PART3B_DIR / f"window_manifest_fold_{fold}.csv")
        man_counts[fold] = int((man["split"] == "train").sum())
    rows = []
    for bs in (16, 32):
        steps = {f: -(-man_counts[f] // bs) for f in man_counts}
        rows.append({
            "batch_size": bs,
            "train_windows_per_fold": str(man_counts),
            "steps_per_epoch_per_fold": str(steps),
            "meas_fwd_ms_nograd_ref_backend": {16: 90, 32: 236}[bs],
            "est_train_step_ms_x3": {16: 270, 32: 708}[bs],
            "est_epoch_min_per_fold": round(
                steps[1] * {16: 270, 32: 708}[bs] / 60000, 1),
            "est_60epoch_3fold_hours": round(
                3 * 60 * steps[1] * {16: 270, 32: 708}[bs] / 3.6e6, 1),
            "caveat": "reference-backend Python scan timing — NOT "
                      "representative of fused Mamba kernels; treat as "
                      "upper bound",
        })
    pd.DataFrame(rows).to_csv(PART5C_DIR / "ssl_compute_estimate.csv",
                              index=False)
    print(pd.DataFrame(rows)[["batch_size", "est_epoch_min_per_fold",
                              "est_60epoch_3fold_hours"]]
          .to_string(index=False))

    # ---- reproducibility -------------------------------------------------
    def git(*a):
        return subprocess.run(["git", *a], cwd=REPO_ROOT, text=True,
                              capture_output=True).stdout.strip()
    repro = {
        "stage": "methodology_v2 Part 5C SSL design study",
        "started_utc": start.isoformat(),
        "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "python": sys.version, "platform": platform.platform(),
        "torch": torch.__version__,
        "git": {"commit": git("rev-parse", "HEAD"),
                "branch": git("rev-parse", "--abbrev-ref", "HEAD")},
        "part5b_checkpoint_commit":
            "2eb0127ac01c2f2143b74e95391d51c85d0cf1bd",
        "seals_verified": ["part2", "part3b", "part4c",
                           "part5b_architecture_hash"],
        "parity_gate": "PASSED (see part5_encoder/"
                       "mamba_reference_parity.json)",
        "data_access": "Fold-1 TRAIN dev windows only (Part-4A subset)",
        "gradient_path_proof_max_grads": grads,
        "decoder_params": dec_params,
        "decoder_pct_of_encoder": round(100 * dec_params / enc_params, 1),
    }
    with open(PART5C_DIR / "part5c_reproducibility.json", "w") as f:
        json.dump(repro, f, indent=1)

    verify_frozen_hashes(); verify_part3b_hashes(); verify_part4c_hashes()
    verify_part5b_hash()
    print("all seals re-verified (post); Part 5C studies complete")


if __name__ == "__main__":
    main()
