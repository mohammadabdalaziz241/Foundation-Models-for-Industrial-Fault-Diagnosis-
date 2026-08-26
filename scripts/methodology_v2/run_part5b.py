#!/usr/bin/env python
"""methodology_v2 Part 5B — PC-STE implementation verification runner.

Fail-closed on Part-2/3B/4C seals. Builds the frozen-config encoder,
verifies geometry on REAL Part-4C representations, runs the mask/
coordinate/cross-band/gradient/determinism battery, audits compute, and
writes the sealed architecture spec + hash. No training of any kind.
"""
from __future__ import annotations

import datetime as dt
import hashlib
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
from src.methodology_v2.encoder import (PCSTE, PCSTEConfig,  # noqa: E402
                                        collate_representations)
from src.methodology_v2.registry import REPO_ROOT  # noqa: E402

PART5B_DIR = REPO_ROOT / "methodology_v2" / "part5_encoder"
DATASETS = ("CWRU", "JNU", "HIT", "MAFAULDA")


def real_items():
    """One fold-1 TRAIN representation per dataset (train-only signal
    access; frozen preprocessing)."""
    man = pd.read_csv(PART3B_DIR / "window_manifest_fold_1.csv")
    items, metas = [], []
    for ds in DATASETS:
        w = man[(man["dataset"] == ds)
                & (man["split"] == "train")].sort_values("window_id").iloc[0]
        x, meta = get_representation(w["window_id"], 1)
        items.append((x, meta["frequency_hz"], meta["time_seconds"]))
        metas.append(meta)
    return items, metas


def geometry_tables(model, items, metas):
    batch = collate_representations(items)
    out = model(**batch)
    geo, rows = [], []
    for i, (ds, meta) in enumerate(zip(DATASETS, metas)):
        bins, frames = items[i][0].shape
        n_bands = int(out["band_mask"][i].sum())
        n_tokens = int(out["token_mask"][i].sum())
        f_khz = out["band_freq_khz"][i][out["band_mask"][i]]
        geo.append({
            "dataset": ds, "orig_bins": bins, "orig_frames": frames,
            "batch_padded_shape": f"({batch['spec'].shape[1]}, "
                                  f"{batch['spec'].shape[2]})",
            "valid_bands": n_bands, "valid_tokens": n_tokens,
            "freq_pad_bins_last_band": 16 - bins % 16 if bins % 16 else 0,
            "time_pad_frames": 0 if frames % 8 == 0 else 8 - frames % 8,
            "pct_padding_in_own_grid": round(
                100 * (n_bands * 16 * (n_tokens // n_bands) * 8
                       - bins * frames)
                / (n_bands * 16 * (n_tokens // n_bands) * 8), 2),
            "max_real_freq_khz": round(float(items[i][1].max()) / 1000, 3),
            "max_band_centre_khz": round(float(f_khz.max()), 3),
        })
        rows.append({
            "dataset": ds, "bands": n_bands,
            "time_patches": n_tokens // n_bands,
            "tokens": n_tokens,
            "band_seq_len": n_tokens // n_bands})
    return pd.DataFrame(geo), pd.DataFrame(rows), batch, out


def verification_battery(model, items) -> dict:
    res = {}
    batch = collate_representations(items)
    base = model(**batch)["global_embedding"].detach()

    # mask invariance: junk in every padded cell
    b2 = {k: (v.clone() if torch.is_tensor(v) else v)
          for k, v in batch.items()}
    junk = torch.full_like(b2["spec"], 1e4)
    b2["spec"] = torch.where(b2["cell_mask"], b2["spec"], junk)
    delta = (model(**b2)["global_embedding"].detach() - base).abs().max()
    res["mask_invariance_max_abs_delta"] = float(delta)

    # coordinate sensitivity: same values, shifted absolute frequencies
    b3 = {k: (v.clone() if torch.is_tensor(v) else v)
          for k, v in batch.items()}
    b3["frequency_hz"] = b3["frequency_hz"] + 5_000.0
    delta_c = (model(**b3)["global_embedding"].detach() - base).abs().max()
    res["coordinate_sensitivity_max_abs_delta"] = float(delta_c)

    # cross-band dependency: perturb ONLY band 5 of sample 0, observe
    # mixed summary of band 0
    with torch.no_grad():
        out_a = model(**batch)
        b4 = {k: (v.clone() if torch.is_tensor(v) else v)
              for k, v in batch.items()}
        b4["spec"][0, 5 * 16:(6 * 16), :] += 3.0
        out_b = model(**b4)
    d_other = (out_b["mixed_band_summaries"][0, 0]
               - out_a["mixed_band_summaries"][0, 0]).abs().max()
    d_pre = (out_b["band_summaries"][0, 0]
             - out_a["band_summaries"][0, 0]).abs().max()
    res["crossband_effect_on_band0_after_mixer"] = float(d_other)
    res["crossband_effect_on_band0_before_mixer"] = float(d_pre)

    # gradient flow
    model.zero_grad()
    out = model(**batch)
    out["global_embedding"].sum().backward()
    for name, mod in (("patch_stem", model.stem),
                      ("coordinate_encoder", model.coords),
                      ("temporal_backbone", model.temporal),
                      ("cross_band_mixer", model.mixer)):
        g = [p.grad.abs().max().item() for p in mod.parameters()
             if p.grad is not None]
        res[f"grad_max_{name}"] = float(max(g)) if g else 0.0
    model.zero_grad()

    # determinism
    torch.manual_seed(123)
    m1 = PCSTE()
    torch.manual_seed(123)
    m2 = PCSTE()
    same_init = all(torch.equal(a, b) for (_, a), (_, b)
                    in zip(m1.state_dict().items(),
                           m2.state_dict().items()))
    y1 = m1(**batch)["global_embedding"]
    y2 = m1(**batch)["global_embedding"]
    res["init_deterministic"] = bool(same_init)
    res["forward_deterministic_cpu"] = bool(torch.equal(y1, y2))
    return res


def compute_audit(model, items) -> pd.DataFrame:
    rows = []
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = model.to(dev)
    for bs in (8, 16, 32, 64):
        reps = (items * ((bs + len(items) - 1) // len(items)))[:bs]
        batch = collate_representations(reps)
        batch = {k: v.to(dev) for k, v in batch.items()}
        if dev == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            m(**batch)
        if dev == "cuda":
            torch.cuda.synchronize()
        dt_ms = 1000 * (time.perf_counter() - t0)
        rows.append({
            "device": dev, "batch_size": bs,
            "forward_ms": round(dt_ms, 1),
            "peak_forward_mem_mb": (round(
                torch.cuda.max_memory_allocated() / 2**20, 1)
                if dev == "cuda" else None),
            "note": "forward/no-grad only; training overhead NOT "
                    "extrapolated from this",
        })
    model.to("cpu")
    return pd.DataFrame(rows)


def main() -> None:
    start = dt.datetime.now(dt.timezone.utc)
    verify_frozen_hashes(); verify_part3b_hashes(); verify_part4c_hashes()
    print("Part-2, 3B, 4C seals verified (pre)")
    PART5B_DIR.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    model = PCSTE()
    breakdown = model.parameter_breakdown()
    tr = PCSTE(PCSTEConfig(backbone="transformer"))
    pd.DataFrame([
        {"component": k, "params": v} for k, v in breakdown.items()
    ] + [{"component": "TOTAL_transformer_ablation",
          "params": tr.parameter_breakdown()["total"]}]).to_csv(
        PART5B_DIR / "encoder_parameter_breakdown.csv", index=False)
    print("params:", breakdown,
          "| transformer swap:", tr.parameter_breakdown()["total"])

    items, metas = real_items()
    geo, tok, batch, out = geometry_tables(model, items, metas)
    geo.to_csv(PART5B_DIR / "patch_geometry_verified.csv", index=False)
    tok.to_csv(PART5B_DIR / "batching_geometry.csv", index=False)
    print(geo.to_string(index=False))

    battery = verification_battery(model, items)
    print("battery:", battery)
    assert battery["mask_invariance_max_abs_delta"] < 1e-5
    assert battery["coordinate_sensitivity_max_abs_delta"] > 1e-4
    assert battery["crossband_effect_on_band0_after_mixer"] > 1e-6
    assert battery["crossband_effect_on_band0_before_mixer"] < 1e-7

    audit = compute_audit(model, items)
    audit.to_csv(PART5B_DIR / "forward_compute_audit.csv", index=False)
    print(audit.to_string(index=False))

    # ---- spec, ablation registry, architecture hash ---------------------
    part4c_master = pd.read_csv(
        REPO_ROOT / "methodology_v2" / "part4_representation_final"
        / "normalizer_hashes.csv").set_index("file").loc[
        "PART4C_MASTER_REPRESENTATION_HASH", "sha256"]
    spec = {
        "architecture": model.cfg.to_dict(),
        "bidirectional_design": "per layer: x + 0.5*(MambaFwd(LN(x)) + "
                                "flip(MambaBwd(flip(LN(x)))))",
        "mixer_equations": "a_j=w2^T tanh(W1[h_j;phi(f_j)]); "
                           "alpha=masked_softmax(a); c=sum alpha_j V h_j; "
                           "g_i=sigmoid(G[h_i;phi(f_i);c]); "
                           "h'_i=h_i+g_i*W_c c",
        "coordinate_features": "8 log-spaced wavelengths per coordinate; "
                               "f in [0.1,51.2] kHz, t in [0.02,2.56] s; "
                               "sin+cos -> 32 dims -> Linear(32,192); "
                               "phi(f)=16-dim freq-only features",
        "mamba_implementation": "reference-faithful pure-PyTorch "
                                "Mamba-1 selective scan (official "
                                "parameterization; mamba_ssm CUDA "
                                "package unbuildable on torch "
                                "2.12.0+cu130 — disclosed)",
        "supported_shapes": {"CWRU": [513, 184], "JNU": [513, 192],
                             "HIT": [257, 192], "MAFAULDA": [513, 192]},
        "parameter_breakdown": breakdown,
        "upstream_part4c_hash": part4c_master,
        "s0_s1_rule": "identical encoder class/config for S0 and S1; "
                      "only pretraining history may differ",
    }
    with open(PART5B_DIR / "pcste_encoder_spec.yaml", "w") as f:
        json.dump(spec, f, indent=1, sort_keys=True)  # json is valid yaml

    arch_hash = hashlib.sha256(
        json.dumps(spec, sort_keys=True).encode()).hexdigest()
    (PART5B_DIR / "part5b_architecture_hash.txt").write_text(
        arch_hash + "\n")
    print("PART5B architecture hash:", arch_hash)

    ablations = {
        "A1_coordinates": {"question": "does physical calibration matter",
                           "config": "PCSTEConfig(use_coordinates=False)"},
        "A2_mixer": {"question": "does cross-band interaction matter",
                     "config": "PCSTEConfig(use_mixer=False)"},
        "A3_backbone": {"question": "does the SSM backbone matter",
                        "config": "PCSTEConfig(backbone='transformer')",
                        "params_transformer":
                            tr.parameter_breakdown()["total"],
                        "params_bimamba": breakdown["total"]},
        "A4_normalization": {"question": "does N2 matter vs N1",
                             "layer": "representation (Part 4C registry); "
                                      "NOT fitted here"},
    }
    with open(PART5B_DIR / "ablation_registry.yaml", "w") as f:
        json.dump(ablations, f, indent=1, sort_keys=True)

    def git(*a):
        return subprocess.run(["git", *a], cwd=REPO_ROOT, text=True,
                              capture_output=True).stdout.strip()
    repro = {
        "stage": "methodology_v2 Part 5B PC-STE implementation",
        "started_utc": start.isoformat(),
        "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "python": sys.version, "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": (torch.cuda.get_device_name(0)
                if torch.cuda.is_available() else None),
        "git": {"commit": git("rev-parse", "HEAD"),
                "branch": git("rev-parse", "--abbrev-ref", "HEAD")},
        "checkpoint_commit_part4c_5a":
            "d9e47dbd84cf74f88911b792bdda2284b3bd5e84",
        "seals_verified": ["part2", "part3b", "part4c"],
        "architecture_hash": arch_hash,
        "verification_battery": battery,
        "mamba_dependency_note": "official mamba-ssm/causal-conv1d "
            "unbuildable against torch 2.12.0+cu130; reference "
            "selective-scan implementation used (disclosed, "
            "architecture unchanged)",
    }
    with open(PART5B_DIR / "part5b_reproducibility.json", "w") as f:
        json.dump(repro, f, indent=1)

    verify_frozen_hashes(); verify_part3b_hashes(); verify_part4c_hashes()
    print("seals re-verified (post); Part 5B verification complete")


if __name__ == "__main__":
    main()
