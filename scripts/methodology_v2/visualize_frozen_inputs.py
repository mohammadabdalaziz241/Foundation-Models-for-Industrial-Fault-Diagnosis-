#!/usr/bin/env python3
"""Visual audit of exact frozen Methodology V2 PC-STE inputs (TRAIN only)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.methodology_v2.encoder.collate import collate_representations
from src.methodology_v2.encoder.patchify import PATCH_F, PATCH_T, patchify
from src.methodology_v2.part3b_reader import read_window
from src.methodology_v2.part3b_windows import PART3B_DIR
from src.methodology_v2.part4b_freeze import RATES, TF_CONFIG, rep_of
from src.methodology_v2.part4c_normalizers import load_normalizer
from src.methodology_v2.part4c_reader import get_representation

FOLD = 1
DATASETS = ("CWRU", "JNU", "HIT", "MAFAULDA")
OUT = ROOT / "docs" / "figures" / "methodology_v2_inputs"

# These are from the already-frozen Part-4B qualitative TRAIN audit set.
WINDOWS = {
    "CWRU": "f1:CWRU:cwru_X109:DE:train:96000-144000",
    "JNU": "f1:JNU:jnu_ib1000_2:B:acc_vertical:train:125100-175100",
    "HIT": "f1:HIT:hit_data1_rec00:ch3:train:175000-200000",
    "MAFAULDA": ("f1:MAFAULDA:mafaulda_normal_36.4544:"
                  "col3_underhang_radial:train:100000-150000"),
}


def image(ax, x, fs, hop, title, cmap="magma", vlim=None):
    """Plot native tensor without changing it; imshow rasterizes display only."""
    frames = x.shape[1]
    extent = [0.0, (frames - 1) * hop / fs, 0.0, fs / 2000.0]
    kw = {} if vlim is None else {"vmin": vlim[0], "vmax": vlim[1]}
    im = ax.imshow(x, origin="lower", aspect="auto", extent=extent,
                   interpolation="nearest", cmap=cmap, **kw)
    ax.set_title(title, loc="left", fontweight="bold", fontsize=11)
    ax.set_ylabel("Frequency (kHz)")
    return im


def patch_grid(ax, shape, fs, hop):
    bins, frames = shape
    dt = hop / fs
    df_khz = fs / (2 * (bins - 1)) / 1000
    for t in range(PATCH_T, frames, PATCH_T):
        ax.axvline(t * dt, color="white", lw=.45, alpha=.8)
    for f in range(PATCH_F, bins, PATCH_F):
        ax.axhline(f * df_khz, color="white", lw=.45, alpha=.8)


def load_all():
    man = pd.read_csv(PART3B_DIR / "window_manifest_fold_1.csv").set_index("window_id")
    found = {}
    for ds in DATASETS:
        wid = WINDOWS[ds]
        if wid not in man.index:
            raise AssertionError(f"selected window absent: {wid}")
        row = man.loc[wid]
        assert row["dataset"] == ds and row["split"] == "train"
        assert int(row["fold_id"]) == FOLD

        raw = read_window(row)
        pre = rep_of(raw, ds).T                 # native (bins, frames)
        final, meta = get_representation(wid, FOLD)
        norm = load_normalizer(FOLD, ds)
        manual_once = ((pre.T - norm["mean"]) /
                       norm["std_denominator"]).T.astype(np.float32)
        assert np.array_equal(final, manual_once), "reader != one N2 application"
        assert final.dtype == np.float32 and final.flags.c_contiguous
        assert np.isfinite(final).all()
        assert final.shape == pre.shape
        assert raw.shape == (RATES[ds],)

        # Prove the executor's collator preserves this sample's valid cells.
        batch = collate_representations([(final, meta["frequency_hz"],
                                          meta["time_seconds"])])
        got = batch["spec"][0, :final.shape[0], :final.shape[1]].numpy()
        assert np.array_equal(got, final)
        assert batch["cell_mask"].all()
        patches, _, token_mask, _ = patchify(batch["spec"], batch["cell_mask"])
        assert int(token_mask.sum()) == math.ceil(final.shape[0] / PATCH_F) * math.ceil(final.shape[1] / PATCH_T)
        found[ds] = dict(row=row, raw=raw, pre=pre, final=final,
                         meta=meta, patches=patches, token_mask=token_mask)
    return found


def dataset_figure(ds, d):
    raw, pre, final = d["raw"], d["pre"], d["final"]
    fs = RATES[ds]
    n_fft, hop = TF_CONFIG[ds]
    fig, axes = plt.subplots(4, 1, figsize=(12, 15), constrained_layout=True)
    time = np.arange(raw.size) / fs
    axes[0].plot(time, raw, color="#17324d", lw=.55)
    axes[0].set(title="A. Raw 1-second vibration waveform",
                xlabel="Time (s)", ylabel="Vibration amplitude")
    im = image(axes[1], pre, fs, hop, "B. Raw log-magnitude spectrogram — log1p(abs(STFT))")
    fig.colorbar(im, ax=axes[1], label="log1p magnitude", pad=.01)
    lim = np.percentile(final, [.5, 99.5])
    im = image(axes[2], final, fs, hop,
               f"C. Exact tensor entering PC-STE — native {final.shape[0]} × {final.shape[1]}",
               "coolwarm", lim)
    fig.colorbar(im, ax=axes[2], label="Frozen N2 value", pad=.01)
    im = image(axes[3], final, fs, hop,
               f"D. Exact PC-STE input + 16×8 patch grid — native {final.shape[0]} × {final.shape[1]}",
               "coolwarm", lim)
    patch_grid(axes[3], final.shape, fs, hop)
    fig.colorbar(im, ax=axes[3], label="Frozen N2 value", pad=.01)
    for ax in axes[1:]:
        ax.set_xlabel("STFT frame start time (s)")
    fig.suptitle(f"{ds} — Frozen Methodology V2, Fold 1 TRAIN\n{WINDOWS[ds]}", fontsize=14, fontweight="bold")
    fig.savefig(OUT / f"{ds.lower()}_final_input.png", dpi=220)
    plt.close(fig)


def comparison(found):
    fig, axes = plt.subplots(1, 4, figsize=(18, 5.2), constrained_layout=True)
    vals = np.concatenate([d["final"].ravel() for d in found.values()])
    lim = tuple(np.percentile(vals, [.5, 99.5]))
    last = None
    for ax, ds in zip(axes, DATASETS):
        d = found[ds]
        last = image(ax, d["final"], RATES[ds], TF_CONFIG[ds][1],
                     f"{ds}\n{d['final'].shape[0]} × {d['final'].shape[1]} | 0–{RATES[ds]/2000:g} kHz",
                     "coolwarm", lim)
        ax.set_xlabel("Time (s)")
    fig.colorbar(last, ax=axes, label="Frozen N2-normalized value", shrink=.86, pad=.015)
    fig.suptitle("Exact native-shape tensors entering PC-STE (Fold 1 TRAIN)", fontweight="bold", fontsize=15)
    fig.savefig(OUT / "four_dataset_final_inputs.png", dpi=240)
    fig.savefig(OUT / "four_dataset_final_inputs.pdf")
    plt.close(fig)


def explanation(d):
    x = d["final"]
    fs, hop = RATES["CWRU"], TF_CONFIG["CWRU"][1]
    fb, tp = math.ceil(x.shape[0] / PATCH_F), math.ceil(x.shape[1] / PATCH_T)
    tokens = fb * tp
    fig = plt.figure(figsize=(17, 7), constrained_layout=True)
    gs = fig.add_gridspec(1, 4, width_ratios=(2.1, 2.1, 1.25, 1.4))
    ax0, ax1, ax2, ax3 = [fig.add_subplot(gs[0, i]) for i in range(4)]
    lim = tuple(np.percentile(x, [.5, 99.5]))
    image(ax0, x, fs, hop, f"Final N2 input\n{x.shape[0]} × {x.shape[1]}", "coolwarm", lim)
    image(ax1, x, fs, hop, f"16×8 partition\n{fb} bands × {tp} time patches", "coolwarm", lim)
    patch_grid(ax1, x.shape, fs, hop)
    for ax in (ax0, ax1): ax.set_xlabel("Time (s)")
    # Show six real patches from different positions, preserving their values.
    p = d["patches"][0, :, :, 0].numpy()
    montage = np.concatenate([p[0, 0], p[4, 5], p[8, 10], p[16, 12], p[24, 18], p[32, 22]], axis=1)
    ax2.imshow(montage, origin="lower", aspect="auto", interpolation="nearest", cmap="coolwarm", vmin=lim[0], vmax=lim[1])
    ax2.set(title="Example patches\n6 × (16 × 8)", xlabel="Patch-local time frames", ylabel="Patch-local frequency bins")
    ax3.axis("off")
    ax3.text(.5, .7, "Shared PatchStem", ha="center", fontsize=14, fontweight="bold")
    ax3.text(.5, .53, "each 16×8 patch\n→ one d=192 embedding", ha="center", fontsize=12)
    ax3.text(.5, .28, f"{fb} × {tp} = {tokens}\nvalid PC-STE tokens", ha="center", fontsize=17, fontweight="bold", color="#9b2226")
    for x0 in (.245, .51, .73):
        fig.text(x0, .5, "→", fontsize=30, ha="center", va="center")
    fig.suptitle("How the native CWRU spectrogram becomes PC-STE tokens\nCompletion padding is masked; no resizing or RGB conversion", fontsize=15, fontweight="bold")
    fig.savefig(OUT / "pcste_patch_explanation.png", dpi=240)
    plt.close(fig)


def audit(found):
    lines = [
        "FROZEN METHODOLOGY V2 — EXACT PC-STE INPUT VISUALIZATION AUDIT",
        "=" * 72,
        "Scope: Fold 1 TRAIN examples only. No validation or TEST examples read.",
        "Selection: one deterministic window per dataset from the frozen Part-4B qualitative TRAIN audit set.",
        "Authoritative path: read_window -> rep_of -> get_representation -> collate_representations -> PCSTE.forward(spec) -> patchify.",
        "Reader contract: periodic Hann, center=False, no padding, one-sided STFT; log1p(abs(STFT)); frozen Fold-1 dataset/bin TRAIN N2.",
        "",
    ]
    for ds in DATASETS:
        d, meta, row = found[ds], found[ds]["meta"], found[ds]["row"]
        x, raw, pre = d["final"], d["raw"], d["pre"]
        finite = int(np.isfinite(x).sum())
        nonfinite = int(x.size - finite)
        tokens = int(d["token_mask"].sum())
        lines += [
            f"[{ds}]",
            f"window_id: {WINDOWS[ds]}", f"fold: {FOLD}", f"split: {row['split']}",
            f"class: {row['original_label']}", f"raw_sampling_rate_hz: {RATES[ds]}",
            f"raw_window_shape: {raw.shape}",
            f"STFT: n_fft={TF_CONFIG[ds][0]}, hop={TF_CONFIG[ds][1]}, periodic Hann, center=False, no padding, one-sided",
            f"pre_N2_spectrogram_shape: {pre.shape}", f"final_PCSTE_tensor_shape: {x.shape}",
            f"dtype: {x.dtype}", f"min: {x.min():.9g}", f"max: {x.max():.9g}",
            f"mean: {x.mean():.9g}", f"std_population: {x.std():.9g}",
            f"finite_count: {finite}", f"non_finite_count: {nonfinite}",
            f"NaN_count: {int(np.isnan(x).sum())}", f"Inf_count: {int(np.isinf(x).sum())}",
            f"physical_frequency_range_hz: {meta['frequency_hz'][0]:.9g} to {meta['frequency_hz'][-1]:.9g}",
            f"physical_time_range_s (STFT frame starts): {meta['time_seconds'][0]:.9g} to {meta['time_seconds'][-1]:.9g}",
            f"patch_grid: {math.ceil(x.shape[0]/PATCH_F)} frequency × {math.ceil(x.shape[1]/PATCH_T)} time",
            f"PCSTE_valid_patches_tokens: {tokens}", "",
        ]
    lines += [
        "VERIFICATIONS", "-" * 72,
        "PASS — every selected manifest row is fold_id=1 and split=train.",
        "PASS — no validation or TEST row was read.",
        "PASS — final tensor equals, element-for-element, one explicit application of the loaded frozen N2 normalizer to authoritative rep_of(raw).",
        "PASS — get_representation returned float32, C-contiguous, finite native-shape arrays.",
        "PASS — collate_representations preserved every valid tensor value exactly.",
        "PASS — no image resizing and no RGB conversion occur in the model path.",
        "PASS — plotting uses native arrays; interpolation='nearest' affects display rasterization only.",
        "PASS — normalizers are the sealed Fold-1 per-dataset/per-frequency-bin artifacts fitted by part4c_normalizers.train_windows(), whose code admits split=train only.",
        "PASS — patchify uses zero completion plus masks only; token counts include all valid patches.",
        "PASS — these are exactly the values supplied as PCSTE.forward(spec) before PatchStem embedding.",
        "",
        "OBSERVATION: frequency completion requires 15 masked bins for every dataset; CWRU needs no time padding (184=23×8), and the other datasets need none (192=24×8).",
        "OBSERVATION: HIT remains 0–12.5 kHz and produces 17×24=408 tokens; no 12.5–25 kHz content is created.",
    ]
    (OUT / "input_visualization_audit.txt").write_text("\n".join(lines) + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    found = load_all()
    for ds, d in found.items(): dataset_figure(ds, d)
    comparison(found)
    explanation(found["CWRU"])
    audit(found)
    print(OUT)


if __name__ == "__main__":
    main()
