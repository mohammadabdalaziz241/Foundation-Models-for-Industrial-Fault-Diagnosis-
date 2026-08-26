"""Part 5A — architecture design mathematics and literature tables.

Analysis ONLY: patch-geometry arithmetic on the frozen representation
shapes, parameter-budget estimates from closed-form formulas, and the
machine-readable literature/novelty tables. No model layers are imported
or instantiated anywhere in this module (test-enforced).
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from .part4b_freeze import RATES, TF_CONFIG
from .registry import REPO_ROOT

PART5A_DIR = REPO_ROOT / "methodology_v2" / "part5_architecture_design"

# frozen representation shapes (verified in Part 4C)
SHAPES = {"CWRU": (513, 184), "JNU": (513, 192),
          "HIT": (257, 192), "MAFAULDA": (513, 192)}

PATCH_CANDIDATES = ((8, 8), (16, 8), (16, 16), (32, 8))  # (freq, time)


# ---------------------------------------------------------------------------
# 5A.5 patch geometry
# ---------------------------------------------------------------------------

def patch_geometry() -> pd.DataFrame:
    rows = []
    for ds, (bins, frames) in SHAPES.items():
        n_fft, hop = TF_CONFIG[ds]
        fs = RATES[ds]
        df_hz = fs / n_fft
        hop_ms = 1000 * hop / fs
        for pf, pt in PATCH_CANDIDATES:
            nf, nt = math.ceil(bins / pf), math.ceil(frames / pt)
            pad_f, pad_t = nf * pf - bins, nt * pt - frames
            padded_elems = nf * pf * nt * pt - bins * frames
            rows.append({
                "dataset": ds, "patch_freq_bins": pf,
                "patch_time_frames": pt,
                "patch_freq_width_hz": round(pf * df_hz, 2),
                "patch_time_width_ms": round(pt * hop_ms, 2),
                "n_freq_patches": nf, "n_time_patches": nt,
                "tokens": nf * nt,
                "pad_freq_bins": pad_f, "pad_time_frames": pad_t,
                "padded_value_pct": round(100 * padded_elems
                                          / (nf * pf * nt * pt), 2),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5A.12 parameter budgets (closed-form estimates, documented formulas)
# ---------------------------------------------------------------------------
# Per-block parameter formulas (bias/small terms folded into the constant):
#   Mamba(-1) block, expand=2, d_state=16:   ~6.6 d^2
#   Bidirectional Mamba block (two dirs):    ~13.2 d^2
#   Transformer block (MHSA 4d^2 + MLP 8d^2): ~12 d^2
#   TCN residual block (2 convs, kernel 5):  ~10 d^2
#   Conv patch stem (16x8 patch -> d):        128 d + d^2 (proj + 1 local conv)
#   Fourier coordinate features -> d:         ~64 d (32 freqs x 2 -> linear)
#   Freq-gated mixer (F3):                    ~3 d^2
#   Masked-pool head (none, mean):            0

BLOCK_FORMULAS = {"bimamba": 13.2, "transformer": 12.0, "tcn": 10.0}


def estimate_params(d: int, n_blocks: int, block: str) -> int:
    core = BLOCK_FORMULAS[block] * d * d * n_blocks
    stem = 128 * d + d * d
    coords = 64 * d
    mixer = 3 * d * d
    return int(core + stem + coords + mixer)


TIERS = [
    ("Tiny", 128, 3, "bimamba"),
    ("Small", 192, 4, "bimamba"),
    ("Medium", 256, 6, "bimamba"),
]

TOKENS_16x8 = {"CWRU": 759, "JNU": 792, "HIT": 408, "MAFAULDA": 792}


def parameter_budget() -> pd.DataFrame:
    rows = []
    for name, d, n, block in TIERS:
        p = estimate_params(d, n, block)
        max_tokens = max(TOKENS_16x8.values())
        act_mb = max_tokens * d * (2 * n + 2) * 4 / 2 ** 20  # rough fwd act
        rows.append({
            "tier": name, "embedding_dim": d, "n_blocks": n,
            "block_type": block,
            "estimated_params": p,
            "estimated_params_m": round(p / 1e6, 2),
            "tokens_16x8_max": max_tokens,
            "rough_fwd_activation_mb_per_window": round(act_mb, 2),
            "batch64_activation_mb": round(64 * act_mb, 1),
            "within_target_range": {
                "Tiny": 0.5e6 <= p <= 1.0e6,
                "Small": 1.0e6 <= p <= 3.0e6,
                "Medium": 3.0e6 <= p <= 8.0e6}[name],
        })
    # backbone-swap equivalents at the Small tier (for the ablation)
    for block in ("transformer", "tcn"):
        p = estimate_params(192, 4, block)
        rows.append({
            "tier": f"Small-{block}", "embedding_dim": 192, "n_blocks": 4,
            "block_type": block, "estimated_params": p,
            "estimated_params_m": round(p / 1e6, 2),
            "tokens_16x8_max": max(TOKENS_16x8.values()),
            "rough_fwd_activation_mb_per_window": None,
            "batch64_activation_mb": None,
            "within_target_range": 1.0e6 <= p <= 3.0e6,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# literature registry + novelty stress test (data tables)
# ---------------------------------------------------------------------------

SEARCH_DATE = "2026-08-12"

LITERATURE = [
    # key, citation, venue/year, url, relevance
    ("VibFM", "Learning the Language of Vibration: A Self-Supervised "
     "Transformer Foundation Model for PHM", "PHM Society European "
     "Conference 2026",
     "https://papers.phmsociety.org/index.php/phme/article/view/4912",
     "masked spectrogram vibration foundation model; 16 datasets ~400 h; "
     "128x128 log-STFT standardization + sampling-rate/resolution "
     "conditioning vector; leakage-resistant bearing-level Paderborn eval"),
    ("ECHO", "ECHO: Frequency-aware Hierarchical Encoding for "
     "Variable-length Signals", "arXiv:2508.14689 (v3 2026)",
     "https://arxiv.org/abs/2508.14689",
     "band-split (32-bin sub-bands, count ~ rate); seconds-specified STFT "
     "(25 ms/10 ms); NORMALIZED f/Nyquist positional encoding; per-band "
     "ViT + CLS concat (no cross-band interaction); EAT-style "
     "teacher-student; audio pretraining (AS2M/MTG/FS); evaluated on "
     "CWRU/MAFAULDA k-NN; 5.5M/22M params"),
    ("FISHER", "FISHER: A Foundation Model for Multi-Modal Industrial "
     "Signal Comprehensive Representation", "arXiv:2507.16696",
     "https://arxiv.org/abs/2507.16696",
     "fixed-duration STFT window/hop; predefined-bandwidth sub-bands "
     "processed individually; teacher-student self-distillation; RMIS "
     "benchmark; small models"),
    ("SSAMBA", "SSAMBA: Self-Supervised Audio Representation Learning "
     "with Mamba State Space Model", "IEEE SLT 2024 / arXiv:2405.11831",
     "https://arxiv.org/abs/2405.11831",
     "bidirectional Mamba; SSAST-style joint discriminative+generative "
     "masked-patch pretraining; tiny/small/base; ~92.7% faster and "
     "~95.4% less memory than SSAST (tiny, 22k tokens)"),
    ("AudioMamba", "Audio Mamba: Selective State Spaces for "
     "Self-Supervised Audio Representations",
     "Interspeech 2024 / arXiv:2406.02178",
     "https://arxiv.org/abs/2406.02178",
     "masked log-mel patch SSL with selective SSM; outperforms SSAST "
     "baselines; AudioSet pretraining"),
    ("SepTr", "SepTr: Separable Transformer for Audio Spectrogram "
     "Processing", "Interspeech 2022 / arXiv:2203.09581",
     "https://arxiv.org/abs/2203.09581",
     "sequential axis-separated attention (within-time then "
     "within-frequency); parameters scale linearly with input"),
    ("SpecTNT", "SpecTNT: A Time-Frequency Transformer for Music Audio",
     "ISMIR 2021 / arXiv:2110.09127",
     "https://arxiv.org/abs/2110.09127",
     "transformer-in-transformer: spectral encoder emits per-frame "
     "frequency class token; temporal transformer exchanges them across "
     "time; hierarchical spectral->temporal"),
    ("VibrMamba", "VibrMamba: A lightweight Mamba based fault diagnosis "
     "of rotating machinery using vibration signal", "Measurement 2025",
     "https://www.sciencedirect.com/science/article/abs/pii/"
     "S0263224125002404",
     "supervised lightweight Mamba on 1D vibration; single-task fault "
     "diagnosis (no SSL, no multi-dataset pretraining)"),
    ("OpenMAE", "OpenMAE: Efficient Masked Autoencoder for Vibration "
     "Sensing with Open-domain Data Enrichment", "ACM IMWUT 2025",
     "https://dl.acm.org/doi/10.1145/3729485",
     "MAE pretraining on vibration with open-domain data enrichment; "
     "sensing focus"),
    ("SwinMSSL", "Unsupervised Bearing Fault Diagnosis Using Masked "
     "Self-Supervised Learning and Swin Transformer", "2025",
     "https://www.researchgate.net/publication/395202165",
     "masked SSL + Swin for bearing diagnosis (single-domain)"),
]

SEARCHES = [
    "VibFM self-supervised transformer foundation model prognostics "
    "health management vibration spectrogram masked",
    "ECHO frequency-aware hierarchical encoding variable sampling rate "
    "machine signals arXiv",
    "SSAMBA self-supervised audio representation learning Mamba state "
    "space model parameters patch masked",
    '"Audio Mamba" selective state spaces self-supervised audio '
    "representations Yadav patch ordering parameters",
    "FISHER foundation model industrial signal band-split sub-band "
    "heterogeneous sampling rate arXiv",
    "Mamba bearing fault diagnosis spectrogram state space model "
    "rotating machinery 2024 2025",
    "masked autoencoder vibration signal self-supervised bearing "
    "pretraining cross-dataset foundation 2025",
    "SepTr separable transformer audio spectrogram Ristea SpecTNT "
    "time-frequency transformer music frequency class token",
]


def literature_registry() -> pd.DataFrame:
    return pd.DataFrame(
        [{"key": k, "title": t, "venue_year": v, "url": u,
          "relevance": r, "search_date": SEARCH_DATE}
         for k, t, v, u, r in LITERATURE])


NOVELTY_TESTS = [
    ("First vibration foundation model",
     "VibFM; FISHER", "REJECTED",
     "none — do not claim"),
    ("First masked vibration spectrogram foundation model",
     "VibFM", "REJECTED", "none — do not claim"),
    ("First frequency-aware machine-signal encoder for variable "
     "sampling rates", "ECHO; FISHER", "REJECTED",
     "none — do not claim"),
    ("First self-supervised Mamba spectrogram model",
     "SSAMBA; Audio Mamba", "REJECTED", "none — do not claim"),
    ("First native-rate no-resize vibration representation",
     "FISHER/ECHO seconds-specified STFT achieves equivalent physical "
     "matching", "TOO STRONG",
     "'a physically matched native-rate STFT representation, in line "
     "with recent seconds-specified STFT practice (FISHER, ECHO), "
     "applied without any image resizing'"),
    ("Absolute physical-Hz coordinate conditioning (vs normalized "
     "f/Nyquist)", "ECHO uses normalized f/Nyquist; VibFM uses a "
     "rate/resolution conditioning vector", "POSSIBLY DEFENSIBLE",
     "'in contrast to normalized-frequency positional encodings (ECHO), "
     "tokens carry absolute physical-frequency (Hz) coordinates, "
     "aligning identical mechanical frequency bands across "
     "heterogeneous sampling rates'"),
    ("Frequency-gated cross-band interaction conditioned on absolute "
     "physical frequency", "FISHER and ECHO process sub-bands "
     "independently (CLS concat, no cross-band module)",
     "POSSIBLY DEFENSIBLE",
     "'a lightweight cross-band mixer gated by absolute physical "
     "frequency, absent from existing band-split machine-signal "
     "encoders'"),
    ("Self-supervised SSM encoder pretrained on heterogeneous "
     "rotating-machinery vibration at native rates",
     "SSAMBA/AuM (audio); VibrMamba (supervised vibration); VibFM "
     "(Transformer + resize)", "POSSIBLY DEFENSIBLE",
     "system-level wording: 'to our knowledge, no prior work combines "
     "state-space sequence modelling with self-supervised pretraining "
     "on multiple rotating-machinery vibration datasets at native "
     "sampling rates' — verify again at write-up time"),
    ("Leakage-controlled matched S0/S1 protocol across four "
     "heterogeneous bearing datasets with sealed folds",
     "VibFM applies bearing-level leakage-resistant evaluation on one "
     "downstream benchmark", "DEFENSIBLE",
     "'a fully sealed, hash-frozen leakage-controlled protocol "
     "(specimen/configuration/temporal grouping per dataset) with a "
     "matched-architecture S0/S1 comparison' — methodological claim, "
     "not architectural"),
]


def novelty_stress_test() -> pd.DataFrame:
    return pd.DataFrame(
        [{"candidate_claim": c, "closest_prior_work": p, "verdict": v,
          "safe_revised_wording": w} for c, p, v, w in NOVELTY_TESTS])


def write_all(out_dir: Path | None = None) -> dict:
    out_dir = Path(out_dir) if out_dir else PART5A_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    pg = patch_geometry()
    pg.to_csv(out_dir / "patch_geometry_study.csv", index=False)
    pb = parameter_budget()
    pb.to_csv(out_dir / "parameter_budget_estimates.csv", index=False)
    lit = literature_registry()
    lit.to_csv(out_dir / "literature_registry.csv", index=False)
    nv = novelty_stress_test()
    nv.to_csv(out_dir / "novelty_claim_stress_test.csv", index=False)
    with open(out_dir / "search_terms_log.txt", "w") as f:
        f.write(f"web searches executed on {SEARCH_DATE}:\n")
        for s in SEARCHES:
            f.write(f"- {s}\n")
    return {"patch": pg, "budget": pb, "literature": lit, "novelty": nv}
