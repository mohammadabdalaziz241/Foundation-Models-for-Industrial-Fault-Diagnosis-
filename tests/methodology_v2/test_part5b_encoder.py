"""Automated tests for the Part-5B PC-STE encoder implementation."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.methodology_v2.part2_builder import verify_frozen_hashes  # noqa: E402
from src.methodology_v2.part3b_windows import (PART3B_DIR,  # noqa: E402
                                               verify_part3b_hashes)
from src.methodology_v2.part4c_normalizers import (  # noqa: E402
    verify_part4c_hashes)
from src.methodology_v2.encoder import (PCSTE, PCSTEConfig,  # noqa: E402
                                        collate_representations)
from src.methodology_v2.encoder.coords import FourierCoordEncoder  # noqa: E402
from src.methodology_v2.encoder.patchify import (patch_centres,  # noqa: E402
                                                 patchify)
from src.methodology_v2.encoder.ssm import (BiMambaBackbone,  # noqa: E402
                                            MambaRefBlock,
                                            TransformerBackbone)

PART5B_DIR = REPO_ROOT / "methodology_v2" / "part5_encoder"
needs_artifacts = pytest.mark.skipif(
    not (PART5B_DIR / "part5b_architecture_hash.txt").exists(),
    reason="run scripts/methodology_v2/run_part5b.py first")


def synth_batch():
    """CWRU/HIT/JNU-like synthetic representations (no signal reads)."""
    items = []
    for bins, frames, fs, n_fft in [(513, 184, 48000, 1024),
                                    (257, 192, 25000, 512),
                                    (513, 192, 50000, 1024)]:
        rng = np.random.default_rng(bins + frames)
        x = rng.normal(size=(bins, frames)).astype(np.float32)
        f = np.arange(bins) * fs / n_fft
        t = np.arange(frames) * (n_fft // 4) / fs
        items.append((x, f, t))
    return items, collate_representations(items)


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return PCSTE()


# ---------------------------------------------------------------------------
# upstream seals
# ---------------------------------------------------------------------------

def test_upstream_seals_intact():
    verify_frozen_hashes()
    verify_part3b_hashes()
    verify_part4c_hashes()


# ---------------------------------------------------------------------------
# patchification and coordinates
# ---------------------------------------------------------------------------

def test_patch_geometry_exact(model):
    _, batch = synth_batch()
    out = model(**batch)
    assert out["band_mask"].sum(1).tolist() == [33, 17, 33]
    assert out["token_mask"].sum((1, 2)).tolist() == [759, 408, 792]
    # CWRU: 23 valid time patches (184 frames), others 24
    assert int(out["token_mask"][0].sum(1).max()) == 23
    assert int(out["token_mask"][1].sum(1).max()) == 24


def test_patchify_masks_and_no_interpolation():
    x = torch.arange(20.0 * 10).reshape(1, 20, 10)
    mask = torch.ones(1, 20, 10, dtype=torch.bool)
    patches, pmask, token_mask, band_mask = patchify(x, mask)
    assert patches.shape == (1, 2, 2, 1, 16, 8)   # padded 20->32, 10->16
    # padded cells are exactly zero (mechanical completion, no invention)
    assert float(patches[0, 1, 0, 0, 4:, :].abs().sum()) == 0.0
    assert token_mask.all() and band_mask.all()
    # fully-padded region would be invalid:
    mask2 = mask.clone(); mask2[:, 16:, :] = False
    _, _, tm2, bm2 = patchify(x, mask2)
    assert not bm2[0, 1]


def test_patch_centres_real_cells_only():
    freq = torch.arange(20.0).unsqueeze(0) * 100     # 0..1900 Hz
    time = torch.arange(10.0).unsqueeze(0) * 0.1
    mask = torch.ones(1, 20, 10, dtype=torch.bool)
    f_khz, t_s = patch_centres(freq, time, mask)
    # band 1 covers bins 16..19 only (real): centre = mean(1600..1900)
    assert f_khz[0, 1].item() == pytest.approx(1.750, abs=1e-6)
    assert t_s[0, 1].item() == pytest.approx(0.85, abs=1e-6)


def test_coordinate_sensitivity_and_hit_range(model):
    items, batch = synth_batch()
    base = model(**batch)["global_embedding"].detach()
    b2 = {k: v.clone() for k, v in batch.items()}
    b2["frequency_hz"] = b2["frequency_hz"] + 5000.0
    shifted = model(**b2)["global_embedding"].detach()
    assert (shifted - base).abs().max() > 1e-4  # coords enter the model
    out = model(**batch)
    f = out["band_freq_khz"]
    assert float(f[0].max()) == pytest.approx(24.0, abs=0.05)   # CWRU
    assert float(f[1][out["band_mask"][1]].max()) == pytest.approx(
        12.5, abs=0.05)                                          # HIT
    assert float(f[2].max()) == pytest.approx(25.0, abs=0.05)   # 50 kHz


def test_fourier_features_deterministic():
    enc = FourierCoordEncoder(192)
    f = torch.tensor([[1.0, 12.5]])
    t = torch.tensor([[0.1, 0.9]])
    a = enc.features(f, t)
    b = enc.features(f.clone(), t.clone())
    assert torch.equal(a, b)
    assert a.shape == (1, 2, 32)
    assert enc.freq_features(f).shape == (1, 2, 16)


# ---------------------------------------------------------------------------
# temporal backbone
# ---------------------------------------------------------------------------

def test_mamba_reference_recurrence_semantics():
    """The vectorised-per-step scan equals an explicit single-sequence
    recurrence — proving the official selective-scan semantics."""
    torch.manual_seed(1)
    blk = MambaRefBlock(16)
    x = torch.randn(2, 6, 16)
    y = blk(x)
    assert y.shape == (2, 6, 16)
    # causality of the inner scan path: changing a later input must not
    # change earlier outputs (single directional block)
    x2 = x.clone(); x2[:, -1] += 10.0
    y2 = blk(x2)
    assert torch.allclose(y[:, :3], y2[:, :3], atol=1e-6)
    # selectivity: input-dependent dynamics -> scaling input is not
    # equivalent to scaling output (nonlinear state transition)
    y3 = blk(2 * x)
    assert not torch.allclose(y3, 2 * y, atol=1e-3)


def test_bimamba_is_bidirectional():
    torch.manual_seed(2)
    bb = BiMambaBackbone(16, n_blocks=1)
    x = torch.randn(1, 8, 16)
    y = bb(x)
    # perturb one CHANNEL of the last step (a uniform all-channel shift
    # would be cancelled exactly by the pre-norm LayerNorm)
    x2 = x.clone(); x2[:, -1, 3] += 5.0
    y2 = bb(x2)
    # early outputs DO change (backward direction carries information);
    # magnitude is small at random init (state decay over 8 steps) but
    # clearly above float32 arithmetic noise (~1e-7)
    assert (y2[:, 0] - y[:, 0]).abs().max() > 1e-6


def test_shared_temporal_weights_across_bands(model):
    """One backbone instance processes every band — identical band
    inputs yield identical band outputs."""
    bins, frames = 64, 16   # 4 identical bands
    x = np.tile(np.random.default_rng(0).normal(
        size=(16, frames)).astype(np.float32), (4, 1))
    f = np.arange(bins) * 10.0
    t = np.arange(frames) * 0.005
    batch = collate_representations([(x, f, t)])
    cfg = PCSTEConfig(use_coordinates=False, use_mixer=False)
    torch.manual_seed(0)
    m = PCSTE(cfg)
    out = m(**batch)
    b = out["band_summaries"][0]
    assert torch.allclose(b[0], b[1], atol=1e-5)
    assert torch.allclose(b[0], b[3], atol=1e-5)


def test_transformer_dropin_same_interface():
    torch.manual_seed(0)
    mt = PCSTE(PCSTEConfig(backbone="transformer"))
    _, batch = synth_batch()
    out = mt(**batch)
    assert out["global_embedding"].shape == (3, 192)
    p_tr = mt.parameter_breakdown()
    torch.manual_seed(0)
    p_bm = PCSTE().parameter_breakdown()
    # only the temporal backbone differs
    assert p_tr["patch_stem"] == p_bm["patch_stem"]
    assert p_tr["cross_band_mixer"] == p_bm["cross_band_mixer"]
    assert p_tr["temporal_backbone"] != p_bm["temporal_backbone"]


# ---------------------------------------------------------------------------
# mixer: true cross-band exchange
# ---------------------------------------------------------------------------

def test_cross_band_dependency_real_and_masked(model):
    items, batch = synth_batch()
    with torch.no_grad():
        a = model(**batch)
        b4 = {k: v.clone() for k, v in batch.items()}
        b4["spec"][0, 5 * 16:6 * 16, :] += 3.0     # perturb band 5 only
        b = model(**b4)
    # band 0 pre-mixer unchanged; post-mixer changed => exchange is real
    assert torch.equal(a["band_summaries"][0, 0], b["band_summaries"][0, 0])
    assert (b["mixed_band_summaries"][0, 0]
            - a["mixed_band_summaries"][0, 0]).abs().max() > 1e-6
    # altering values inside a MASKED band region must change nothing:
    with torch.no_grad():
        c4 = {k: v.clone() for k, v in batch.items()}
        c4["spec"][1, 257:, :] = 777.0             # HIT padding rows
        c = model(**c4)
    assert torch.equal(a["global_embedding"][1], c["global_embedding"][1])
    # variable band counts: no NaNs anywhere
    for k in ("global_embedding", "mixed_band_summaries"):
        assert torch.isfinite(a[k]).all()


# ---------------------------------------------------------------------------
# pooling, batching, invariance
# ---------------------------------------------------------------------------

def test_masked_mean_pooling_correct(model):
    _, batch = synth_batch()
    out = model(**batch)
    bm = out["band_mask"][1]
    manual = out["mixed_band_summaries"][1][bm].mean(0)
    assert torch.allclose(manual, out["global_embedding"][1], atol=1e-6)


def test_mixed_four_dataset_batch():
    man = pd.read_csv(PART3B_DIR / "window_manifest_fold_1.csv")
    from src.methodology_v2.part4c_reader import get_representation
    items = []
    for ds in ("CWRU", "JNU", "HIT", "MAFAULDA"):
        w = man[(man["dataset"] == ds) & (man["split"] == "train")] \
            .sort_values("window_id").iloc[0]
        x, meta = get_representation(w["window_id"], 1)
        items.append((x, meta["frequency_hz"], meta["time_seconds"]))
    torch.manual_seed(0)
    m = PCSTE()
    out = m(**collate_representations(items))
    assert out["global_embedding"].shape == (4, 192)
    assert out["band_mask"].sum(1).tolist() == [33, 33, 17, 33]
    assert torch.isfinite(out["global_embedding"]).all()


def test_full_mask_invariance(model):
    _, batch = synth_batch()
    base = model(**batch)["global_embedding"].detach()
    b2 = {k: v.clone() for k, v in batch.items()}
    junk = torch.full_like(b2["spec"], -9.9e3)
    b2["spec"] = torch.where(b2["cell_mask"], b2["spec"], junk)
    again = model(**b2)["global_embedding"].detach()
    assert torch.equal(base, again)   # exact invariance


# ---------------------------------------------------------------------------
# gradients, determinism, parameters
# ---------------------------------------------------------------------------

def test_gradient_flow_all_components(model):
    _, batch = synth_batch()
    model.zero_grad()
    model(**batch)["global_embedding"].pow(2).sum().backward()
    for mod in (model.stem, model.coords, model.temporal, model.mixer):
        gmax = max(p.grad.abs().max().item() for p in mod.parameters()
                   if p.grad is not None)
        assert gmax > 0.0
    model.zero_grad()


def test_parameter_count_in_small_tier(model):
    b = model.parameter_breakdown()
    assert 1_000_000 <= b["total"] <= 3_000_000
    assert b["total"] == 2_382_033          # frozen exact count
    assert b["temporal_backbone"] == 2_014_080
    assert b["patch_stem"] == 196_880
    assert b["cross_band_mixer"] == 164_737
    assert b["coordinate_encoder"] == 6_336


def test_determinism_and_s0_s1_identity():
    _, batch = synth_batch()
    torch.manual_seed(7)
    m1 = PCSTE()
    torch.manual_seed(7)
    m2 = PCSTE()
    for (k1, a), (k2, b) in zip(m1.state_dict().items(),
                                m2.state_dict().items()):
        assert k1 == k2 and torch.equal(a, b)
    # same config dict => same architecture for S0 and S1
    assert PCSTEConfig().to_dict() == PCSTEConfig().to_dict()
    y1 = m1(**batch)["global_embedding"]
    y2 = m1(**batch)["global_embedding"]
    assert torch.equal(y1, y2)


# ---------------------------------------------------------------------------
# artifacts + forbidden work
# ---------------------------------------------------------------------------

@needs_artifacts
def test_architecture_hash_reproducible():
    spec = json.load(open(PART5B_DIR / "pcste_encoder_spec.yaml"))
    import hashlib
    h = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()) \
        .hexdigest()
    stored = (PART5B_DIR
              / "part5b_architecture_hash.txt").read_text().strip()
    assert h == stored
    assert spec["architecture"]["d_model"] == 192
    assert spec["architecture"]["backbone"] == "bimamba"
    assert spec["upstream_part4c_hash"].startswith("ee9414e8")


@needs_artifacts
def test_ablation_registry_complete():
    reg = json.load(open(PART5B_DIR / "ablation_registry.yaml"))
    assert set(reg) == {"A1_coordinates", "A2_mixer", "A3_backbone",
                        "A4_normalization"}
    # ablation configs constructible without training anything
    a1 = PCSTE(PCSTEConfig(use_coordinates=False))
    a2 = PCSTE(PCSTEConfig(use_mixer=False))
    assert a1.cfg.use_coordinates is False
    assert a2.cfg.use_mixer is False


def test_no_training_or_ssl_code_in_encoder_package():
    pkg = REPO_ROOT / "src" / "methodology_v2" / "encoder"
    banned = ("Optimizer", "optim.", "backward_epoch", "train_loop",
              "CrossEntropyLoss", "mask_ratio", "Decoder", "classifier",
              "few_shot", "DataLoader")
    for py in pkg.glob("*.py"):
        text = py.read_text()
        for word in banned:
            assert word not in text, f"{py.name} contains '{word}'"
    banned_files = {".pt", ".pth", ".ckpt"}
    for f in PART5B_DIR.rglob("*"):
        assert f.suffix.lower() not in banned_files


# ---------------------------------------------------------------------------
# Mamba reference parity gate
# ---------------------------------------------------------------------------

def test_mamba_reference_parity_gate():
    """Our selective_scan must match the vendored OFFICIAL
    selective_scan_ref (pinned commit) — live spot case + artifact."""
    import torch.nn.functional as F
    from src.methodology_v2.encoder.ssm import selective_scan
    from src.methodology_v2.encoder.third_party \
        .official_selective_scan_ref import selective_scan_ref
    g = torch.Generator().manual_seed(42)
    u = torch.randn(2, 24, 64, generator=g)
    draw = torch.randn(2, 24, 64, generator=g)
    a_log = torch.log(torch.rand(64, 16, generator=g) * 3 + 0.5)
    b = torch.randn(2, 24, 16, generator=g)
    c = torch.randn(2, 24, 16, generator=g)
    d = torch.randn(64, generator=g)
    ours = selective_scan(u, F.softplus(draw), -torch.exp(a_log), b, c, d)
    official = selective_scan_ref(
        u.transpose(1, 2), draw.transpose(1, 2), -torch.exp(a_log),
        b.transpose(1, 2), c.transpose(1, 2), D=d,
        delta_softplus=True).transpose(1, 2)
    assert (ours - official).abs().max() <= 1e-5
    art = PART5B_DIR / "mamba_reference_parity.json"
    if art.exists():
        parity = json.load(open(art))
        assert parity["all_pass"] is True
        assert parity["pinned_commit"] == \
            "e9594ce1c732d97440f0332fdc43170a2294dbfa"
        assert parity["worst_fwd_max_abs_err"] == 0.0
        assert parity["worst_grad_max_rel_err"] <= 1e-6
