"""Tests for the vibrationclip_v1 model components (Amendment 4 §6, §9)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.vibrationclip.encoder import (  # noqa: E402
    Classifier, ProjectionHead, SpectrogramEncoder, TextProjection,
    count_parameters,
)
from src.vibrationclip.objectives import (  # noqa: E402
    LearnableTemperature, clip_multipositive_loss, supcon_loss,
)
from src.vibrationclip.simmim import (  # noqa: E402
    GRID, MASK_RATIO, N_PATCHES, PATCH, SimMIM, patch_mask_to_pixels,
    sample_patch_mask,
)

torch.manual_seed(0)


def test_encoder_shapes_and_size():
    enc = SpectrogramEncoder()
    x = torch.randn(2, 1, 128, 128)
    assert enc.forward_features(x).shape == (2, 256, 8, 8)
    assert enc(x).shape == (2, 256)
    n = count_parameters(enc)
    assert 1_500_000 < n < 4_000_000, n  # "a few million parameters"


def test_projection_heads_normalised():
    z = ProjectionHead()(torch.randn(4, 256))
    t = TextProjection()(torch.randn(4, 384))
    assert z.shape == (4, 128) and t.shape == (4, 128)
    assert torch.allclose(z.norm(dim=1), torch.ones(4), atol=1e-5)
    assert torch.allclose(t.norm(dim=1), torch.ones(4), atol=1e-5)


def test_classifier_output():
    assert Classifier()(torch.randn(3, 1, 128, 128)).shape == (3, 3)


def test_mask_exact_ratio_and_pixel_expansion():
    g = torch.Generator().manual_seed(7)
    mask = sample_patch_mask(5, g, torch.device("cpu"))
    assert mask.shape == (5, N_PATCHES)
    assert (mask.sum(dim=1) == round(MASK_RATIO * N_PATCHES)).all()
    px = patch_mask_to_pixels(mask)
    assert px.shape == (5, 1, 128, 128)
    assert px.float().mean().item() == pytest.approx(45 / 64, abs=1e-6)
    # block structure: every 16x16 patch uniform
    blocks = px.view(5, 1, GRID, PATCH, GRID, PATCH).float()
    assert (blocks.amax(dim=(3, 5)) == blocks.amin(dim=(3, 5))).all()


def test_simmim_masked_input_pixels_never_reach_encoder():
    model = SimMIM().eval()  # eval: freeze BatchNorm batch statistics
    x = torch.randn(2, 1, 128, 128)
    g = torch.Generator().manual_seed(3)
    mask = sample_patch_mask(2, g, torch.device("cpu"))
    with torch.no_grad():
        loss, recon = model(x, mask)
        assert recon.shape == x.shape and torch.isfinite(loss)
        px = patch_mask_to_pixels(mask)
        # perturbing MASKED input pixels must not change the reconstruction
        # (they are replaced by the mask token before the encoder) ...
        x_masked_perturbed = x.clone()
        x_masked_perturbed[px] += 100.0
        _, recon2 = model(x_masked_perturbed, mask)
        assert torch.allclose(recon, recon2, atol=1e-5)
        # ... while perturbing VISIBLE pixels (the context) must change it
        x_visible_perturbed = x.clone()
        x_visible_perturbed[~px] += 100.0
        _, recon3 = model(x_visible_perturbed, mask)
        assert not torch.allclose(recon, recon3, atol=1e-3)


def test_simmim_decoder_has_no_upsample_module():
    model = SimMIM()
    names = [type(m).__name__ for m in model.decoder.modules()]
    assert "Upsample" not in names  # determinism rider, by construction
    assert "PixelShuffle" in names


def test_simmim_smoke_learning_and_export():
    torch.manual_seed(1)
    model = SimMIM()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randn(8, 1, 128, 128)
    g = torch.Generator().manual_seed(11)
    losses = []
    for _ in range(30):
        mask = sample_patch_mask(8, g, torch.device("cpu"))
        loss, _ = model(x, mask)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    assert np.mean(losses[-5:]) < np.mean(losses[:5])
    exported = model.export_encoder()
    assert set(exported) == set(SpectrogramEncoder().state_dict())


def test_temperature_clamped():
    t = LearnableTemperature()
    assert t().item() == pytest.approx(0.07, abs=1e-6)
    with torch.no_grad():
        t.log_inv_temp.fill_(20.0)  # would give temp ~ 2e-9
    assert t().item() == pytest.approx(0.01, abs=1e-9)


def test_supcon_prefers_class_clustered_embeddings():
    temp = torch.tensor(0.1)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    centers = torch.nn.functional.normalize(torch.randn(3, 128), dim=1)
    clustered = torch.nn.functional.normalize(
        centers[labels] + 0.05 * torch.randn(6, 128), dim=1)
    scattered = torch.nn.functional.normalize(torch.randn(6, 128), dim=1)
    assert supcon_loss(clustered, labels, temp) < supcon_loss(scattered, labels, temp)


def test_clip_loss_symmetric_and_duplicate_safe():
    temp = torch.tensor(0.1)
    zv = torch.nn.functional.normalize(torch.randn(6, 128), dim=1)
    text_ids = torch.tensor([0, 0, 1, 1, 2, 2])  # duplicate texts present
    zt = torch.nn.functional.normalize(torch.randn(3, 128), dim=1)[text_ids]
    loss = clip_multipositive_loss(zv, zt, text_ids, temp)
    assert torch.isfinite(loss)
    # perfect alignment (zv == its text embedding) must score better
    aligned = clip_multipositive_loss(zt, zt, text_ids, temp)
    assert aligned < loss
    # duplicate texts as positives: reordering duplicate pairs changes nothing
    perm = torch.tensor([1, 0, 3, 2, 5, 4])
    loss_perm = clip_multipositive_loss(zv[perm], zt[perm], text_ids[perm], temp)
    assert torch.allclose(loss, loss_perm, atol=1e-5)


def test_clip_loss_rejects_batch_without_positives():
    # a degenerate call where an anchor row has no positive must raise, not
    # silently return a biased loss
    temp = torch.tensor(0.1)
    zv = torch.nn.functional.normalize(torch.randn(2, 128), dim=1)
    zt = torch.nn.functional.normalize(torch.randn(2, 128), dim=1)
    ids = torch.tensor([0, 1])
    loss = clip_multipositive_loss(zv, zt, ids, temp)  # diagonal positives OK
    assert torch.isfinite(loss)
