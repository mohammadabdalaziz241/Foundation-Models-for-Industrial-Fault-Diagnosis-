"""Pre-registered 3-band envelope fallback (env3): the Stage B single-band path must be byte-identical in configuration and
behaviour; the 3-band representation must be deterministic, exactly gain-invariant and physically masked at 12 kHz."""
import hashlib, json, sys
from pathlib import Path
import numpy as np, pandas as pd, pytest, torch
REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO))
from src.pcste_v2.envelope import envelope3_bands_hz, envelope3_order_matrix, envelope_order_vector, EnvelopeBranch, N_ORDER_CELLS, N_ENVELOPE3_BANDS
from src.pcste_v2.model import PCSTEv2, PCSTEv2Config, collate_v2
from src.pcste_v2.representation import ItemBuilder, env3_norm_path
PDIR = REPO / "pcste_v2/protocol/global_v2_PB_sealedMAF_v1"
STAGE_B_ENV_CONFIG_SHA = "5e262c90f435852b54a47ed0b4daf9df4754b5e8b8f3e43852eaa70ce1043a4e"   # pcstev2_b_env_f3_s42 (both attempts)


def stage_b_cfg(variant, fold):
    sys.path.insert(0, str(REPO / "scripts/pcste_v2")); import run as R
    grid_set, envelope, n_channels, control = R.VARIANTS[variant]; env_bands = R.ENVELOPE_BANDS.get(variant, 1); grids = R.GRID_SETS[grid_set]
    cfg = {"run_id": f"pcstev2_b_{variant}_f{fold}_s42", "stage": "b", "variant": variant, "fold": fold, "seed": 42, "protocol": "global_v2_PB_sealedMAF_v1", "grids": list(grids), "envelope_branch": envelope,
           "n_channels": n_channels, "control": control, "level_gain_range": R.LEVEL_GAIN_RANGE if control == "level_shuffle" else None,
           "model": PCSTEv2Config(envelope_branch=envelope, multires=(grid_set == "multires"), n_channels=n_channels, envelope_bands=env_bands).to_dict(),
           **({"envelope_bands": env_bands, "envelope_version": R.ENVELOPE3_VERSION} if env_bands == 3 else {}),
           "optimizer": R.OPTIMIZER_SPEC, "epochs": R.DOWNSTREAM_EPOCHS, "effective_batch": R.EFFECTIVE_BATCH, "micro_batch": R.MICRO_BATCH[grid_set],
           "checkpoint_rule": "max validation MacroDomainF1 (strict >, earlier epoch on ties)", "init": "random (S0)", "label_fraction": 1.0, "test_policy": "TEST never read"}
    return cfg


def test_stage_b_env_config_sha_unchanged():
    cfg = stage_b_cfg("env", 3)
    assert hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest() == STAGE_B_ENV_CONFIG_SHA
    assert "envelope_bands" not in cfg["model"] and "envelope_bands" not in cfg


def test_env3_config_differs_and_declares_bands():
    cfg = stage_b_cfg("env3", 3)
    assert cfg["model"]["envelope_bands"] == 3 and cfg["envelope_bands"] == 3 and cfg["envelope_version"] == "pcste_v2.envelope3.v1"
    assert cfg["optimizer"] == stage_b_cfg("env", 3)["optimizer"] and cfg["epochs"] == 50 and cfg["micro_batch"] == 32 and cfg["effective_batch"] == stage_b_cfg("env", 3)["effective_batch"]


def test_single_band_branch_unchanged():
    torch.manual_seed(0); b1 = EnvelopeBranch(192)
    assert set(b1.state_dict()) == {"stem.weight", "stem.bias", "coords.lam_o", "coords.proj.weight", "coords.proj.bias", "branch_type", "factor_known.weight", "order_c"}
    assert b1.stem.weight.shape == (192, 16)
    b3 = EnvelopeBranch(192, n_bands=3); assert b3.stem.weight.shape == (192, 48) and b3.band_valid_embed.weight.shape == (192, 3)


def test_bands_by_sampling_rate():
    assert envelope3_bands_hz(48000) == [(500.0, 2000.0), (2000.0, 8000.0), (8000.0, 19200.0)]
    assert envelope3_bands_hz(50000) == [(500.0, 2000.0), (2000.0, 8000.0), (8000.0, 20000.0)]
    assert envelope3_bands_hz(25000) == [(500.0, 2000.0), (2000.0, 8000.0), (8000.0, 10000.0)]
    assert envelope3_bands_hz(12000) == [(500.0, 2000.0), (2000.0, 4800.0), None]         # 8 kHz band absent below 0.8*Nyquist


def test_matrix_deterministic_gain_invariant_and_masked():
    rng = np.random.default_rng(1); x = rng.standard_normal(12000) + 0.3 * np.sin(2 * np.pi * 107 * np.arange(12000) / 12000)
    m1, v1 = envelope3_order_matrix(x, 12000, 1797.0); m2, _ = envelope3_order_matrix(x, 12000, 1797.0); m3, v3 = envelope3_order_matrix(4.0 * x, 12000, 1797.0)
    assert m1.shape == (3, N_ORDER_CELLS) and v1.tolist() == [True, True, False] and np.array_equal(m1, m2)
    assert np.allclose(m1, m3, rtol=1e-6, atol=1e-6) and v3.tolist() == v1.tolist() and not m1[2].any()
    assert np.allclose(m1[1], envelope_order_vector(x, 12000, 1797.0, (2000.0, 4800.0)))     # band 2 = the capped single band
    x48 = rng.standard_normal(48000); m, v = envelope3_order_matrix(x48, 48000, 1797.0); assert v.all() and (np.abs(m[2]).sum() > 0)


def test_collate_and_forward_synthetic():
    torch.manual_seed(0); rng = np.random.default_rng(0)
    def item(valid):
        return {"streams": {"g0_c0": (rng.standard_normal((513, 192)).astype(np.float32), np.linspace(0, 24000, 513).astype(np.float32), np.linspace(0, 1, 192).astype(np.float32))},
                "env": (rng.standard_normal((3, N_ORDER_CELLS)).astype(np.float32) * np.array(valid, dtype=np.float32)[:, None], 3.5848, 1, np.array(valid, dtype=bool))}
    b = collate_v2([item([True, True, True]), item([True, True, False])], 1, 1, True, 3)
    assert b["env"].shape == (2, 3, N_ORDER_CELLS) and b["env_band_valid"].tolist() == [[True, True, True], [True, True, False]]
    m = PCSTEv2(PCSTEv2Config(envelope_branch=True, envelope_bands=3)); out = m(**{k: v for k, v in b.items()})
    assert out["global_embedding"].shape == (2, 192) and torch.isfinite(out["global_embedding"]).all()


@pytest.mark.skipif(not env3_norm_path(PDIR, 1, "CWRU").exists(), reason="env3 normalisers not fitted yet")
def test_items_real_windows_env3():
    man = pd.read_csv(PDIR / "global_v2_fold_1.csv"); tr = man[man.split == "train"]
    rows = [tr[(tr.dataset == "CWRU") & (tr.native_sampling_rate_hz == 12000)].iloc[0], tr[(tr.dataset == "CWRU") & (tr.native_sampling_rate_hz == 48000)].iloc[0], tr[tr.dataset == "HIT"].iloc[0], tr[tr.dataset == "MAFAULDA"].iloc[0]]
    ib = ItemBuilder(PDIR, 1, ("v1",), {"MAFAULDA": 1}, True, envelope_bands=3); items = [ib.build(r) for r in rows]
    assert items[0]["env"][3].tolist() == [True, True, False] and not np.abs(items[0]["env"][0][2]).any()
    assert all(it["env"][3].tolist() == [True, True, True] for it in items[1:]) and all(np.isfinite(it["env"][0]).all() for it in items)
    ib1 = ItemBuilder(PDIR, 1, ("v1",), {"MAFAULDA": 1}, True); ref = ib1.build(rows[1])
    assert np.array_equal(ref["streams"]["g0_c0"][0], items[1]["streams"]["g0_c0"][0])           # STFT stream identical to the Stage B item
    b = collate_v2(items, 1, 1, True, 3); m = PCSTEv2(PCSTEv2Config(envelope_branch=True, envelope_bands=3)); out = m(**b)
    assert torch.isfinite(out["global_embedding"]).all()


# ---------------------------------------------------------------------------------------------- Stage C: two-channel path
def test_two_channel_masking_and_micro_batch():
    """Non-MaFaulDa items carry no channel 1: their placeholder stream is masked out of the channel mean; MaFaulDa items use both.
    The two-channel micro-batch is an infrastructure setting: the objective is the exact effective-batch mean for any chunking."""
    from src.pcste_v2.model import PCSTEv2, PCSTEv2Config, collate_v2
    torch.manual_seed(0); rng = np.random.default_rng(0)
    def stream(bins, frames, fs):
        return (rng.standard_normal((bins, frames)).astype(np.float32), np.linspace(0, fs / 2, bins).astype(np.float32), np.linspace(0, 1, frames).astype(np.float32))
    items = [{"streams": {"g0_c0": stream(513, 184, 48000)}},                                                # CWRU48: channel 1 absent
             {"streams": {"g0_c0": stream(513, 192, 50000), "g0_c1": stream(513, 192, 50000)}},              # MaFaulDa: both channels
             {"streams": {"g0_c0": stream(257, 192, 25000)}}]                                                # HIT: channel 1 absent
    b = collate_v2(items, 1, 2, False)
    assert b["stream_valid"].tolist() == [[[True, False]], [[True, True]], [[True, False]]]
    assert not b["cell_mask_g0_c1"][0].any() and not b["cell_mask_g0_c1"][2].any() and b["cell_mask_g0_c1"][1].any()
    m = PCSTEv2(PCSTEv2Config(n_channels=2)); out = m(**b)
    z, Z = out["global_embedding"], out["channel_embeddings"]
    assert torch.isfinite(z).all() and torch.isfinite(Z).all()
    assert torch.allclose(z[0], Z[0, 0]) and torch.allclose(z[2], Z[2, 0])            # single-channel items: mean over the valid channel only
    assert torch.allclose(z[1], 0.5 * (Z[1, 0] + Z[1, 1]))                            # MaFaulDa: mean of both channels
    z.sum().backward(); assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    sys.path.insert(0, str(REPO / "scripts/pcste_v2")); import run as R
    assert R.MICRO_BATCH_2CH == 16 and R.MICRO_BATCH["single"] == 32
    # exact objective under chunking: L = mean_ds CE_ds; chunk weights w = sum_ds_in_chunk(n_chunk/n_batch)/n_ds sum to 1 per dataset
    batch = [("CWRU", 0, 0)] * 20 + [("HIT", 0, 0)] * 12 + [("MAFAULDA", 0, 0)] * 32
    n_ds = len({dd for dd, _, _ in batch}); cnt_batch = {dd: sum(1 for x in batch if x[0] == dd) for dd in {x[0] for x in batch}}
    for mb in (32, 16):
        tot = 0.0
        for lo in range(0, len(batch), mb):
            chunk = batch[lo:lo + mb]; cnt = {dd: sum(1 for x in chunk if x[0] == dd) for dd in {x[0] for x in chunk}}
            tot += sum(cnt[dd] / cnt_batch[dd] for dd in cnt) / n_ds
        assert abs(tot - 1.0) < 1e-12
