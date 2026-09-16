"""Stage A verification for PC-STE v2 (run: .venv/bin/python -m pytest tests/pcste_v2 -q)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, pytest, torch
REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO))
from src.pcste_v2.model import PCSTEv2, PCSTEv2Config, collate_v2
from src.pcste_v2.representation import ItemBuilder, norm_path, env_norm_path, N2B_FLOOR, read_raw, stft_rep
from src.pcste_v2.envelope import envelope_order_vector, N_ORDER_CELLS
from src.pcste_v2.protocol import verify_global
from src.pcste_v2.protocol_cwru import verify_protocol, FOLDS
from src.methodology_v2.encoder import PCSTE, collate_representations
from src.methodology_v2.part4c_reader import get_representation
PDIR = REPO / "pcste_v2/protocol/global_v2_PB_sealedMAF_v1"; CWRU_DIR = REPO / "pcste_v2/protocol/cwru_v2_PB_v1"
SEALED_NORM = REPO / "methodology_v2/part4_representation_final/normalizers"


@pytest.fixture(scope="module")
def man1():
    verify_global(PDIR, 1); return pd.read_csv(PDIR / "global_v2_fold_1.csv")


def test_protocol_hashes_and_invariants(man1):
    verify_protocol(CWRU_DIR)
    for fold in (1, 2, 3):
        w = pd.read_csv(CWRU_DIR / f"cwru_v2_windows_fold_{fold}.csv")
        s = {k: set(w[w.split == k].physical_specimen) for k in ("train", "validation", "test")}
        assert not (s["train"] & s["validation"]) and not (s["train"] & s["test"]) and not (s["validation"] & s["test"])
        for cls in ("inner_race", "ball", "outer_race"):
            assert w[(w.split == "train") & (w.fault_type == cls)].physical_specimen.nunique() >= 3
            for sp in ("train", "validation", "test"):
                assert set(w[(w.split == sp) & (w.fault_type == cls)].native_sampling_rate_hz) == {12000, 48000}
    tests = [set(FOLDS[f]["test"]) for f in FOLDS]; assert not (tests[0] & tests[1]) and not (tests[1] & tests[2]) and not (tests[0] & tests[2])
    assert man1.window_id.is_unique and set(man1.dataset) == {"CWRU", "JNU", "HIT", "MAFAULDA"}


def test_sealed_windows_unchanged(man1):
    sealed = pd.read_csv(REPO / "methodology_v2/part3_windows/window_manifest_fold_1.csv")
    for ds in ("JNU", "HIT", "MAFAULDA"):
        assert set(man1[man1.dataset == ds].window_id) == set(sealed[sealed.dataset == ds].window_id)


@pytest.mark.parametrize("fold", [1, 3])
@pytest.mark.parametrize("ds", ["jnu", "hit", "mafaulda"])
def test_n2b_equals_sealed_n2_where_std_above_floor(fold, ds):
    a = np.load(SEALED_NORM / f"fold_{fold}" / f"{ds}.npz"); b = np.load(norm_path(PDIR, fold, "v1", ds.upper(), 0))
    assert np.allclose(a["mean"], b["mean"], atol=1e-10) and np.allclose(a["std_raw"], b["std_raw"], atol=1e-10)
    assert (a["std_raw"] >= N2B_FLOOR).all() and np.array_equal(a["std_denominator"], b["std_denominator"]) and b["floored_bins"].size == 0


@pytest.mark.parametrize("fold", [1, 3])
def test_n2b_cwru_floor_applied(fold):
    b = np.load(norm_path(PDIR, fold, "v1", "CWRU48", 0))
    assert float(b["std_denominator"].min()) >= N2B_FLOOR - 1e-12 and 1.0 / float(b["std_denominator"].min()) <= 20.0 + 1e-9
    assert np.array_equal(b["std_denominator"], np.maximum(b["std_raw"], N2B_FLOOR))


def test_control_identity_with_sealed_encoder(man1):
    torch.manual_seed(0); m = PCSTEv2(PCSTEv2Config()); core = PCSTE(); core.load_state_dict(m.core.state_dict())
    w = pd.read_csv(REPO / "methodology_v2/part3_windows/window_manifest_fold_1.csv"); w = w[w.split == "validation"]
    reps, items = [], []
    for ds in ("JNU", "HIT", "MAFAULDA"):
        r = w[w.dataset == ds].iloc[0]; x, meta = get_representation(r.window_id, 1)
        f = np.asarray(meta["frequency_hz"], np.float32); t = np.asarray(meta["time_seconds"], np.float32)
        reps.append((x, f, t)); items.append({"streams": {"g0_c0": (x, f, t)}})
    o1 = core(**collate_representations(reps))["global_embedding"]; o2 = m(**collate_v2(items, 1, 1, False))["global_embedding"]
    assert torch.equal(o1, o2)


def test_items_and_forward_real_windows(man1):
    tr = man1[man1.split == "train"]
    rows = [tr[(tr.dataset == "CWRU") & (tr.native_sampling_rate_hz == 48000)].iloc[0], tr[(tr.dataset == "CWRU") & (tr.native_sampling_rate_hz == 12000)].iloc[0],
            tr[tr.dataset == "JNU"].iloc[0], tr[tr.dataset == "HIT"].iloc[0], tr[tr.dataset == "MAFAULDA"].iloc[0]]
    for grids, env in ((("v1",), True), (("mr_short", "mr_long"), False)):
        ib = ItemBuilder(PDIR, 1, grids, {"MAFAULDA": 1}, env); items = [ib.build(r) for r in rows]
        if env:
            assert items[0]["env"][0].shape == (N_ORDER_CELLS,) and items[0]["env"][2] == 1 and items[2]["env"][2] == 0
            assert items[0]["streams"]["g0_c0"][0].shape == (513, 184) and items[1]["streams"]["g0_c0"][0].shape == (129, 184)
        cfg = PCSTEv2Config(envelope_branch=env, multires=(len(grids) == 2)); m = PCSTEv2(cfg)
        out = m(**collate_v2(items, len(grids), 1, env))["global_embedding"]; assert out.shape == (5, 192) and torch.isfinite(out).all()


def test_envelope_deterministic_and_level_invariant(man1):
    r = man1[(man1.dataset == "CWRU") & (man1.split == "train")].iloc[0]; x = read_raw(r); fs = float(r.native_sampling_rate_hz)
    v1 = envelope_order_vector(x, fs, r.rpm, (2000, 8000)); v2 = envelope_order_vector(x, fs, r.rpm, (2000, 8000)); v3 = envelope_order_vector(3.0 * x, fs, r.rpm, (2000, 8000))
    assert np.array_equal(v1, v2) and np.abs(v1 - v3).max() < 1e-5          # level removed by construction (log domain, median subtracted)


def test_sealed_artefacts_untouched():
    reg = pd.read_csv(REPO / "methodology_v2/part4_representation_final/normalizer_hashes.csv")
    from src.methodology_v2.part4c_normalizers import verify_part4c_hashes; verify_part4c_hashes()
    from src.methodology_v2.part2_builder import verify_frozen_hashes; verify_frozen_hashes()
    from src.methodology_v2.part3b_windows import verify_part3b_hashes; verify_part3b_hashes()
