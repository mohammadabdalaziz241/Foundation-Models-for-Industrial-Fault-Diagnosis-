"""CWRU intervention (cwru_xspec.v1) integration tests: verbatim kit components, default-off configuration, stream replacement
invariants on the real sealed manifests, micro-batch group coverage, and gradients reaching the real encoder."""
import hashlib, inspect, json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts/pcste_v2"))
from src.pcste_v2 import cwru_intervention as CI
from src.pcste_v2.model import PCSTEv2, PCSTEv2Config, collate_v2
from src.pcste_v2.representation import ItemBuilder
from src.pcste_v2.protocol import label_subset_frame
from src.methodology_v2.experiment.samplers import SupervisedSampler
from src.methodology_v2.experiment.trainers import DOWNSTREAM_EPOCHS, steps_per_epoch
import run as R
KIT = REPO / "cwru_intervention_kit"; PDIR = REPO / "pcste_v2/protocol/global_v2_PB_MAFv2_v1"
STAGE_C_N2B_STRAT_F1_CONFIG_SHA = json.load(open(REPO / "results/pcste_v2/c/pcstev2_c_radial_strat_f1_s42/fingerprint.json"))["config_sha256"]


def _func_src(src: str, name: str) -> str:
    i = src.index(f"def {name}("); j = src.find("\n\n\n", i); return src[i:j if j > 0 else None].rstrip()


def test_kit_components_verbatim():
    samp = (KIT / "specimen_sampling.py").read_text(); loss = (KIT / "cross_specimen_loss.py").read_text()
    assert hashlib.sha256(samp.encode()).hexdigest() == CI.KIT_SHA256["specimen_sampling.py"] and hashlib.sha256(loss.encode()).hexdigest() == CI.KIT_SHA256["cross_specimen_loss.py"]
    for name in ("make_epoch_plan", "replace_positions"):
        assert _func_src(samp, name) == inspect.getsource(getattr(CI, name)).rstrip()
    for name in ("pair_masks", "cross_specimen_loss"):
        assert _func_src(loss, name) == inspect.getsource(getattr(CI, name)).rstrip()


def _cfg(variant, fold=1, seed=42, protocol="global_v2_PB_MAFv2_v1"):
    grid_set, envelope, n_channels, control = R.VARIANTS[variant]; env_bands = R.ENVELOPE_BANDS.get(variant, 1); grids = R.GRID_SETS[grid_set]
    ci = R.CWRU_INTERVENTION.get(variant, {"sampling": False, "xspec": False}); w_cwru = 0.25
    return {"run_id": f"pcstev2_c_radial_strat_f{fold}_s{seed}", "stage": "c", "variant": variant, "fold": fold, "seed": seed, "protocol": protocol, "grids": list(grids), "envelope_branch": envelope,
            "n_channels": n_channels, "control": control, "level_gain_range": None, "model": PCSTEv2Config(envelope_branch=envelope, multires=False, n_channels=n_channels, envelope_bands=env_bands).to_dict(),
            **({"cwru_sampling": {"version": CI.CWRU_PLAN_VERSION, "items_per_step": CI.CWRU_ITEMS_PER_STEP, "class_quota_rotation": "5/5/6", "replaces_only_cwru_slots_of_original_stream": True}} if ci["sampling"] else {}),
            **({"cross_specimen_loss": {"version": CI.CWRU_XSPEC_VERSION, "tau": CI.XSPEC_TAU, "lambda": CI.XSPEC_LAMBDA, "cwru_domain_weight": w_cwru, "effective_coefficient": w_cwru * CI.XSPEC_LAMBDA,
                                         "embedding": "global_embedding (192-d encoder output, L2-normalised inside the loss)", "reduction": "anchor->specimen->class means", "same_class_same_specimen_pairs_excluded": True}} if ci["xspec"] else {}),
            "optimizer": R.OPTIMIZER_SPEC, "epochs": R.DOWNSTREAM_EPOCHS, "effective_batch": R.EFFECTIVE_BATCH, "micro_batch": R.MICRO_BATCH[grid_set],
            "checkpoint_rule": "max validation MacroDomainF1 (strict >, earlier epoch on ties)", "init": "random (S0)", "label_fraction": 1.0, "test_policy": "TEST never read"}


def test_default_off_config_unchanged_and_new_variants_flagged():
    assert hashlib.sha256(json.dumps(_cfg("n2b"), sort_keys=True).encode()).hexdigest() == STAGE_C_N2B_STRAT_F1_CONFIG_SHA     # Stage C control config reproduced bit-for-bit
    assert "cwru_sampling" not in _cfg("n2b") and "cross_specimen_loss" not in _cfg("n2b")
    c1, c2 = _cfg("n2b_c1"), _cfg("n2b_c2"); assert "cwru_sampling" in c1 and "cross_specimen_loss" not in c1 and "cwru_sampling" in c2 and c2["cross_specimen_loss"]["effective_coefficient"] == 0.025
    assert c1["optimizer"] == _cfg("n2b")["optimizer"] and c1["model"] == _cfg("n2b")["model"] and c1["micro_batch"] == 32 and c1["epochs"] == 50
    assert CI.XSPEC_TAU == 0.10 and CI.XSPEC_LAMBDA == 0.10


def test_universe_premise_all_folds_both_protocols():
    for pdir in ("global_v2_PB_MAFv2_v1", "global_v2_PB_sealedMAF_v1"):
        for f in (1, 2, 3):
            u = CI.cwru_train_universe(pd.read_csv(REPO / "pcste_v2/protocol" / pdir / f"global_v2_fold_{f}.csv", dtype={"fault_severity": str}))
            assert u["class_names"] == ["inner_race", "outer_race", "ball"] and len(u["spec_index"]) == 9 and u["n_windows"] == len(set(u["window_ids"]))


def _streams(fold=1, seed=42):
    man = pd.read_csv(PDIR / f"global_v2_fold_{fold}.csv", dtype={"fault_severity": str}); spe = steps_per_epoch(int((man.split == "train").sum()))
    torch.manual_seed(seed); np.random.seed(seed); smp = SupervisedSampler(label_subset_frame(man), 1.0, seed); base = [smp.next_batch() for _ in range(DOWNSTREAM_EPOCHS * spe)]
    u = CI.cwru_train_universe(man); new, info = CI.balanced_stream(base, u, spe, DOWNSTREAM_EPOCHS, seed); return man, base, new, u, info, spe


def test_balanced_stream_invariants_real_manifest():
    man, base, new, u, info, spe = _streams()
    assert CI.non_cwru_preserved(base, new) and len(new) == len(base) == 50 * spe and info["n_cwru_items_replaced"] == 16 * len(new)
    train_ids = set(u["window_ids"]); wid_cls = dict(zip(u["window_ids"], [u["class_names"][l] for l in u["labels"]])); wid_spec = dict(zip(u["window_ids"], u["specimens"]))
    for k, b in enumerate(new):
        cw = [it for it in b if it[0] == "CWRU"]; assert len(cw) == 16 and len({it[2] for it in cw}) == 16 and all(it[2] in train_ids and wid_cls[it[2]] == it[1] for it in cw)
        counts = pd.Series([it[1] for it in cw]).value_counts(); assert sorted(counts.values) == [5, 5, 6]
        for c in u["class_names"]:
            assert len({wid_spec[it[2]] for it in cw if it[1] == c}) >= 2
        assert [i for i, it in enumerate(b) if it[0] == "CWRU"] == list(range(16))          # CWRU slots are positions 0-15 -> inside the first 32-item micro-batch
    quotas = [max(pd.Series([it[1] for it in b if it[0] == "CWRU"]).value_counts().items(), key=lambda x: x[1])[0] for b in new[:6]]; assert len(set(quotas)) == 3   # 6-quota rotates over classes
    # determinism and C1 == C2 (same function, same seed)
    _, _, new2, _, info2, _ = _streams(); assert info2["plan_sha256"] == info["plan_sha256"] and new2 == new
    _, _, new3, _, info3, _ = _streams(seed=1337); assert info3["plan_sha256"] != info["plan_sha256"]
    # exposure balance over an epoch: per class and per specimen within class (difference <= 1, as in the kit test)
    ep = new[:spe]; cls_counts = pd.Series([it[1] for b in ep for it in b if it[0] == "CWRU"]).value_counts(); assert cls_counts.max() - cls_counts.min() <= 1
    for c in u["class_names"]:
        sc = pd.Series([wid_spec[it[2]] for b in ep for it in b if it[0] == "CWRU" and it[1] == c]).value_counts(); assert sc.max() - sc.min() <= 1


def test_loss_gradients_reach_real_encoder():
    man = pd.read_csv(PDIR / "global_v2_fold_1.csv", dtype={"fault_severity": str}); u = CI.cwru_train_universe(man)
    _, _, new, _, _, _ = _streams(); chunk = new[0][:32]                                 # first micro-batch: 16 CWRU + 16 JNU
    ib = ItemBuilder(PDIR, 1, ("v1",), {"MAFAULDA": 1}, False); mi = man.set_index("window_id")
    items = [ib.build(mi.loc[w].rename(w).to_frame().T.assign(window_id=w).iloc[0]) if False else ib.build(pd.Series(mi.loc[w].to_dict() | {"window_id": w})) for _, _, w in chunk]
    torch.manual_seed(0); model = PCSTEv2(PCSTEv2Config()); b = collate_v2(items, 1, 1, False)
    z = model(**b)["global_embedding"]; pos, y, s = CI.cwru_group_tensors(chunk, u, z.device)
    assert pos == list(range(16)) and y.dtype == torch.long and s.dtype == torch.long and len(set(s.tolist())) >= 6
    positive, eligible = CI.pair_masks(y, s); assert (positive.sum(1) >= 1).all() and not eligible.diag().any() and not (positive & (s[:, None] == s[None, :])).any()
    aux = CI.cross_specimen_loss(z[pos], y, s, temperature=CI.XSPEC_TAU); assert aux.dtype == torch.float32 and torch.isfinite(aux) and aux.item() > 0
    (0.25 * CI.XSPEC_LAMBDA * aux).backward()
    grads = [p.grad for n, p in model.named_parameters() if p.grad is not None]; assert grads and all(torch.isfinite(g).all() for g in grads) and sum(float(g.abs().sum()) for g in grads) > 0
    assert any(n.startswith("core.stem") or "temporal" in n for n, p in model.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0)   # gradient reaches the encoder body, not only the last layer
