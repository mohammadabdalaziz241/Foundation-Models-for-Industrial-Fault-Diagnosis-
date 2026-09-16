"""
test_repro_seeding.py — unit tests for E0 (seeding, provenance, per-window
normalisation, discriminative-LR optimiser) and the E1 metric helpers.

Run:
  python tests/test_repro_seeding.py
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cv_paderborn import (
    derive_fold_seed,
    per_window_normalise,
    seed_everything,
)
from src.models import build_model
from src.models.masked_ssl import MaskedSSL
from src.repro import manifest_digest, provenance, source_manifest
from src.repr_analysis import (
    bearing_identity_probe,
    class_bearing_variance_ratio,
    knn_bearing_accuracy,
    knn_class_accuracy_lobo,
    knn_class_accuracy_ref,
    silhouette_by,
)

GREEN, RED, RESET = "\033[32m", "\033[31m", "\033[0m"
n_pass = n_fail = 0


def check(label: str, cond: bool) -> None:
    global n_pass, n_fail
    if cond:
        n_pass += 1
        print(f"  {label:60s} {GREEN}PASS{RESET}")
    else:
        n_fail += 1
        print(f"  {label:60s} {RED}FAIL{RESET}")


def _load_train_script():
    spec = importlib.util.spec_from_file_location(
        "train_cv_paderborn", ROOT / "scripts" / "train_cv_paderborn.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


print("\n" + "=" * 72)
print("  E0/E1 unit tests — seeding, provenance, per-window norm, repr metrics")
print("=" * 72)

# ---------------------------------------------------------------------------
# derive_fold_seed
# ---------------------------------------------------------------------------

s1 = derive_fold_seed(42, 0, 0)
s2 = derive_fold_seed(42, 0, 0)
check("R01 derive_fold_seed deterministic", s1 == s2)
check("R02 derive_fold_seed in [0, 2^31)", 0 <= s1 < 2**31)

combos = {derive_fold_seed(42, r, f) for r in range(3) for f in range(4)}
check("R03 12 (repeat, fold) combos give 12 distinct seeds", len(combos) == 12)

check("R04 different train seeds give different fold seeds",
      derive_fold_seed(42, 0, 0) != derive_fold_seed(43, 0, 0))

# ---------------------------------------------------------------------------
# seed_everything — model init, mask sampling
# ---------------------------------------------------------------------------

seed_everything(123)
m1 = build_model("lstm", num_classes=3)
seed_everything(123)
m2 = build_model("lstm", num_classes=3)
same = all(torch.equal(a, b) for (_, a), (_, b)
           in zip(m1.state_dict().items(), m2.state_dict().items()))
check("R05 same seed → identical LSTM init", same)

seed_everything(124)
m3 = build_model("lstm", num_classes=3)
diff = any(not torch.equal(a, b) for (_, a), (_, b)
           in zip(m1.state_dict().items(), m3.state_dict().items()))
check("R06 different seed → different LSTM init", diff)

seed_everything(7)
ssl_a = MaskedSSL(encoder_model=build_model("lstm", num_classes=3))
mask_a = ssl_a._make_patch_mask(4, torch.device("cpu"))
seed_everything(7)
ssl_b = MaskedSSL(encoder_model=build_model("lstm", num_classes=3))
mask_b = ssl_b._make_patch_mask(4, torch.device("cpu"))
check("R07 same seed → identical SSL patch masks", torch.equal(mask_a, mask_b))

check("R08 cudnn deterministic flags set",
      torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark)

# ---------------------------------------------------------------------------
# per_window_normalise
# ---------------------------------------------------------------------------

rng = np.random.default_rng(0)
X = rng.normal(3.0, 5.0, size=(32, 1024)).astype(np.float32)
Xn = per_window_normalise(X)
check("R09 per-window mean ≈ 0", float(np.abs(Xn.mean(1)).max()) < 1e-4)
check("R10 per-window std ≈ 1", float(np.abs(Xn.std(1) - 1).max()) < 1e-3)

X_aff = (2.5 * X - 7.0).astype(np.float32)
check("R11 per-window invariant to global affine transform",
      np.allclose(per_window_normalise(X_aff), Xn, atol=1e-3))

X_const = np.zeros((2, 1024), dtype=np.float32)
Xc = per_window_normalise(X_const)
check("R12 constant window does not produce inf/nan",
      np.isfinite(Xc).all())

# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------

prov = provenance(root=ROOT, argv=["test"])
check("R13 provenance has required keys",
      {"command", "git_commit", "source_digest", "source_manifest",
       "environment"} <= set(prov.keys()))
check("R14 manifest tracks train_cv_paderborn.py and cv_paderborn.py",
      "scripts/train_cv_paderborn.py" in prov["source_manifest"]
      and "src/cv_paderborn.py" in prov["source_manifest"])
check("R15 source digest stable across calls",
      manifest_digest(source_manifest(ROOT)) == prov["source_digest"])
check("R16 git commit recorded (repo initialised)",
      isinstance(prov["git_commit"], str) and len(prov["git_commit"]) == 40)

# ---------------------------------------------------------------------------
# CLI defaults and optimiser groups
# ---------------------------------------------------------------------------

tcp = _load_train_script()
args = tcp._parse_args(["--model", "lstm"])
check("R17 --train-seed default 42", args.train_seed == 42)
check("R18 --lp-warmup-epochs default 0", args.lp_warmup_epochs == 0)
check("R19 --encoder-lr-scale default 1.0", args.encoder_lr_scale == 1.0)
args_pw = tcp._parse_args(["--normalisation-policy", "per_window"])
check("R20 per_window accepted as normalisation policy",
      args_pw.normalisation_policy == "per_window")

model = build_model("lstm", num_classes=3)
mask = {n: True for n, _ in model.named_parameters()}
ns = argparse.Namespace(optimizer="adamw", weight_decay=1e-2,
                        encoder_lr_scale=1.0)
opt1 = tcp.make_adaptation_optimizer(model, ns, 3e-4, mask)
check("R21 scale=1.0 → single param group (historical behaviour)",
      len(opt1.param_groups) == 1)

ns.encoder_lr_scale = 0.1
opt2 = tcp.make_adaptation_optimizer(model, ns, 3e-4, mask)
lrs = sorted(g["lr"] for g in opt2.param_groups)
check("R22 scale=0.1 → two groups with lrs 3e-5 / 3e-4",
      len(opt2.param_groups) == 2
      and abs(lrs[0] - 3e-5) < 1e-12 and abs(lrs[1] - 3e-4) < 1e-12)

n_head = sum(p.numel() for p in model.head.parameters())
head_group = [g for g in opt2.param_groups if abs(g["lr"] - 3e-4) < 1e-12][0]
check("R23 head group contains exactly the head parameters",
      sum(p.numel() for p in head_group["params"]) == n_head)

mask_frozen = {n: n.startswith("head.") for n, _ in model.named_parameters()}
opt3 = tcp.make_adaptation_optimizer(model, ns, 3e-4, mask_frozen)
enc_group = [g for g in opt3.param_groups if abs(g["lr"] - 3e-5) < 1e-12][0]
check("R24 mask-frozen encoder params excluded from encoder group",
      sum(p.numel() for p in enc_group["params"]) == 0)

# ---------------------------------------------------------------------------
# E1 metric helpers (synthetic geometry)
# ---------------------------------------------------------------------------

rng = np.random.default_rng(1)


def _synth(class_sep: float, bearing_sep: float):
    """
    Two classes × two bearings each, 100 windows per bearing, 8-D.
    class_sep displaces class means; bearing_sep displaces bearing means
    within a class.
    """
    emb, y, g = [], [], []
    for c in range(2):
        for b in range(2):
            centre = np.zeros(8)
            centre[0] = c * class_sep
            centre[1] = b * bearing_sep
            emb.append(rng.normal(0, 0.1, size=(100, 8)) + centre)
            y += [c] * 100
            g += [f"c{c}b{b}"] * 100
    return np.concatenate(emb), np.array(y), np.array(g)


emb_c, y_c, g_c = _synth(class_sep=5.0, bearing_sep=0.1)
check("E01 class-separated: knn LOBO ≈ 1",
      knn_class_accuracy_lobo(emb_c, y_c, g_c, k=5) > 0.95)
check("E02 class-separated: var ratio ≫ 1",
      class_bearing_variance_ratio(emb_c, y_c, g_c)["ratio"] > 10)

def _synth_bearing_dominant():
    """
    Four well-separated bearing clusters along one axis at 0/5/10/15 with
    classes interleaved (c0, c1, c1, c0) so the two class means coincide:
    bearing identity is fully decodable, class is not.
    """
    emb, y, g = [], [], []
    for i, (c, pos) in enumerate([(0, 0.0), (1, 5.0), (1, 10.0), (0, 15.0)]):
        centre = np.zeros(8)
        centre[1] = pos
        emb.append(rng.normal(0, 0.1, size=(100, 8)) + centre)
        y += [c] * 100
        g += [f"b{i}"] * 100
    return np.concatenate(emb), np.array(y), np.array(g)


emb_b, y_b, g_b = _synth_bearing_dominant()
check("E03 bearing-separated: knn LOBO ≈ chance",
      knn_class_accuracy_lobo(emb_b, y_b, g_b, k=5) < 0.7)
check("E04 bearing-separated: var ratio ≪ 1",
      class_bearing_variance_ratio(emb_b, y_b, g_b)["ratio"] < 0.1)
check("E05 bearing-separated: knn bearing ID ≈ 1",
      knn_bearing_accuracy(emb_b, g_b, k=5) > 0.95)
probe = bearing_identity_probe(emb_b, g_b, seed=0)
check("E06 bearing probe ≈ 1 on separable bearings",
      probe["accuracy"] > 0.95 and abs(probe["chance"] - 0.25) < 1e-9)

check("E07 knn_class_accuracy_ref ≈ 1 with class-separated ref",
      knn_class_accuracy_ref(emb_c, y_c, emb_c, y_c, k=5) > 0.95)

sil_c = silhouette_by(emb_c, y_c, max_samples=200, seed=0)
sil_b = silhouette_by(emb_b, y_b, max_samples=200, seed=0)
check("E08 silhouette(class) higher when classes separate", sil_c > sil_b)
check("E09 silhouette deterministic given seed",
      silhouette_by(emb_c, y_c, max_samples=200, seed=0) == sil_c)
check("E10 silhouette nan for single label",
      np.isnan(silhouette_by(emb_c, np.zeros(len(emb_c)), max_samples=200)))

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print()
sep = "=" * 72
if n_fail == 0:
    print(f"{sep}\n  {GREEN}ALL {n_pass} TESTS PASSED{RESET}\n{sep}")
else:
    print(f"{sep}\n  {RED}{n_fail} TEST(S) FAILED{RESET}  ({n_pass} passed)\n{sep}")
sys.exit(0 if n_fail == 0 else 1)
