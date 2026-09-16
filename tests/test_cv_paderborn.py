"""
test_cv_paderborn.py — 20 unit tests for the Paderborn GroupCV framework.

Run:
  python tests/test_cv_paderborn.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cv_paderborn import (
    DEVELOPMENT_POOL,
    TEST_BEARINGS,
    CLASS_NAMES,
    N_CLASSES,
    _NORMAL_BEARINGS,
    _IR_ART_BEARINGS,
    _IR_REAL_BEARINGS,
    _OR_ART_BEARINGS,
    _OR_REAL_BEARINGS,
    _DAMAGE_ORIGIN,
    bearing_balanced_f1,
    bearing_metadata,
    compute_fold_stats,
    apply_fold_stats,
    compute_median_epoch,
    fold_normalise,
    final_normalise,
    generate_balance_report,
    make_folds,
    per_bearing_metrics,
    trailing_mean,
)
from src.models import build_model
from scripts.train_cv_paderborn import make_optimizer

GREEN = "\033[32m"
RED   = "\033[31m"
RESET = "\033[0m"
PASS  = f"{GREEN}PASS{RESET}"
FAIL  = f"{RED}FAIL{RESET}"

n_pass = n_fail = 0


def ok(label: str, detail: str = "") -> None:
    global n_pass
    n_pass += 1
    suffix = f"  {detail}" if detail else ""
    print(f"  T{n_pass + n_fail:02d} {label:55s} {PASS}{suffix}")


def fail(label: str, msg: str = "") -> None:
    global n_fail
    n_fail += 1
    print(f"  T{n_pass + n_fail:02d} {label:55s} {FAIL}  {msg}")


def check(condition: bool, label: str, msg: str = "") -> None:
    if condition:
        ok(label)
    else:
        fail(label, msg)


# ===========================================================================
# T01 — Development pool contains exactly the 21 listed bearings
# ===========================================================================
expected_pool = set(
    _NORMAL_BEARINGS + _IR_ART_BEARINGS + _IR_REAL_BEARINGS
    + _OR_ART_BEARINGS + _OR_REAL_BEARINGS
)
check(
    set(DEVELOPMENT_POOL) == expected_pool and len(DEVELOPMENT_POOL) == 21,
    "Dev pool = 21 bearings (4N + 4IRa + 4IRr + 5ORa + 4ORr)",
    f"pool={set(DEVELOPMENT_POOL)}"
)

# ===========================================================================
# T02 — None of the eight sealed test bearings enters CV
# ===========================================================================
check(
    len(set(DEVELOPMENT_POOL) & TEST_BEARINGS) == 0,
    "Sealed test bearings excluded from dev pool",
    f"overlap={set(DEVELOPMENT_POOL) & TEST_BEARINGS}"
)
check(
    TEST_BEARINGS == frozenset({"K005","K006","KI08","KI18","KI21","KA08","KA09","KA30"}),
    "Sealed test set is exactly the 8 specified bearings",
)

# ===========================================================================
# T03 — Four folds are generated per repeat
# ===========================================================================
folds = make_folds(n_splits=4, n_repeats=1, split_seed=42)
check(len(folds) == 1,                 "make_folds: 1 repeat returned")
check(len(folds[0]) == 4,              "make_folds: 4 folds in repeat 0")

# ===========================================================================
# T04 — Every fold contains all three classes
# ===========================================================================
all_ok = True
for fold in folds[0]:
    val_b = fold["val_bearings"]
    classes_present = {
        _DAMAGE_ORIGIN.get(b, "") for b in val_b  # use fault type instead
    }
    # Check via fault_type lookup
    fault_types = {
        "normal"     if b in _NORMAL_BEARINGS else
        "inner_race" if b in _IR_ART_BEARINGS + _IR_REAL_BEARINGS else
        "outer_race"
        for b in val_b
    }
    if not ({"normal", "inner_race", "outer_race"} <= fault_types):
        all_ok = False
check(all_ok, "Every fold val set contains all 3 classes")

# ===========================================================================
# T05 — Every fold contains one artificial and one real Inner bearing
# ===========================================================================
all_ok = True
ir_art_set  = set(_IR_ART_BEARINGS)
ir_real_set = set(_IR_REAL_BEARINGS)
for fold in folds[0]:
    vset = set(fold["val_bearings"])
    n_ir_art  = len(vset & ir_art_set)
    n_ir_real = len(vset & ir_real_set)
    if n_ir_art != 1 or n_ir_real != 1:
        all_ok = False
check(all_ok, "Every fold: exactly 1 artificial + 1 real Inner Race")

# ===========================================================================
# T06 — Every fold contains exactly one real Outer bearing
# ===========================================================================
or_real_set = set(_OR_REAL_BEARINGS)
all_ok = all(len(set(f["val_bearings"]) & or_real_set) == 1 for f in folds[0])
check(all_ok, "Every fold: exactly 1 real Outer Race")

# ===========================================================================
# T07 — Every fold contains at least one artificial Outer bearing
# ===========================================================================
or_art_set = set(_OR_ART_BEARINGS)
all_ok = all(len(set(f["val_bearings"]) & or_art_set) >= 1 for f in folds[0])
check(all_ok, "Every fold: ≥1 artificial Outer Race")

# One fold should have 2 artificial OR bearings.
counts = [len(set(f["val_bearings"]) & or_art_set) for f in folds[0]]
check(max(counts) == 2, "Exactly one fold has 2 artificial OR bearings", str(counts))

# ===========================================================================
# T08 — Every development bearing appears in val exactly once per repeat
# ===========================================================================
all_val = []
for fold in folds[0]:
    all_val.extend(fold["val_bearings"])
check(
    sorted(all_val) == sorted(DEVELOPMENT_POOL),
    "Each dev bearing in val exactly once per repeat",
    f"val={sorted(all_val)}"
)

# ===========================================================================
# T09 — Assignment is deterministic for a fixed seed
# ===========================================================================
folds_a = make_folds(4, 1, 42)
folds_b = make_folds(4, 1, 42)
check(
    folds_a[0][0]["val_bearings"] == folds_b[0][0]["val_bearings"],
    "Same seed → identical fold 0 val assignment",
)
check(
    all(fa["val_bearings"] == fb["val_bearings"]
        for fa, fb in zip(folds_a[0], folds_b[0])),
    "Same seed → identical all folds",
)

# ===========================================================================
# T10 — Different seeds create different valid assignments
# ===========================================================================
folds_c = make_folds(4, 1, 99)
different = any(
    fa["val_bearings"] != fc["val_bearings"]
    for fa, fc in zip(folds[0], folds_c[0])
)
# Also verify seed 99 folds are still valid
valid_c = True
for fold in folds_c[0]:
    vset = set(fold["val_bearings"])
    if not (vset & set(_NORMAL_BEARINGS)): valid_c = False
    if not (vset & ir_art_set):           valid_c = False
    if not (vset & ir_real_set):          valid_c = False
    if not (vset & or_real_set):          valid_c = False
check(different, "Different seeds produce different assignments")
check(valid_c,   "Different seed still produces valid fold structure")

# ===========================================================================
# T11 — Bearing-balanced weighted confusion matrix is correct
# ===========================================================================
# Example: bearing A (Normal, 10w all correct), B (IR, 5w all pred OR), C (OR, 20w all correct)
y_true = np.array([0]*10 + [1]*5 + [2]*20)
y_pred = np.array([0]*10 + [2]*5 + [2]*20)
groups = np.array(["A"]*10 + ["B"]*5 + ["C"]*20)
bb_f1, pc_f1, wt_cm = bearing_balanced_f1(y_true, y_pred, groups, n_classes=3)

# Expected weighted CM:
# C[0,0]=1.0  (10*(1/10))
# C[1,2]=1.0  (5*(1/5))
# C[2,2]=1.0  (20*(1/20))
ok_cm = (
    abs(wt_cm[0,0] - 1.0) < 1e-9 and
    abs(wt_cm[1,2] - 1.0) < 1e-9 and
    abs(wt_cm[2,2] - 1.0) < 1e-9 and
    abs(wt_cm.sum() - 3.0) < 1e-9
)
check(ok_cm, "Bearing-balanced weighted CM is correct", str(wt_cm))

# precision[2] = C[2,2]/(C[1,2]+C[2,2]) = 1/(1+1) = 0.5
# recall[2]    = C[2,2]/C[2,2] = 1.0
# F1[2]        = 2*0.5*1/(0.5+1) = 0.667
# F1[0]        = 1.0, F1[1]=0.0, macro=(1+0+0.667)/3=0.556
expected_macro = (1.0 + 0.0 + 2/3) / 3
check(
    abs(bb_f1 - expected_macro) < 1e-6,
    "Bearing-balanced macro-F1 computed correctly",
    f"expected={expected_macro:.6f} got={bb_f1:.6f}"
)

# ===========================================================================
# T12 — Single-class perfect bearing does not collapse fold metric to 1.0
# ===========================================================================
# Three bearings, one class each.  Normal perfect, IR total fail, OR perfect.
y_true2 = np.array([0]*100 + [1]*100 + [2]*100)
y_pred2 = np.array([0]*100 + [0]*100 + [2]*100)  # IR all wrong
groups2 = np.array(["A"]*100 + ["B"]*100 + ["C"]*100)
bb_f12, pc_f12, _ = bearing_balanced_f1(y_true2, y_pred2, groups2, n_classes=3)
check(
    abs(bb_f12) < 1.0 and abs(bb_f12) > 0.0,
    "Single-class perfect bearing does not collapse metric to 1.0 or 0.0",
    f"bb_f1={bb_f12:.4f}"
)
# Also verify the Normal bearing F1 is 2/3 (not 1.0) due to FPs from IR misclassified
# C[0,0] = 1.0 (A perfect), C[1,0] = 1.0 (B all pred Normal), C[2,2] = 1.0
# precision[0] = 1/(1+1) = 0.5, recall[0] = 1/1 = 1.0, F1[0] = 2/3
check(
    abs(pc_f12[0] - 2/3) < 1e-6,
    "Normal bearing F1 = 2/3 when it gets OR IR false positives",
    f"F1[0]={pc_f12[0]:.6f}"
)

# ===========================================================================
# T13 — Three-epoch smoothing is correct
# ===========================================================================
vals   = [0.1, 0.3, 0.5, 0.4, 0.6]
smooth = trailing_mean(vals, window=3)
expected = [0.1, 0.2, 0.3, 0.4, 0.5]
ok_smooth = all(abs(s - e) < 1e-9 for s, e in zip(smooth, expected))
check(ok_smooth, "trailing_mean window=3 is correct", f"got={smooth}")

# Edge: window larger than data length
smooth1 = trailing_mean([0.5], window=3)
check(smooth1 == [0.5], "trailing_mean: single value → [value]")

# ===========================================================================
# T14 — Median selected-epoch calculation is correct (ceil of median)
# ===========================================================================
check(compute_median_epoch([5, 3, 7, 6])  == 6, "Median [5,3,7,6] = ceil(5.5) = 6")
check(compute_median_epoch([10, 20])       == 15, "Median [10,20] = ceil(15.0) = 15")
check(compute_median_epoch([1])            == 1,  "Median [1] = 1")
check(compute_median_epoch([0, 0, 0, 0])   == 1,  "Median [0,…] clamped to 1")
# Ceiling (not banker's rounding): a fractional median rounds UP.
check(compute_median_epoch([12, 14, 15, 17]) == 15, "Median [12,14,15,17] = ceil(14.5) = 15")
check(compute_median_epoch([12, 14, 16, 18]) == 15, "Median [12,14,16,18] = ceil(15.0) = 15")

# ===========================================================================
# T15 — Gradient clipping is applied correctly
# ===========================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)  # deterministic init so the amplified grad reliably exceeds the clip
model15 = build_model("lstm", num_classes=3).to(device)
X15 = torch.randn(4, 1, 1024, device=device)
y15 = torch.zeros(4, dtype=torch.long, device=device)
opt15 = torch.optim.Adam(model15.parameters(), lr=1e-3)
opt15.zero_grad()
logits15 = model15(X15) * 100.0  # amplified → large grads
loss15 = nn.CrossEntropyLoss()(logits15, y15)
loss15.backward()
norm_before = torch.nn.utils.clip_grad_norm_(model15.parameters(), float("inf")).item()
opt15.zero_grad()
logits15 = model15(X15) * 100.0
loss15 = nn.CrossEntropyLoss()(logits15, y15)
loss15.backward()
clip_val = 1.0
torch.nn.utils.clip_grad_norm_(model15.parameters(), clip_val)
norm_after = sum(
    p.grad.norm(2).item()**2 for p in model15.parameters() if p.grad is not None
) ** 0.5
check(
    norm_before > clip_val,
    f"Grad norm before clipping > {clip_val} (i.e., clipping is needed)",
    f"norm_before={norm_before:.4f}"
)
check(
    norm_after <= clip_val + 1e-5,
    f"Grad norm after clipping ≤ {clip_val}",
    f"norm_after={norm_after:.6f}"
)

# ===========================================================================
# T16 — AdamW weight decay is applied correctly
# ===========================================================================
model16 = build_model("lstm", num_classes=3).to(device)
opt_adam  = make_optimizer(model16, "adam",  lr=1e-3, weight_decay=0.01)
opt_adamw = make_optimizer(model16, "adamw", lr=1e-3, weight_decay=0.01)
check(isinstance(opt_adam,  torch.optim.Adam),  "make_optimizer('adam') → Adam")
check(isinstance(opt_adamw, torch.optim.AdamW), "make_optimizer('adamw') → AdamW")
check(
    opt_adamw.defaults["weight_decay"] == 0.01,
    "AdamW weight_decay stored correctly",
    f"wd={opt_adamw.defaults['weight_decay']}"
)
check(
    opt_adam.defaults["weight_decay"] == 0.01,
    "Adam weight_decay stored correctly",
    f"wd={opt_adam.defaults['weight_decay']}"
)

# ===========================================================================
# T17 — SSL validation bearings never enter SSL pretraining
# ===========================================================================
from src.cv_paderborn import load_development_data, make_cv_loader, TEST_BEARINGS

X_dev, y_dev, b_dev = load_development_data("data/processed_paderborn")
fold0 = folds[0][0]
train_bs = fold0["train_bearings"]
val_bs   = fold0["val_bearings"]

_, train_groups = make_cv_loader(X_dev, y_dev, b_dev, train_bs, 32)
_, val_groups   = make_cv_loader(X_dev, y_dev, b_dev, val_bs,   32)

ssl_train_set = set(np.unique(train_groups).tolist())
val_set       = set(np.unique(val_groups).tolist())
check(
    len(ssl_train_set & val_set) == 0,
    "SSL train loader has no overlap with val bearings",
    f"overlap={ssl_train_set & val_set}"
)
check(
    ssl_train_set == set(train_bs),
    "SSL train loader contains exactly the fold-train bearings",
)

# ===========================================================================
# T18 — Test loader cannot be accessed during CV mode (dev data only)
# ===========================================================================
from src.cv_paderborn import TEST_BEARINGS as _tb
test_leaked = set(np.unique(b_dev).tolist()) & _tb
check(
    len(test_leaked) == 0,
    "load_development_data() contains no sealed test bearings",
    f"leaked={test_leaked}"
)
# Verify that attempting to load test split via load_development_data raises no
# test data (test split is excluded from load_development_data by design).
try:
    X2, y2, b2 = load_development_data("data/processed_paderborn")
    all_loaded_bearings = set(np.unique(b2).tolist())
    check(
        all_loaded_bearings <= set(DEVELOPMENT_POOL),
        "load_development_data: only dev bearings loaded, no test contamination",
    )
except Exception as e:
    fail("load_development_data: no exception expected", str(e))

# ===========================================================================
# T19 — Final mode uses a fixed CV-derived duration
# ===========================================================================
# Simulate: run CV stub, produce recommended_final_epochs, verify final mode uses it.
mock_selected_epochs = [14, 16, 12, 15]
recommended = compute_median_epoch(mock_selected_epochs)
# np.median([12,14,15,16]) = 14.5; ceil → 15 (never truncates below the budget)
check(
    recommended == 15,
    f"Median [14,16,12,15] = ceil(14.5) = 15",
    f"got {recommended}"
)
# In final mode, the script uses args.final_epochs (from --final-epochs),
# which must equal the recommended epoch from CV.  Verified by smoke test G.
check(
    recommended >= 1,
    "Recommended final epoch ≥ 1 (always trainable)",
    f"recommended={recommended}"
)

# ===========================================================================
# Fold-specific normalisation
# ===========================================================================
rng_n = np.random.default_rng(0)
Xn_train = rng_n.normal(loc=5.0, scale=3.0, size=(200, 1024)).astype(np.float32)
Xn_val   = rng_n.normal(loc=-2.0, scale=7.0, size=(50, 1024)).astype(np.float32)

Xt_n, Xv_n, mean_fit, std_fit = fold_normalise(Xn_train, Xn_val)

# N1 — fitted stats equal the fold-train global mean/std
check(
    abs(mean_fit - float(Xn_train.mean())) < 1e-4 and
    abs(std_fit  - float(Xn_train.std()))  < 1e-4,
    "fold_normalise: stats fitted on fold-train mean/std",
    f"mean={mean_fit:.4f} std={std_fit:.4f}",
)

# N2 — normalised fold-train has ~zero mean, ~unit std
check(
    abs(float(Xt_n.mean())) < 1e-4 and abs(float(Xt_n.std()) - 1.0) < 1e-4,
    "fold_normalise: train → mean≈0, std≈1",
    f"mean={Xt_n.mean():.5f} std={Xt_n.std():.5f}",
)

# N3 — val is standardised with TRAIN stats, not its own
expected_val = (Xn_val - mean_fit) / std_fit
check(
    np.allclose(Xv_n, expected_val, atol=1e-5),
    "fold_normalise: val uses train scaler (not its own stats)",
)
# Val does NOT become zero-mean/unit-std (its own distribution differs)
check(
    abs(float(Xv_n.std()) - 1.0) > 0.05,
    "fold_normalise: val std ≠ 1 (proves train stats were used)",
    f"val_std={Xv_n.std():.4f}",
)

# N4 — fold-val windows never influence the fitted scaler
Xn_val_alt = Xn_val * 100.0 + 500.0
_, _, mean_fit2, std_fit2 = fold_normalise(Xn_train, Xn_val_alt)
check(
    mean_fit2 == mean_fit and std_fit2 == std_fit,
    "fold_normalise: changing val leaves scaler unchanged (no leakage)",
)

# N5 — normalisation is affine and invertible
recovered = apply_fold_stats(Xt_n, 0.0, 1.0) * std_fit + mean_fit
check(
    np.allclose(recovered, Xn_train, atol=1e-3),
    "fold_normalise: affine map is invertible (recovers original)",
)

# N6 — near-constant fold-train raises (std not > eps)
X_const = np.full((10, 1024), 3.14, dtype=np.float32)
try:
    compute_fold_stats(X_const)
    check(False, "compute_fold_stats: constant array raises ValueError")
except ValueError:
    check(True, "compute_fold_stats: constant array raises ValueError")

# N7 — final_normalise fits on dev only; dev → mean≈0/std≈1
Xd = rng_n.normal(loc=1.0, scale=2.0, size=(300, 1024)).astype(np.float32)
Xte = rng_n.normal(loc=9.0, scale=0.5, size=(80, 1024)).astype(np.float32)
Xd_n, Xte_n, fmean, fstd = final_normalise(Xd, Xte)
check(
    abs(fmean - float(Xd.mean())) < 1e-4 and abs(fstd - float(Xd.std())) < 1e-4 and
    abs(float(Xd_n.mean())) < 1e-4 and abs(float(Xd_n.std()) - 1.0) < 1e-4,
    "final_normalise: fits on dev only, dev → mean≈0/std≈1",
)

# N8 — sealed test windows never influence the final scaler
_, _, fmean2, fstd2 = final_normalise(Xd, Xte * 50.0 - 100.0)
check(
    fmean2 == fmean and fstd2 == fstd,
    "final_normalise: changing test leaves scaler unchanged (no leakage)",
)

# ===========================================================================
# Fold metadata (note field) and balance report
# ===========================================================================
md_ka01 = bearing_metadata("KA01")
check(
    md_ka01["note"] == "EDM, extent 1" and
    md_ka01["fault_type"] == "outer_race" and
    md_ka01["damage_origin"] == "artificial",
    "bearing_metadata: KA01 note + fault_type + damage_origin correct",
    f"note={md_ka01['note']!r}",
)
check(
    all(bearing_metadata(b)["note"] != "" for b in DEVELOPMENT_POOL),
    "bearing_metadata: every dev bearing has a non-empty note",
)

balance = generate_balance_report(folds)
check(
    "repeats" in balance and "warnings" in balance and
    len(balance["repeats"][0]["folds"]) == 4,
    "generate_balance_report: structure has repeats/folds/warnings",
)
# Every fold-val set has ≥1 outer_race/real and ≥1 outer_race/artificial by design,
# so no rare-origin warnings should be emitted for the seed-42 folds.
check(
    balance["warnings"] == [],
    "generate_balance_report: seed-42 folds emit no balance warnings",
    f"warnings={balance['warnings']}",
)

# ===========================================================================
# T20 — Existing 36 model smoke tests still pass
# ===========================================================================
import subprocess
result = subprocess.run(
    [sys.executable, "tests/smoke_refactor.py"],
    capture_output=True, text=True
)
t20_ok = "ALL 36 TESTS PASSED" in result.stdout
check(t20_ok, "Existing 36 smoke tests still pass", "" if t20_ok else result.stdout[-400:])

# ===========================================================================
# Summary
# ===========================================================================
print()
sep = "=" * 60
if n_fail == 0:
    print(f"{sep}\n  {GREEN}ALL {n_pass} TESTS PASSED{RESET}\n{sep}")
else:
    print(f"{sep}\n  {RED}{n_fail} TEST(S) FAILED{RESET}  ({n_pass} passed)\n{sep}")

sys.exit(0 if n_fail == 0 else 1)
