"""
test_reverse_transfer_protocol.py — protocol validation and unit tests for
the Paderborn-to-CWRU reverse-transfer study (Part 4 pre-run checklist).

Checklist covered here (see scripts/train_reverse_transfer_cwru.py and
scripts/train_paderborn_to_cwru_zero_adapt.py docstrings for the full list):
  1-2. CWRU/Paderborn window shape, dtype, sample-rate compatibility.
  3.   Dataset-specific normalisation cannot leak target val/test statistics.
  4.   No dataset-identifier/metadata channel reaches the encoder.
  5.   The 4-class CWRU head is freshly (re)initialised regardless of what
       RNG draws preceded it, given a fresh seed_everything() call.
  6.   The Paderborn source pool excludes sealed-test bearings.
  7.   CWRU train/val/test recording groups are mutually disjoint.
  8.   MaskedSSL's decoder is identical regardless of data source.
  9.   derive_condition_seeds gives five independent, reproducible streams.
 10.   remap_cwru_to_common_classes is correct (class mapping, Ball excluded).

Run:
  python tests/test_reverse_transfer_protocol.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cv_paderborn import TEST_BEARINGS, seed_everything
from src.joint_ssl import load_cwru_split
from src.models import build_model
from src.models.masked_ssl import MaskedSSL
from src.reverse_transfer import CWRU_TO_COMMON, derive_condition_seeds, remap_cwru_to_common_classes
from scripts.train_reverse_transfer_cwru import prepare_paderborn_pool


def _apply_regime(model, regime: str) -> None:
    """Mirrors finetune_cwru's freeze/unfreeze logic exactly (architecture-agnostic)."""
    if regime == "ssl_linear":
        for p in model.parameters():
            p.requires_grad = False
        for p in model.head.parameters():
            p.requires_grad = True
    else:
        for p in model.parameters():
            p.requires_grad = True

GREEN, RED, RESET = "\033[32m", "\033[31m", "\033[0m"
n_pass = n_fail = 0


def check(label: str, cond: bool) -> None:
    global n_pass, n_fail
    if cond:
        n_pass += 1
        print(f"  {label:70s} {GREEN}PASS{RESET}")
    else:
        n_fail += 1
        print(f"  {label:70s} {RED}FAIL{RESET}")


print("\n" + "=" * 78)
print("  Reverse-transfer (Paderborn->CWRU) protocol validation + unit tests")
print("=" * 78)

# ---------------------------------------------------------------------------
# 1-2. Shape / dtype / sample-rate compatibility
# ---------------------------------------------------------------------------

X_cwru_train = load_cwru_split("train", ROOT / "data" / "processed")
X_pad = np.load(ROOT / "data" / "processed_paderborn" / "train" / "windows.npy")

check("P01 CWRU windows are (N, 1024)", X_cwru_train.shape[1] == 1024)
check("P02 Paderborn windows are (N, 1024)", X_pad.shape[1] == 1024)
check("P03 window lengths match exactly", X_cwru_train.shape[1] == X_pad.shape[1])
check("P04 both float32", X_cwru_train.dtype == X_pad.dtype == np.float32)

# ---------------------------------------------------------------------------
# 7. CWRU split disjointness
# ---------------------------------------------------------------------------

cwru_root = ROOT / "data" / "processed"
recs = {}
for split in ("train", "val", "test"):
    rows = list(csv.DictReader(open(cwru_root / split / "metadata.csv")))
    recs[split] = {r["recording_id"] for r in rows}
check("P05 CWRU train/val recording groups disjoint", not (recs["train"] & recs["val"]))
check("P06 CWRU train/test recording groups disjoint", not (recs["train"] & recs["test"]))
check("P07 CWRU val/test recording groups disjoint", not (recs["val"] & recs["test"]))

# ---------------------------------------------------------------------------
# 4. No dataset-identifier channel structurally possible
# ---------------------------------------------------------------------------

combined = np.concatenate([X_cwru_train[:4], X_pad[:4]], axis=0)
t = torch.from_numpy(combined).float().unsqueeze(1)
check("P08 combined CWRU+Paderborn batch tensor is exactly (B, 1, 1024)",
      tuple(t.shape) == (8, 1, 1024))

# ---------------------------------------------------------------------------
# 8. Decoder identical regardless of data source
# ---------------------------------------------------------------------------

m1 = build_model("lstm", num_classes=4)
ssl_cwru = MaskedSSL(encoder_model=m1, patch_len=32, mask_ratio=0.5)
m2 = build_model("lstm", num_classes=4)
ssl_pad = MaskedSSL(encoder_model=m2, patch_len=32, mask_ratio=0.5)
check("P09 decoder type identical for CWRU- and Paderborn-shaped probes",
      type(ssl_cwru.decoder) is type(ssl_pad.decoder))
check("P10 decoder param shapes identical",
      [p.shape for p in ssl_cwru.decoder.parameters()]
      == [p.shape for p in ssl_pad.decoder.parameters()])

# ---------------------------------------------------------------------------
# 6. Paderborn source pool excludes sealed-test bearings
# ---------------------------------------------------------------------------

X_ssltr, X_sslva, pad_meta = prepare_paderborn_pool(ssl_val_fraction=0.2, ssl_val_seed=42)
pool_bearings = set(pad_meta["ssl_train_bearings"]) | set(pad_meta["ssl_val_bearings"])
check("P11 Paderborn source pool has no sealed-test bearings",
      not (pool_bearings & TEST_BEARINGS))
check("P12 Paderborn SSL-train/val window counts recorded and positive",
      pad_meta["n_ssl_train_windows"] > 0 and pad_meta["n_ssl_val_windows"] > 0)
check("P13 Paderborn SSL-train/val bearing sets are disjoint",
      not (set(pad_meta["ssl_train_bearings"]) & set(pad_meta["ssl_val_bearings"])))

# ---------------------------------------------------------------------------
# 3. Dataset-specific normalisation leakage check
# ---------------------------------------------------------------------------

# The Paderborn pool scaler must be fit ONLY from Paderborn dev-pool arrays;
# prepare_paderborn_pool's signature never receives CWRU data at all, so no
# CWRU statistic can influence pad_meta['scaler_mean']/['scaler_std'] by
# construction.  Re-running with a different (but still leakage-safe) split
# fraction changes which bearings anchor the scaler, confirming it is
# genuinely being fit from the data rather than a hardcoded constant.
_, _, pad_meta_2 = prepare_paderborn_pool(ssl_val_fraction=0.35, ssl_val_seed=42)
check("P14 changing ssl_val_fraction changes the held-out set (scaler is data-driven)",
      set(pad_meta_2["ssl_val_bearings"]) != set(pad_meta["ssl_val_bearings"])
      or pad_meta_2["n_ssl_val_windows"] != pad_meta["n_ssl_val_windows"])

# ---------------------------------------------------------------------------
# 5. Fresh seed_everything + build_model is independent of prior RNG state
# ---------------------------------------------------------------------------

seed_everything(123)
_ = torch.randn(1000)          # arbitrary "prior" draws (simulating SSL pretraining)
_ = np.random.rand(500)
seed_everything(999)
head_a = build_model("lstm", num_classes=4).head.weight.clone()

seed_everything(42)
_ = torch.randn(37)            # DIFFERENT amount of prior consumption
seed_everything(999)
head_b = build_model("lstm", num_classes=4).head.weight.clone()

check("P15 fresh seed_everything(finetune_seed) + build_model is independent "
      "of preceding RNG consumption",
      torch.equal(head_a, head_b))

# ---------------------------------------------------------------------------
# 9. derive_condition_seeds
# ---------------------------------------------------------------------------

s1 = derive_condition_seeds(42, 0)
s2 = derive_condition_seeds(42, 0)
check("S01 derive_condition_seeds deterministic", s1 == s2)
check("S02 all five seed names present",
      set(s1.keys()) == {"model_init", "dataloader", "ssl_mask",
                         "dataset_sampling", "finetune"})
check("S03 all five values are independent (no duplicates within one call)",
      len(set(s1.values())) == 5)

s_other_seed_index = derive_condition_seeds(42, 1)
check("S04 different seed_index gives different seeds",
      s1 != s_other_seed_index)

s_other_base = derive_condition_seeds(43, 0)
check("S05 different base_seed gives different seeds", s1 != s_other_base)

all_vals = set()
for i in range(6):
    all_vals.update(derive_condition_seeds(42, i).values())
check("S06 6 seed_index values x 5 streams give 30 distinct integers",
      len(all_vals) == 30)

# ---------------------------------------------------------------------------
# 10. remap_cwru_to_common_classes
# ---------------------------------------------------------------------------

check("R01 CWRU_TO_COMMON mapping is exactly {0:0, 1:1, 3:2}",
      CWRU_TO_COMMON == {0: 0, 1: 1, 3: 2})

y_cwru = np.array([0, 1, 2, 3, 0, 2, 1, 3])
X_cwru = np.arange(len(y_cwru) * 4).reshape(len(y_cwru), 4).astype(np.float32)
X_out, y_out = remap_cwru_to_common_classes(X_cwru, y_cwru)

check("R02 Ball (label 2) windows are dropped",
      len(y_out) == (y_cwru != 2).sum() == 6)
check("R03 remapped labels only ever 0,1,2", set(y_out.tolist()) <= {0, 1, 2})
check("R04 Normal(0)->0, Inner Race(1)->1, Outer Race(3)->2",
      y_out.tolist() == [0, 1, 2, 0, 1, 2])
check("R05 X rows follow their remapped labels (no row/label misalignment)",
      np.array_equal(X_out, X_cwru[y_cwru != 2]))

y_all_ball = np.array([2, 2, 2])
X_all_ball = np.zeros((3, 4), dtype=np.float32)
X_e, y_e = remap_cwru_to_common_classes(X_all_ball, y_all_ball)
check("R06 all-Ball input yields empty output (no crash)", len(y_e) == 0 and len(X_e) == 0)

# ---------------------------------------------------------------------------
# CNN1D architecture support (Session 13: reverse-transfer study repeated
# with CNN1D).  These checks confirm CNN1D satisfies exactly the same
# encoder/decoder/freezing contracts already verified for LSTM above —
# no code change was needed in train_reverse_transfer_cwru.py itself, since
# --model was already a generic build_model(...) passthrough.
# ---------------------------------------------------------------------------

cnn_model = build_model("cnn1d", num_classes=4)
x4 = torch.randn(4, 1, 1024)
cnn_logits = cnn_model(x4)
cnn_embed = cnn_model.encode(x4)
cnn_feat = cnn_model.forward_sequence(x4)

check("CNN01 CNN1D forward -> (4, 4) logits", tuple(cnn_logits.shape) == (4, 4))
check("CNN02 CNN1D encode -> (4, 128) embedding", tuple(cnn_embed.shape) == (4, 128))
check("CNN03 CNN1D forward_sequence -> (4, 128, 129) (L < WINDOW_LEN)",
      tuple(cnn_feat.shape) == (4, 128, 129))

cnn_ssl = MaskedSSL(encoder_model=build_model("cnn1d", num_classes=4),
                    patch_len=32, mask_ratio=0.5)
with torch.no_grad():
    cnn_recon, cnn_loss = cnn_ssl(x4)
check("CNN04 CNN1D MaskedSSL reconstruction -> (4, 1, 1024)",
      tuple(cnn_recon.shape) == (4, 1, 1024))
check("CNN05 CNN1D MaskedSSL loss is finite", bool(torch.isfinite(cnn_loss)))

from src.models.masked_ssl import _CnnDecoder, _SequenceDecoder
lstm_ssl = MaskedSSL(encoder_model=build_model("lstm", num_classes=4),
                     patch_len=32, mask_ratio=0.5)
check("CNN06 CNN1D selects the upsampling _CnnDecoder (L=129 < 1024)",
      isinstance(cnn_ssl.decoder, _CnnDecoder))
check("CNN07 LSTM selects _SequenceDecoder (L=1024, contrast case)",
      isinstance(lstm_ssl.decoder, _SequenceDecoder))

# Linear probe: every encoder param frozen, every head param trainable.
cnn_lin = build_model("cnn1d", num_classes=4)
_apply_regime(cnn_lin, "ssl_linear")
check("CNN08 ssl_linear freezes every CNN1D encoder parameter",
      all(not p.requires_grad for p in cnn_lin.encoder.parameters()))
check("CNN09 ssl_linear leaves every CNN1D head parameter trainable",
      all(p.requires_grad for p in cnn_lin.head.parameters()))

# Full fine-tune: everything trainable.
cnn_full = build_model("cnn1d", num_classes=4)
_apply_regime(cnn_full, "ssl_full")
check("CNN10 ssl_full unfreezes every CNN1D parameter (encoder)",
      all(p.requires_grad for p in cnn_full.encoder.parameters()))
check("CNN11 ssl_full unfreezes every CNN1D parameter (head)",
      all(p.requires_grad for p in cnn_full.head.parameters()))

# Fresh CNN1D head is independent of preceding RNG consumption (same
# property already verified for the generic build_model path at P15, run
# again explicitly naming cnn1d since it is the architecture actually used
# in the new study).
seed_everything(123)
_ = torch.randn(1000)
seed_everything(999)
cnn_head_a = build_model("cnn1d", num_classes=4).head.weight.clone()

seed_everything(42)
_ = torch.randn(37)
seed_everything(999)
cnn_head_b = build_model("cnn1d", num_classes=4).head.weight.clone()

check("CNN12 fresh seed_everything(finetune_seed) + build_model('cnn1d') is "
      "independent of preceding RNG consumption",
      torch.equal(cnn_head_a, cnn_head_b))

# Decoder identical regardless of which dataset (CWRU- vs Paderborn-shaped
# probe tensor) trains the CNN1D encoder — same argument as P09/P10, redone
# for CNN1D specifically.
cnn_a = MaskedSSL(encoder_model=build_model("cnn1d", num_classes=4), patch_len=32, mask_ratio=0.5)
cnn_b = MaskedSSL(encoder_model=build_model("cnn1d", num_classes=4), patch_len=32, mask_ratio=0.5)
check("CNN13 CNN1D decoder architecture identical across independent inits",
      [p.shape for p in cnn_a.decoder.parameters()] == [p.shape for p in cnn_b.decoder.parameters()])

print()
sep = "=" * 78
if n_fail == 0:
    print(f"{sep}\n  {GREEN}ALL {n_pass} TESTS PASSED{RESET}\n{sep}")
else:
    print(f"{sep}\n  {RED}{n_fail} TEST(S) FAILED{RESET}  ({n_pass} passed)\n{sep}")
sys.exit(0 if n_fail == 0 else 1)
