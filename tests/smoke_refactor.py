"""
Smoke test for the refactored framework (src/config, src/models/__init__, src/trainer).

Does NOT require sklearn or matplotlib.  Exercises:
  1.  build_model registry (cnn1d + lstm)
  2.  class_weights, subsample_train, train_epoch, evaluate from src.trainer
  3.  training_loop with early stopping (2 epochs, val metric = simple accuracy)
  4.  plot_history graceful degradation
  5.  Checkpoint load/save round-trip
  6.  CompactCNN1D forward pass (pruning path)
  7.  MaskedSSL forward pass + loss.backward (default CNN1D encoder)
  8.  CNN1D.encoder load from ssl_encoder_*.pt (finetune path)
  9.  GPU memory stress (B=512)
  20. CNN1D.forward_sequence shape
  21. LSTM1D forward + encode + forward_sequence shapes
  22. MaskedSSL with LSTM encoder_model (SequenceDecoder path)
  23. build_model("lstm") param count and registry
  24. setup_partial_finetune for LSTM and CNN1D

Usage (from project root):
    .venv/bin/python tests/smoke_refactor.py
    .venv/bin/python tests/smoke_refactor.py --device cpu
"""

from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (
    CWRU_CLASSES, ENCODER_SEQ_INDICES, N_ENCODER_BLOCKS,
    CNN1D_KERNELS, CNN1D_STRIDES, WINDOW_LEN,
)
from src.datasets import make_loader
from src.models import MODEL_REGISTRY, build_model
from src.models.cnn1d import CNN1D
from src.models.cnn1d_pruned import CompactCNN1D, KERNELS, STRIDES
from src.models.lstm1d import LSTM1D
from src.models.masked_ssl import MaskedSSL
from src.models.tcn1d import TCN1D
from src.trainer import (
    class_weights, evaluate, plot_history,
    setup_partial_finetune, subsample_train, train_epoch, training_loop,
)

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"


def simple_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Minimal metrics function that does not require sklearn."""
    acc = float((y_true == y_pred).mean())
    return {"macro_f1": acc, "accuracy": acc}


def run(device: torch.device) -> None:
    print(f"\n{'='*60}")
    print(f"  Refactor smoke test  |  device={device}")
    print(f"{'='*60}\n")

    n_fail = 0

    # ── 1. src/config constants ───────────────────────────────────────────
    t = "1 src.config constants"
    try:
        assert WINDOW_LEN == 1024
        assert CNN1D_KERNELS == (64, 32, 16)
        assert CNN1D_STRIDES == (2, 1, 1)
        assert N_ENCODER_BLOCKS == 3
        assert ENCODER_SEQ_INDICES == (0, 2, 4)
        assert CWRU_CLASSES == ["Normal", "Inner Race", "Ball", "Outer Race"]
        assert KERNELS == (64, 32, 16)    # cnn1d_pruned.py re-exports from config
        assert STRIDES == (2, 1, 1)
        print(f"  {t:45s} {PASS}")
    except Exception as e:
        print(f"  {t:45s} {FAIL}  {e}")
        n_fail += 1

    # ── 2. build_model registry ───────────────────────────────────────────
    t = "2 build_model('cnn1d') param count + registry"
    try:
        m = build_model("cnn1d", num_classes=4)
        n = sum(p.numel() for p in m.parameters())
        assert n == 199620, f"Got {n}"
        assert list(MODEL_REGISTRY) == sorted(MODEL_REGISTRY)  # sorted
        assert "lstm" in MODEL_REGISTRY, "lstm missing from MODEL_REGISTRY"
        print(f"  {t:45s} {PASS}  ({n:,} params)")
    except Exception as e:
        print(f"  {t:45s} {FAIL}  {e}")
        n_fail += 1

    # ── 3. state_dict key compatibility ───────────────────────────────────
    t = "3 state_dict keys unchanged"
    try:
        keys = list(m.state_dict().keys())
        for expected in ("encoder.0.0.weight", "encoder.2.0.weight",
                         "encoder.4.0.weight", "head.weight", "head.bias"):
            assert expected in keys, f"Missing key: {expected}"
        print(f"  {t:45s} {PASS}")
    except Exception as e:
        print(f"  {t:45s} {FAIL}  {e}")
        n_fail += 1

    # ── 4. class_weights ──────────────────────────────────────────────────
    t = "4 class_weights (CWRU)"
    try:
        w = class_weights("data/processed", device, n_classes=4)
        assert w.shape == (4,)
        assert abs(w.sum().item() - 4.0) < 1e-4
        print(f"  {t:45s} {PASS}  sum={w.sum().item():.4f}")
    except Exception as e:
        print(f"  {t:45s} {FAIL}  {e}")
        n_fail += 1

    # ── 5. subsample_train (full, no sklearn) ─────────────────────────────
    t = "5 subsample_train (full split)"
    try:
        X_full, y_full = subsample_train("data/processed", 1.0, seed=42)
        assert X_full.shape[1:] == (1, WINDOW_LEN), f"{X_full.shape}"
        assert len(X_full) == len(y_full)
        print(f"  {t:45s} {PASS}  N={len(X_full):,}")
    except Exception as e:
        print(f"  {t:45s} {FAIL}  {e}")
        n_fail += 1

    # ── 6. CWRU DataLoader ────────────────────────────────────────────────
    t = "6 CWRU DataLoader shapes"
    try:
        tr = make_loader("train", "data/processed", batch_size=32)
        X, y = next(iter(tr))
        assert X.shape == (32, 1, WINDOW_LEN), f"{X.shape}"
        print(f"  {t:45s} {PASS}  X={tuple(X.shape)}")
    except Exception as e:
        print(f"  {t:45s} {FAIL}  {e}")
        n_fail += 1

    # ── 7. Paderborn DataLoader ───────────────────────────────────────────
    t = "7 Paderborn DataLoader shapes"
    try:
        tr_pu = make_loader("train", "data/processed_paderborn", batch_size=32)
        Xp, yp = next(iter(tr_pu))
        assert Xp.shape == (32, 1, WINDOW_LEN), f"{Xp.shape}"
        assert set(yp.numpy().tolist()).issubset({0, 1, 2})
        print(f"  {t:45s} {PASS}  X={tuple(Xp.shape)}  classes={yp.unique().tolist()}")
    except Exception as e:
        print(f"  {t:45s} {FAIL}  {e}")
        n_fail += 1

    # ── 8. GPU forward pass ───────────────────────────────────────────────
    t = "8 CNN1D GPU forward + encode"
    try:
        m = CNN1D(num_classes=4).to(device)
        x = torch.randn(8, 1, WINDOW_LEN, device=device)
        logits = m(x)
        emb = m.encode(x)
        assert logits.shape == (8, 4)
        assert emb.shape == (8, 128)
        print(f"  {t:45s} {PASS}  logits={tuple(logits.shape)}  emb={tuple(emb.shape)}")
    except Exception as e:
        print(f"  {t:45s} {FAIL}  {e}")
        n_fail += 1

    # ── 9. SSL model ──────────────────────────────────────────────────────
    t = "9 MaskedSSL GPU forward + backward"
    try:
        ssl = MaskedSSL(patch_len=32, mask_ratio=0.5).to(device)
        x = torch.randn(4, 1, WINDOW_LEN, device=device)
        recon, loss = ssl(x)
        assert recon.shape == (4, 1, WINDOW_LEN)
        assert torch.isfinite(loss)
        loss.backward()
        print(f"  {t:45s} {PASS}  loss={loss.item():.4f}")
    except Exception as e:
        print(f"  {t:45s} {FAIL}  {e}")
        n_fail += 1

    # ── 10. train_epoch (2 batches) ───────────────────────────────────────
    t = "10 train_epoch (2 batches, GPU)"
    try:
        m = CNN1D(num_classes=4).to(device)
        loader = make_loader("train", "data/processed", batch_size=64)
        crit = nn.CrossEntropyLoss()
        opt = Adam(m.parameters(), lr=1e-3)
        tl, ta = train_epoch(m, loader, crit, opt, device, max_batches=2)
        assert 0.0 <= ta <= 1.0
        print(f"  {t:45s} {PASS}  loss={tl:.4f} acc={ta:.4f}")
    except Exception as e:
        print(f"  {t:45s} {FAIL}  {e}")
        n_fail += 1

    # ── 11. train_epoch bn_eval=True ──────────────────────────────────────
    t = "11 train_epoch bn_eval=True"
    try:
        opt2 = Adam(m.parameters(), lr=1e-3)
        tl2, _ = train_epoch(m, loader, crit, opt2, device,
                              max_batches=2, bn_eval=True)
        assert torch.isfinite(torch.tensor(tl2))
        print(f"  {t:45s} {PASS}  loss={tl2:.4f}")
    except Exception as e:
        print(f"  {t:45s} {FAIL}  {e}")
        n_fail += 1

    # ── 12. evaluate ──────────────────────────────────────────────────────
    t = "12 evaluate (val loader)"
    try:
        val_loader = make_loader("val", "data/processed", batch_size=256)
        y_true, y_pred = evaluate(m, val_loader, device)
        assert y_true.ndim == 1 and y_pred.ndim == 1
        assert len(y_true) == 4255
        print(f"  {t:45s} {PASS}  N={len(y_true)}")
    except Exception as e:
        print(f"  {t:45s} {FAIL}  {e}")
        n_fail += 1

    # ── 13. training_loop (2 epochs, early-stop) ──────────────────────────
    t = "13 training_loop 2 epochs"
    try:
        torch.manual_seed(7)
        m3 = CNN1D(num_classes=4).to(device)
        opt3 = Adam(m3.parameters(), lr=1e-3)
        sch3 = CosineAnnealingLR(opt3, T_max=2)
        crit3 = nn.CrossEntropyLoss()
        best_sd, hist = training_loop(
            m3, loader, val_loader, crit3, opt3, sch3, device,
            epochs=2, patience=5, compute_metrics_fn=simple_metrics,
            max_batches=4, verbose=True,
        )
        assert len(hist) == 2
        assert best_sd is not None
        required_keys = {"epoch", "train_loss", "val_loss", "train_accuracy",
                         "val_accuracy", "train_macro_f1", "val_macro_f1",
                         "learning_rate", "epoch_time_seconds"}
        assert required_keys.issubset(hist[0].keys()), \
            f"Missing keys: {required_keys - set(hist[0].keys())}"
        print(f"  {t:45s} {PASS}  best_val_f1={max(h['val_macro_f1'] for h in hist):.4f}")
    except Exception as e:
        print(f"  {t:45s} {FAIL}  {e}")
        n_fail += 1

    # ── 14. plot_history (graceful without matplotlib) ────────────────────
    t = "14 plot_history (graceful)"
    try:
        plot_history(hist, "/tmp/smoke_history.png", title="Smoke")
        print(f"  {t:45s} {PASS}")
    except Exception as e:
        print(f"  {t:45s} {FAIL}  {e}")
        n_fail += 1

    # ── 15. existing checkpoint compatibility ─────────────────────────────
    t = "15 supervised checkpoint load"
    try:
        ckpts = sorted(glob.glob("models/saved/cnn1d_supervised_*.pt"))
        assert ckpts, "No supervised checkpoints found"
        sd = torch.load(ckpts[-1], map_location="cpu", weights_only=True)
        m4 = CNN1D(num_classes=4)
        m4.load_state_dict(sd)
        print(f"  {t:45s} {PASS}  {Path(ckpts[-1]).name}")
    except Exception as e:
        print(f"  {t:45s} {FAIL}  {e}")
        n_fail += 1

    t = "16 SSL encoder checkpoint load"
    try:
        ssl_ckpts = sorted(glob.glob("models/saved/ssl_encoder_*.pt"), reverse=True)
        assert ssl_ckpts, "No SSL encoder checkpoints found"
        # Find most recent checkpoint with CNN1D encoder keys (some may be TCN)
        cnn_ckpt = None
        for ckpt_path in ssl_ckpts:
            sd_probe = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            if any(k.startswith("0.") for k in sd_probe):
                cnn_ckpt = ckpt_path
                enc_sd = sd_probe
                break
        assert cnn_ckpt, "No CNN1D-compatible ssl_encoder checkpoint found"
        m5 = CNN1D(num_classes=4)
        m5.encoder.load_state_dict(enc_sd)
        print(f"  {t:45s} {PASS}  {Path(cnn_ckpt).name}")
    except Exception as e:
        print(f"  {t:45s} {FAIL}  {e}")
        n_fail += 1

    t = "17 transfer checkpoint load (3-class)"
    try:
        tr_ckpts = sorted(glob.glob("models/saved_transfer_d2/*.pt"))
        if tr_ckpts:
            td = torch.load(tr_ckpts[-1], map_location="cpu", weights_only=True)
            m6 = CNN1D(num_classes=3)
            m6.load_state_dict(td)
            print(f"  {t:45s} {PASS}  {Path(tr_ckpts[-1]).name}")
        else:
            print(f"  {t:45s}  SKIP (no transfer_d2 checkpoints)")
    except Exception as e:
        print(f"  {t:45s} {FAIL}  {e}")
        n_fail += 1

    # ── 18. CompactCNN1D forward ──────────────────────────────────────────
    t = "18 CompactCNN1D (pruned) forward"
    try:
        comp = CompactCNN1D(channels=(16, 32, 64), num_classes=4).to(device)
        out = comp(torch.randn(4, 1, WINDOW_LEN, device=device))
        assert out.shape == (4, 4)
        n_c = sum(p.numel() for p in comp.parameters())
        print(f"  {t:45s} {PASS}  params={n_c:,}")
    except Exception as e:
        print(f"  {t:45s} {FAIL}  {e}")
        n_fail += 1

    # ── 19. GPU memory stress ─────────────────────────────────────────────
    t = "19 GPU memory stress (B=512)"
    try:
        torch.cuda.reset_peak_memory_stats()
        m7 = CNN1D(num_classes=4).to(device)
        xb = torch.randn(512, 1, WINDOW_LEN, device=device)
        _ = m7(xb)
        peak = torch.cuda.max_memory_allocated() / 1e6
        torch.cuda.empty_cache()
        print(f"  {t:45s} {PASS}  peak={peak:.1f} MB")
    except Exception as e:
        print(f"  {t:45s} {FAIL}  {e}")
        n_fail += 1

    # ── 20. CNN1D.forward_sequence ────────────────────────────────────────
    t = "20 CNN1D.forward_sequence shape"
    try:
        m20 = CNN1D(num_classes=4).to(device)
        x20 = torch.randn(4, 1, WINDOW_LEN, device=device)
        feat20 = m20.forward_sequence(x20)
        assert feat20.shape == (4, 128, 129), f"Got {tuple(feat20.shape)}"
        print(f"  {t:45s} {PASS}  feat={tuple(feat20.shape)}")
    except Exception as e:
        print(f"  {t:45s} {FAIL}  {e}")
        n_fail += 1

    # ── 21. LSTM1D shapes ─────────────────────────────────────────────────
    t = "21 LSTM1D forward + encode + forward_sequence"
    try:
        lstm = LSTM1D(num_classes=4).to(device)
        x21 = torch.randn(4, 1, WINDOW_LEN, device=device)
        logits21 = lstm(x21)
        emb21    = lstm.encode(x21)
        seq21    = lstm.forward_sequence(x21)
        assert logits21.shape == (4, 4),           f"logits {tuple(logits21.shape)}"
        assert emb21.shape    == (4, 128),          f"embed {tuple(emb21.shape)}"
        assert seq21.shape    == (4, 128, WINDOW_LEN), f"seq {tuple(seq21.shape)}"
        n_lstm = sum(p.numel() for p in lstm.parameters())
        print(f"  {t:45s} {PASS}  params={n_lstm:,}")
    except Exception as e:
        print(f"  {t:45s} {FAIL}  {e}")
        n_fail += 1

    # ── 22. MaskedSSL with LSTM encoder_model (SequenceDecoder) ──────────
    t = "22 MaskedSSL + LSTM encoder_model (forward + backward)"
    try:
        lstm_base = LSTM1D(num_classes=4)
        ssl22 = MaskedSSL(encoder_model=lstm_base, patch_len=32, mask_ratio=0.5).to(device)
        x22 = torch.randn(4, 1, WINDOW_LEN, device=device)
        recon22, loss22 = ssl22(x22)
        assert recon22.shape == (4, 1, WINDOW_LEN), f"recon {tuple(recon22.shape)}"
        assert torch.isfinite(loss22), f"loss not finite: {loss22.item()}"
        loss22.backward()
        print(f"  {t:45s} {PASS}  loss={loss22.item():.4f}")
    except Exception as e:
        print(f"  {t:45s} {FAIL}  {e}")
        n_fail += 1

    # ── 23. MaskedSSL default path still works (CNN1D encoder) ───────────
    t = "23 MaskedSSL default (no encoder_model) forward"
    try:
        ssl23 = MaskedSSL(patch_len=32, mask_ratio=0.5).to(device)
        x23 = torch.randn(4, 1, WINDOW_LEN, device=device)
        recon23, loss23 = ssl23(x23)
        assert recon23.shape == (4, 1, WINDOW_LEN), f"recon {tuple(recon23.shape)}"
        print(f"  {t:45s} {PASS}  loss={loss23.item():.4f}")
    except Exception as e:
        print(f"  {t:45s} {FAIL}  {e}")
        n_fail += 1

    # ── 24. setup_partial_finetune ────────────────────────────────────────
    t = "24 setup_partial_finetune (lstm + cnn1d)"
    try:
        lstm24 = LSTM1D(num_classes=4)
        n_lstm_partial = setup_partial_finetune(lstm24, "lstm")
        # Only l1 weights + head should be trainable (l0 frozen)
        assert n_lstm_partial < sum(p.numel() for p in lstm24.parameters()), \
            "partial finetune should reduce trainable count"
        # head always trainable
        assert all(p.requires_grad for p in lstm24.head.parameters())

        cnn24 = CNN1D(num_classes=4)
        n_cnn_partial = setup_partial_finetune(cnn24, "cnn1d")
        assert n_cnn_partial < sum(p.numel() for p in cnn24.parameters()), \
            "CNN1D partial finetune should reduce trainable count"
        assert all(p.requires_grad for p in cnn24.head.parameters())

        print(f"  {t:45s} {PASS}  lstm={n_lstm_partial:,}  cnn={n_cnn_partial:,} trainable")
    except Exception as e:
        print(f"  {t:45s} {FAIL}  {e}")
        n_fail += 1

    # ── 25. TCN1D forward + encode shape ──────────────────────────────────
    t = "25 TCN1D forward (CWRU) + encode shapes"
    try:
        tcn25 = TCN1D(num_classes=4).to(device)
        x25 = torch.randn(4, 1, WINDOW_LEN, device=device)
        logits25 = tcn25(x25)
        emb25    = tcn25.encode(x25)
        assert logits25.shape == (4, 4),   f"logits {tuple(logits25.shape)}"
        assert emb25.shape    == (4, 128), f"embed {tuple(emb25.shape)}"
        n_tcn = sum(p.numel() for p in tcn25.parameters())
        assert 150_000 <= n_tcn <= 250_000, f"param count {n_tcn:,} out of range"
        print(f"  {t:45s} {PASS}  params={n_tcn:,}")
    except Exception as e:
        print(f"  {t:45s} {FAIL}  {e}")
        n_fail += 1

    # ── 26. TCN1D Paderborn (3-class) ─────────────────────────────────────
    t = "26 TCN1D forward (Paderborn, 3-class)"
    try:
        tcn26 = TCN1D(num_classes=3).to(device)
        x26 = torch.randn(4, 1, WINDOW_LEN, device=device)
        logits26 = tcn26(x26)
        emb26    = tcn26.encode(x26)
        assert logits26.shape == (4, 3),   f"logits {tuple(logits26.shape)}"
        assert emb26.shape    == (4, 128), f"embed {tuple(emb26.shape)}"
        n_tcn26 = sum(p.numel() for p in tcn26.parameters())
        print(f"  {t:45s} {PASS}  params={n_tcn26:,}")
    except Exception as e:
        print(f"  {t:45s} {FAIL}  {e}")
        n_fail += 1

    # ── 27. TCN1D.forward_sequence shape ─────────────────────────────────
    t = "27 TCN1D.forward_sequence shape"
    try:
        seq27 = tcn25.forward_sequence(torch.randn(4, 1, WINDOW_LEN, device=device))
        assert seq27.shape == (4, 128, WINDOW_LEN), f"Got {tuple(seq27.shape)}"
        print(f"  {t:45s} {PASS}  feat={tuple(seq27.shape)}")
    except Exception as e:
        print(f"  {t:45s} {FAIL}  {e}")
        n_fail += 1

    # ── 28. MaskedSSL + TCN (SequenceDecoder) ────────────────────────────
    t = "28 MaskedSSL + TCN encoder_model (forward + backward)"
    try:
        tcn_base28 = TCN1D(num_classes=4)
        ssl28 = MaskedSSL(encoder_model=tcn_base28, patch_len=32, mask_ratio=0.5).to(device)
        assert type(ssl28.decoder).__name__ == "_SequenceDecoder", \
            f"Expected _SequenceDecoder, got {type(ssl28.decoder).__name__}"
        x28 = torch.randn(4, 1, WINDOW_LEN, device=device)
        recon28, loss28 = ssl28(x28)
        assert recon28.shape == (4, 1, WINDOW_LEN), f"recon {tuple(recon28.shape)}"
        assert torch.isfinite(loss28), f"loss not finite: {loss28.item()}"
        loss28.backward()
        print(f"  {t:45s} {PASS}  loss={loss28.item():.4f}")
    except Exception as e:
        print(f"  {t:45s} {FAIL}  {e}")
        n_fail += 1

    # ── 29. TCN backward pass + finite gradients ──────────────────────────
    t = "29 TCN1D backward + finite gradients"
    try:
        tcn29 = TCN1D(num_classes=4).to(device)
        x29 = torch.randn(4, 1, WINDOW_LEN, device=device)
        logits29 = tcn29(x29)
        loss29 = logits29.sum()
        loss29.backward()
        for name, p in tcn29.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"Non-finite grad at {name}"
        print(f"  {t:45s} {PASS}")
    except Exception as e:
        print(f"  {t:45s} {FAIL}  {e}")
        n_fail += 1

    # ── 30. build_model('tcn') in registry ───────────────────────────────
    t = "30 build_model('tcn') registry construction"
    try:
        tcn30 = build_model("tcn", num_classes=4)
        assert isinstance(tcn30, TCN1D), f"Expected TCN1D, got {type(tcn30)}"
        assert "tcn" in MODEL_REGISTRY, "tcn missing from MODEL_REGISTRY"
        print(f"  {t:45s} {PASS}")
    except Exception as e:
        print(f"  {t:45s} {FAIL}  {e}")
        n_fail += 1

    # ── 31. TCN encoder checkpoint save + reload ──────────────────────────
    t = "31 TCN encoder checkpoint save/reload"
    try:
        import tempfile, os
        tcn31 = TCN1D(num_classes=4)
        ssl31 = MaskedSSL(encoder_model=tcn31, patch_len=32, mask_ratio=0.5)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            tmp31 = f.name
        ssl31.export_encoder_weights(tmp31)
        enc31 = torch.load(tmp31, map_location="cpu", weights_only=True)
        tcn31b = TCN1D(num_classes=4)
        tcn31b.encoder.load_state_dict(enc31)
        for (k, v1), v2 in zip(tcn31.encoder.state_dict().items(),
                               tcn31b.encoder.state_dict().values()):
            assert torch.allclose(v1, v2), f"Weight mismatch at {k}"
        os.unlink(tmp31)
        print(f"  {t:45s} {PASS}")
    except Exception as e:
        print(f"  {t:45s} {FAIL}  {e}")
        n_fail += 1

    # ── 32. TCN linear-probe freezing ────────────────────────────────────
    t = "32 TCN linear-probe (encoder frozen)"
    try:
        tcn32 = TCN1D(num_classes=4)
        for p in tcn32.encoder.parameters():
            p.requires_grad = False
        n_train = sum(p.numel() for p in tcn32.parameters() if p.requires_grad)
        n_head  = sum(p.numel() for p in tcn32.head.parameters())
        assert n_train == n_head, f"Expected {n_head}, got {n_train}"
        print(f"  {t:45s} {PASS}  trainable={n_train:,}")
    except Exception as e:
        print(f"  {t:45s} {FAIL}  {e}")
        n_fail += 1

    # ── 33. TCN partial fine-tuning ───────────────────────────────────────
    t = "33 TCN partial finetune (blocks 0,1 frozen)"
    try:
        tcn33 = TCN1D(num_classes=4)
        n_total33 = sum(p.numel() for p in tcn33.parameters())
        n_partial33 = setup_partial_finetune(tcn33, "tcn")
        assert n_partial33 < n_total33, "partial finetune must reduce trainable count"
        assert all(p.requires_grad for p in tcn33.head.parameters()), "head not trainable"
        for p in tcn33.encoder.blocks[0].parameters():
            assert not p.requires_grad, "block 0 should be frozen"
        for p in tcn33.encoder.blocks[2].parameters():
            assert p.requires_grad, "block 2 should be trainable"
        print(f"  {t:45s} {PASS}  trainable={n_partial33:,}")
    except Exception as e:
        print(f"  {t:45s} {FAIL}  {e}")
        n_fail += 1

    # ── 34. TCN full fine-tuning ──────────────────────────────────────────
    t = "34 TCN full finetune (all params trainable)"
    try:
        tcn34 = TCN1D(num_classes=4)
        n_total34 = sum(p.numel() for p in tcn34.parameters())
        n_req = sum(p.numel() for p in tcn34.parameters() if p.requires_grad)
        assert n_req == n_total34, f"Expected {n_total34:,} trainable, got {n_req:,}"
        print(f"  {t:45s} {PASS}  trainable={n_req:,}")
    except Exception as e:
        print(f"  {t:45s} {FAIL}  {e}")
        n_fail += 1

    # ── 35. TCN cross-dataset classifier replacement ──────────────────────
    t = "35 TCN cross-dataset head replacement"
    try:
        import tempfile, os
        # Save a 4-class CWRU supervised checkpoint
        tcn35_src = TCN1D(num_classes=4)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            tmp35 = f.name
        torch.save(tcn35_src.state_dict(), tmp35)
        # Load encoder into 3-class Paderborn model
        full35 = torch.load(tmp35, map_location="cpu", weights_only=True)
        enc35  = {k[len("encoder."):]: v for k, v in full35.items()
                  if k.startswith("encoder.")}
        tcn35_tgt = TCN1D(num_classes=3)
        missing, unexpected = tcn35_tgt.encoder.load_state_dict(enc35, strict=False)
        assert not missing, f"Missing keys: {missing}"
        assert not unexpected, f"Unexpected keys: {unexpected}"
        os.unlink(tmp35)
        # Forward pass on target shape
        out35 = tcn35_tgt(torch.randn(4, 1, WINDOW_LEN))
        assert out35.shape == (4, 3), f"Expected (4,3), got {tuple(out35.shape)}"
        print(f"  {t:45s} {PASS}  4-class→3-class encoder transfer OK")
    except Exception as e:
        print(f"  {t:45s} {FAIL}  {e}")
        n_fail += 1

    # ── 36. TCN param count in intended range ─────────────────────────────
    t = "36 TCN param count (150 K–250 K)"
    try:
        n4 = sum(p.numel() for p in TCN1D(num_classes=4).parameters())
        n3 = sum(p.numel() for p in TCN1D(num_classes=3).parameters())
        assert n4 == 219_780, f"CWRU params: expected 219,780 got {n4:,}"
        assert n3 == 219_651, f"PU params:   expected 219,651 got {n3:,}"
        assert 150_000 <= n4 <= 250_000, f"Out of range: {n4:,}"
        print(f"  {t:45s} {PASS}  CWRU={n4:,}  PU={n3:,}")
    except Exception as e:
        print(f"  {t:45s} {FAIL}  {e}")
        n_fail += 1

    # ── Result ────────────────────────────────────────────────────────────
    n_tests = 36
    print(f"\n{'='*60}")
    if n_fail == 0:
        print(f"  ALL {n_tests} TESTS PASSED")
    else:
        print(f"  {n_fail} TEST(S) FAILED  (out of {n_tests})")
    print(f"{'='*60}\n")
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    run(torch.device(args.device))
