import numpy as np
import torch

from src.models import CNN1D, LSTM1D, TCN1D
from src.paderborn_interventions import (
    INPUT_VARIANTS, PhysicalAugment, bearing_balanced_weights, make_input_channels,
)
from src.cv_paderborn import TEST_BEARINGS, make_folds


def test_all_input_variants_have_expected_shape_and_preserve_raw():
    X=np.arange(4*33,dtype=np.float32).reshape(4,33)
    for name,channels in INPUT_VARIANTS.items():
        out=make_input_channels(X,name,window=9)
        assert out.shape==(4,len(channels),33)
        assert np.isfinite(out).all()
        if channels[0]=="raw": assert np.array_equal(out[:,0],X)


def test_moving_filters_do_not_mix_examples():
    X=np.stack([np.zeros(33,dtype=np.float32),np.ones(33,dtype=np.float32)*9])
    out=make_input_channels(X,"raw_mean_median",9)
    assert np.all(out[0]==0); assert np.all(out[1]==9)


def test_invalid_transform_arguments_rejected():
    X=np.zeros((2,32),dtype=np.float32)
    for window in (0,2,4):
        try: make_input_channels(X,"raw",window)
        except ValueError: pass
        else: raise AssertionError("invalid moving window accepted")


def test_bearing_balanced_weights_equalise_total_mass():
    b=np.array(["A"]*2+["B"]*5+["C"]*11)
    w=bearing_balanced_weights(b)
    totals=[w[b==x].sum() for x in np.unique(b)]
    assert np.allclose(totals,totals[0])


def test_augmentation_is_shape_and_finiteness_safe():
    torch.manual_seed(4); x=torch.randn(3,128)
    y=PhysicalAugment(channel_dropout=.5)(x)
    assert y.shape==x.shape; assert torch.isfinite(y).all()


def test_models_accept_multichannel_input():
    x=torch.randn(2,3,128)
    for model in (TCN1D(input_channels=3,num_classes=3),
                  LSTM1D(input_channels=3,num_classes=3),
                  CNN1D(input_channels=3,num_classes=3)):
        assert model(x).shape==(2,3)


def test_confirmation_fold_seed_is_new_and_still_sealed_safe():
    screen=make_folds(4,3,42); confirm=make_folds(4,3,20260728)
    assert [[x["val_bearings"] for x in r] for r in screen] != [[x["val_bearings"] for x in r] for r in confirm]
    for repeat in confirm:
        for fold in repeat:
            assert not ((set(fold["train_bearings"])|set(fold["val_bearings"])) & set(TEST_BEARINGS))
