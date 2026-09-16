import torch
from src.models.cnn1d import CNN1D
from src.models.masked_ssl import MaskedSSL
def test_ssl_auxiliary_losses_are_finite_and_weighted():
 x=torch.randn(4,1,1024)
 for fw,cw in ((0,0),(.1,0),(0,.1)):
  torch.manual_seed(42);m=MaskedSSL(encoder_model=CNN1D(4),frequency_weight=fw,consistency_weight=cw);_,loss=m(x);parts=m.last_loss_components
  assert torch.isfinite(loss) and all(torch.isfinite(v) for v in parts.values())
  assert torch.allclose(loss,parts['time_loss']+fw*parts['frequency_loss']+cw*parts['consistency_loss'])
def test_o0_is_masked_position_only_o1():
 m=MaskedSSL(encoder_model=CNN1D(4));recon=torch.zeros(1,1,1024);target=torch.zeros_like(recon);target[...,0]=10;mask=torch.zeros(1,32,dtype=torch.bool);mask[:,1]=True
 assert m._masked_mse(recon,target,mask)==0
