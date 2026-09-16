import torch
from src.foundation_objectives import AdversarialHead,gradient_reverse,nt_xent
from src.models import TCN1D
from src.models.masked_ssl import MaskedSSL

def test_gradient_reversal_changes_only_gradient_sign():
    x=torch.tensor([[1.,2.]],requires_grad=True); y=gradient_reverse(x,.25)
    assert torch.equal(x,y); y.sum().backward(); assert torch.allclose(x.grad,torch.full_like(x,-.25))

def test_adversarial_head_shapes():
    assert AdversarialHead(128,7)(torch.randn(5,128)).shape==(5,7)

def test_nt_xent_prefers_aligned_pairs():
    torch.manual_seed(1); z=torch.randn(16,32)
    assert nt_xent(z,z+.01*torch.randn_like(z)) < nt_xent(z,z[torch.randperm(len(z))])

def test_masked_ssl_supports_three_channels():
    base=TCN1D(input_channels=3,num_classes=3); ssl=MaskedSSL(encoder_model=base,input_channels=3)
    x=torch.randn(2,3,1024); recon,loss=ssl(x)
    assert recon.shape==x.shape and loss.ndim==0 and torch.isfinite(loss)
