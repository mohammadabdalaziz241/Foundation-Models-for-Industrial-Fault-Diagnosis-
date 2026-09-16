import pytest
from src.foundation_adaptation import configure_adaptation
from src.models import TCN1D,LSTM1D,CNN1D

@pytest.mark.parametrize("arch,model",[("tcn",TCN1D(num_classes=3)),("lstm",LSTM1D(num_classes=3)),("cnn1d",CNN1D(num_classes=3))])
@pytest.mark.parametrize("regime",["linear","last_block","last_two_blocks","full"])
def test_adaptation_masks_are_nested_and_include_head(arch,model,regime):
    names=configure_adaptation(model,arch,regime)
    assert any(x.startswith("head.") for x in names)
    if regime=="linear": assert all(x.startswith("head.") for x in names)
    if regime=="full": assert len(names)==len(list(model.named_parameters()))

def test_bad_regime_is_rejected():
    with pytest.raises(ValueError): configure_adaptation(TCN1D(),"tcn","unknown")
