import pytest
import torch

from rookie_lab.deep.models import Autoencoder, MLPClassifier, SequenceClassifier, activation, initialize


def test_deep_model_shapes() -> None:
    assert MLPClassifier(6)(torch.zeros(4, 6)).shape == (4, 2)
    for cell in ("rnn", "lstm", "gru"):
        assert SequenceClassifier(cell)(torch.zeros(4, 12, 3)).shape == (4, 2)
    assert Autoencoder(7)(torch.zeros(4, 7)).shape == (4, 7)
    assert activation("gelu") is not None


@pytest.mark.parametrize("norm", ["none", "batch", "layer", "group"])
def test_all_normalization_variants(norm: str) -> None:
    assert MLPClassifier(6, norm=norm)(torch.ones(8, 6)).shape == (8, 2)


def test_initialization_zeros_biases() -> None:
    model = MLPClassifier(6)
    initialize(model, "xavier")
    biases = [module.bias for module in model.modules() if isinstance(module, torch.nn.Linear)]
    assert all(torch.count_nonzero(bias) == 0 for bias in biases)


def test_invalid_deep_choices_fail_loudly() -> None:
    with pytest.raises(ValueError, match="activation"):
        activation("mystery")
    with pytest.raises(ValueError, match="normalization"):
        MLPClassifier(6, norm="mystery")
    with pytest.raises(ValueError, match="recurrent"):
        SequenceClassifier("mystery")
