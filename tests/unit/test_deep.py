import torch

from rookie_lab.deep.models import Autoencoder, MLPClassifier, SequenceClassifier, activation


def test_deep_model_shapes() -> None:
    assert MLPClassifier(6)(torch.zeros(4, 6)).shape == (4, 2)
    for cell in ("rnn", "lstm", "gru"):
        assert SequenceClassifier(cell)(torch.zeros(4, 12, 3)).shape == (4, 2)
    assert Autoencoder(7)(torch.zeros(4, 7)).shape == (4, 7)
    assert activation("gelu") is not None
