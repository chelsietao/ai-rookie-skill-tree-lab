import torch

from rookie_lab.vision.models import TinyObjectDetector, model_zoo


def test_vision_model_shapes() -> None:
    inputs = torch.zeros(2, 3, 32, 32)
    for model in model_zoo().values():
        assert model(inputs).shape == (2, 3)
    classes, boxes = TinyObjectDetector()(inputs)
    assert classes.shape == (2, 3)
    assert boxes.shape == (2, 4)
