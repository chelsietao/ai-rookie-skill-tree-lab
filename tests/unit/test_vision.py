import pytest
import torch

from rookie_lab.vision.experiment import _iou
from rookie_lab.vision.models import TinyObjectDetector, model_zoo


def test_vision_model_shapes() -> None:
    inputs = torch.zeros(2, 3, 32, 32)
    for model in model_zoo().values():
        assert model(inputs).shape == (2, 3)
    classes, boxes = TinyObjectDetector()(inputs)
    assert classes.shape == (2, 3)
    assert boxes.shape == (2, 4)


def test_model_zoo_contains_distinct_parameter_budgets() -> None:
    counts = {
        name: sum(parameter.numel() for parameter in model.parameters())
        for name, model in model_zoo().items()
    }
    assert len(counts) == 7
    assert len(set(counts.values())) >= 5


def test_iou_exact_and_disjoint_boxes() -> None:
    prediction = torch.tensor([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.2, 0.2]])
    target = torch.tensor([[0.0, 0.0, 1.0, 1.0], [0.8, 0.8, 1.0, 1.0]])
    values = _iou(prediction, target)
    assert float(values[0]) == pytest.approx(1.0, abs=1e-6)
    assert float(values[1]) == pytest.approx(0.0)
