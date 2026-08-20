"""Train classification architecture zoo and a one-object detector."""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from rookie_lab.vision.models import BasicCNN, TinyObjectDetector, model_zoo


class AugmentedImages(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, x: np.ndarray, y: np.ndarray, augment: bool, seed: int):
        self.x = torch.from_numpy(x.astype("float32") / 255)
        self.y = torch.from_numpy(y.astype("int64"))
        self.augment = augment
        self.generator = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.x[index].clone()
        if self.augment and torch.rand((), generator=self.generator) > 0.5:
            image = image.flip(-1)
        if self.augment:
            noise = torch.randn(image.shape, generator=self.generator) * 0.02
            image = (image + noise).clamp(0, 1)
        return image, self.y[index]


def _train_classifier(
    model: nn.Module,
    train: Dataset[tuple[torch.Tensor, torch.Tensor]],
    valid: TensorDataset,
    epochs: int,
    seed: int,
) -> dict[str, object]:
    torch.manual_seed(seed)
    loader = DataLoader(
        train, batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(seed)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    started = time.perf_counter()
    losses: list[float] = []
    for _ in range(epochs):
        model.train()
        epoch_losses: list[float] = []
        for images, labels in loader:
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
        losses.append(float(np.mean(epoch_losses)))
    model.eval()
    with torch.no_grad():
        logits = model(valid.tensors[0])
        before = time.perf_counter()
        for _ in range(20):
            model(valid.tensors[0][:32])
        inference_ms = (time.perf_counter() - before) * 1000 / 20
    return {
        "accuracy": float(accuracy_score(valid.tensors[1].numpy(), logits.argmax(1).numpy())),
        "parameters": float(sum(value.numel() for value in model.parameters())),
        "fit_seconds": time.perf_counter() - started,
        "batch32_inference_ms": inference_ms,
        "training_loss": losses,
    }


def _iou(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    px1, py1 = (
        torch.minimum(prediction[:, 0], prediction[:, 2]),
        torch.minimum(prediction[:, 1], prediction[:, 3]),
    )
    px2, py2 = (
        torch.maximum(prediction[:, 0], prediction[:, 2]),
        torch.maximum(prediction[:, 1], prediction[:, 3]),
    )
    tx1, ty1, tx2, ty2 = target.unbind(1)
    intersection = (torch.minimum(px2, tx2) - torch.maximum(px1, tx1)).clamp(0) * (
        torch.minimum(py2, ty2) - torch.maximum(py1, ty1)
    ).clamp(0)
    pred_area = (px2 - px1).clamp(0) * (py2 - py1).clamp(0)
    target_area = (tx2 - tx1) * (ty2 - ty1)
    return intersection / (pred_area + target_area - intersection + 1e-7)


def _train_detector(
    data_dir: Path, x: np.ndarray, y: np.ndarray, epochs: int, seed: int
) -> dict[str, object]:
    rows = json.loads((data_dir / "detection_labels.json").read_text())
    boxes = np.array([row["bbox_xyxy"] for row in rows], dtype="float32") / x.shape[-1]
    indices = np.arange(len(x))
    train_idx, valid_idx = train_test_split(indices, test_size=0.25, stratify=y, random_state=seed)
    train = TensorDataset(
        torch.from_numpy(x[train_idx].astype("float32") / 255),
        torch.from_numpy(y[train_idx]),
        torch.from_numpy(boxes[train_idx]),
    )
    model = TinyObjectDetector()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loader = DataLoader(train, batch_size=64, shuffle=True)
    for _ in range(epochs):
        for images, labels, target_boxes in loader:
            optimizer.zero_grad()
            logits, predicted_boxes = model(images)
            loss = nn.functional.cross_entropy(logits, labels) + 4 * nn.functional.smooth_l1_loss(
                predicted_boxes, target_boxes
            )
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        logits, predicted = model(torch.from_numpy(x[valid_idx].astype("float32") / 255))
        iou = _iou(predicted, torch.from_numpy(boxes[valid_idx]))
        probabilities = logits.softmax(1).numpy()
    predicted_class = logits.argmax(1).numpy()
    target_class = y[valid_idx]
    correct_class = predicted_class == target_class
    thresholds = np.arange(0.5, 1.0, 0.05)
    ap_by_threshold: dict[str, float] = {}
    for threshold in thresholds:
        class_ap: list[float] = []
        valid_iou = (iou.numpy() >= threshold).astype(float)
        for class_index in range(probabilities.shape[1]):
            target_binary = (target_class == class_index).astype(int)
            matched_score = probabilities[:, class_index] * valid_iou
            class_ap.append(float(average_precision_score(target_binary, matched_score)))
        ap_by_threshold[f"{threshold:.2f}"] = float(np.mean(class_ap))
    true_positive = int((correct_class & (iou.numpy() >= 0.5)).sum())
    precision = true_positive / max(1, len(valid_idx))
    recall = true_positive / max(1, len(valid_idx))

    failures = np.flatnonzero(~correct_class | (iou.numpy() < 0.5))[:12]
    figure_path = Path("artifacts/figures") / "detection-error-cases.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 4, figsize=(10, 7))
    for axis in axes.flat:
        axis.axis("off")
    for axis, failure in zip(axes.flat, failures, strict=False):
        image = x[valid_idx[failure]].transpose(1, 2, 0)
        axis.imshow(image)
        target_box = boxes[valid_idx[failure]] * x.shape[-1]
        predicted_box = predicted[failure].numpy() * x.shape[-1]
        for box, color in ((target_box, "lime"), (predicted_box, "red")):
            x1, y1, x2, y2 = box
            axis.add_patch(
                plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color=color, linewidth=1.5)
            )
        axis.set_title(
            f"IoU={iou[failure]:.2f} true={target_class[failure]} pred={predicted_class[failure]}"
        )
        axis.axis("off")
    fig.suptitle("Detector errors: green target, red prediction")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)
    return {
        "class_accuracy": float(correct_class.mean()),
        "mean_iou": float(iou.mean()),
        "precision_at_50": precision,
        "recall_at_50": recall,
        "map_at_50": ap_by_threshold["0.50"],
        "map_50_95": float(np.mean(list(ap_by_threshold.values()))),
        "ap_by_iou_threshold": ap_by_threshold,
        "error_figure": str(figure_path),
    }


def _transfer_proxy(
    train: Dataset[tuple[torch.Tensor, torch.Tensor]], valid: TensorDataset, seed: int
) -> dict[str, object]:
    """Compare one-epoch scratch training with a frozen reusable convolutional representation."""
    pretrained = BasicCNN()
    _train_classifier(pretrained, train, valid, epochs=2, seed=seed)
    for parameter in pretrained.features.parameters():
        parameter.requires_grad = False
    pretrained.output.reset_parameters()
    transfer = _train_classifier(pretrained, train, valid, epochs=1, seed=seed)
    scratch = _train_classifier(BasicCNN(), train, valid, epochs=1, seed=seed)
    return {
        "protocol": "two-epoch source pretraining, frozen features, one-epoch head adaptation",
        "frozen_feature_transfer": transfer,
        "one_epoch_from_scratch": scratch,
        "limitation": "same synthetic domain; demonstrates mechanics rather than external transfer",
    }


def run(data_dir: Path, profile: str, seed: int = 20260820) -> dict[str, object]:
    torch.set_num_threads(2)
    payload = np.load(data_dir / "product_images.npz")
    x, y = payload["x"], payload["y"]
    train_idx, valid_idx = train_test_split(
        np.arange(len(x)), test_size=0.25, stratify=y, random_state=seed
    )
    train = AugmentedImages(x[train_idx], y[train_idx], True, seed)
    valid = TensorDataset(
        torch.from_numpy(x[valid_idx].astype("float32") / 255), torch.from_numpy(y[valid_idx])
    )
    epochs = 3 if profile == "ci" else 6
    classification = {
        name: _train_classifier(model, train, valid, epochs, seed) for name, model in model_zoo().items()
    }
    pooling = {
        name: _train_classifier(BasicCNN(pooling=name), train, valid, epochs, seed)
        for name in ("max", "average")
    }
    return {
        "classification": classification,
        "pooling_ablation": pooling,
        "transfer_learning_proxy": _transfer_proxy(train, valid, seed),
        "object_detection": _train_detector(data_dir, x, y, epochs, seed),
        "augmentation": ["horizontal flip", "Gaussian pixel noise"],
        "scope": "educational tiny variants; not canonical full-size ImageNet models",
    }
