"""Educational small versions of the official CNN architecture names."""

from __future__ import annotations

import torch
from torch import nn


class BasicCNN(nn.Module):
    def __init__(self, classes: int = 3, pooling: str = "max"):
        super().__init__()
        pool: nn.Module = nn.MaxPool2d(2) if pooling == "max" else nn.AvgPool2d(2)
        self.features = nn.Sequential(
            nn.Conv2d(3, 12, 3, padding=1),
            nn.ReLU(),
            pool,
            nn.Conv2d(12, 24, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.output = nn.Linear(24, classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.output(self.features(inputs).flatten(1))


class TinyVGG(nn.Module):
    def __init__(self, classes: int = 3):
        super().__init__()
        layers: list[nn.Module] = []
        channels = (3, 12, 12, 24, 24)
        for index in range(len(channels) - 1):
            layers.extend([nn.Conv2d(channels[index], channels[index + 1], 3, padding=1), nn.ReLU()])
            if index in {1, 3}:
                layers.append(nn.MaxPool2d(2))
        self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1))
        self.output = nn.Linear(24, classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.output(self.features(inputs).flatten(1))


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.relu(inputs + self.block(inputs))


class TinyResNet(nn.Module):
    def __init__(self, classes: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 20, 3, padding=1),
            ResidualBlock(20),
            nn.MaxPool2d(2),
            ResidualBlock(20),
            nn.AdaptiveAvgPool2d(1),
        )
        self.output = nn.Linear(20, classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.output(self.features(inputs).flatten(1))


class DepthwiseSeparable(nn.Module):
    def __init__(self, inputs: int, outputs: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(inputs, inputs, 3, stride=stride, padding=1, groups=inputs),
            nn.BatchNorm2d(inputs),
            nn.ReLU6(),
            nn.Conv2d(inputs, outputs, 1),
            nn.BatchNorm2d(outputs),
            nn.ReLU6(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class TinyMobileNet(nn.Module):
    def __init__(self, classes: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 12, 3, padding=1),
            nn.ReLU6(),
            DepthwiseSeparable(12, 20, 2),
            DepthwiseSeparable(20, 28, 2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.output = nn.Linear(28, classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.output(self.features(inputs).flatten(1))


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.scale = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(2, channels // 4), 1),
            nn.SiLU(),
            nn.Conv2d(max(2, channels // 4), channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs * self.scale(inputs)


class TinyEfficientNet(nn.Module):
    def __init__(self, classes: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.SiLU(),
            DepthwiseSeparable(16, 24, 2),
            SqueezeExcitation(24),
            DepthwiseSeparable(24, 32, 2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.output = nn.Linear(32, classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.output(self.features(inputs).flatten(1))


class Fire(nn.Module):
    def __init__(self, inputs: int, squeeze: int, expand: int):
        super().__init__()
        self.squeeze = nn.Sequential(nn.Conv2d(inputs, squeeze, 1), nn.ReLU())
        self.expand1 = nn.Conv2d(squeeze, expand, 1)
        self.expand3 = nn.Conv2d(squeeze, expand, 3, padding=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        squeezed = self.squeeze(inputs)
        return torch.relu(torch.cat([self.expand1(squeezed), self.expand3(squeezed)], dim=1))


class TinySqueezeNet(nn.Module):
    def __init__(self, classes: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            Fire(16, 8, 12),
            nn.MaxPool2d(2),
            Fire(24, 8, 16),
            nn.AdaptiveAvgPool2d(1),
        )
        self.output = nn.Linear(32, classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.output(self.features(inputs).flatten(1))


class TinyViT(nn.Module):
    def __init__(self, classes: int = 3, image_size: int = 32, patch: int = 8, dimensions: int = 32):
        super().__init__()
        self.patch = nn.Conv2d(3, dimensions, patch, stride=patch)
        patches = (image_size // patch) ** 2
        self.position = nn.Parameter(torch.zeros(1, patches, dimensions))
        layer = nn.TransformerEncoderLayer(dimensions, nhead=4, dim_feedforward=64, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.output = nn.Linear(dimensions, classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        tokens = self.patch(inputs).flatten(2).transpose(1, 2) + self.position
        return self.output(self.encoder(tokens).mean(1))


class TinyObjectDetector(nn.Module):
    def __init__(self, classes: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 24, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(24, classes)
        self.box = nn.Sequential(nn.Linear(24, 4), nn.Sigmoid())

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features(inputs).flatten(1)
        return self.classifier(features), self.box(features)


def model_zoo() -> dict[str, nn.Module]:
    return {
        "CNN": BasicCNN(),
        "VGG": TinyVGG(),
        "ResNet": TinyResNet(),
        "MobileNet": TinyMobileNet(),
        "EfficientNet": TinyEfficientNet(),
        "SqueezeNet": TinySqueezeNet(),
        "ViT": TinyViT(),
    }
