"""Small ResNet model for image classification."""

from __future__ import annotations

import torch
from torch import nn

from bayesian_optimization.dlmodels.resnet_blocks import ResnetBlock


class ResNetSmall(nn.Module):
    """A small ResNet model."""

    def __init__(self, num_classes: int) -> None:
        """Initialize the small ResNet model."""
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3))
        self.layers.append(nn.BatchNorm2d(64))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.MaxPool2d(kernel_size=3, stride=2))
        for i in range(1, 2):
            self.layers.append(
                ResnetBlock(64*i, 64*(i+1), stride=1, padding=1)
            )
        self.layers.append(nn.AveragePool2d(kernel_size=1))
        self.fc = nn.Linear(64*2, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the small ResNet model."""
        input = x
        for layer in self.layers:
            input = layer(input)
        input = torch.flatten(input, 1)
        output = self.fc(input)
        return self.softmax(output)