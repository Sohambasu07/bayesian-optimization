"""Main training loop."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import torch

from bayesian_optimization.utils import Average, accuracy

if TYPE_CHECKING:
    from torch.nn import Module
    from torch.nn.modules.loss import _Loss
    from torch.optim import Optimizer
    from torch.utils.data import DataLoader


def train_fn(
    model: Module,
    optimizer: Optimizer,
    criterion: _Loss,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float, float]:
    """Training method.

    Args:
        model: The neural network model to be trained.
        optimizer: The optimizer used for training.
        criterion: The loss function.
        loader: DataLoader providing the training data.
        device: The device (CPU/GPU) to run the training on.

    Returns:
        Tuple containing average training accuracy, average training loss and training time.
    """
    start_time = time.time()
    score = Average()
    losses = Average()
    model.train()

    for images, labels in loader:
        _images = images.to(device)
        _labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(_images).squeeze()
        loss = criterion(logits, _labels)
        loss.backward()
        optimizer.step()

        acc = accuracy(logits, _labels)
        n = _images.size(0)
        losses.update(loss.item(), n)
        score.update(acc.item(), n)

    train_time = time.time() - start_time
    return score.avg, losses.avg, train_time


def eval_fn(
    model: Module,
    criterion: _Loss,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float, float]:
    """Evaluate the model.

    Args:
        model: The neural network model to be evaluated.
        criterion: The loss function.
        loader: DataLoader providing the evaluation data.
        device: The device (CPU/GPU) to run the evaluation on.

    Returns:
        Tuple containing average evaluation accuracy, average evaluation loss and evaluation time.
    """
    start_time = time.time()
    score = Average()
    losses = Average()
    model.eval()

    with torch.no_grad():
        for images, labels in loader:
            _images = images.to(device)
            _labels = labels.to(device)

            logits = model(_images).squeeze()
            loss = criterion(logits, _labels)

            acc = accuracy(logits, _labels)
            n = _images.size(0)
            losses.update(loss.item(), n)
            score.update(acc.item(), n)

    eval_time = time.time() - start_time

    return score.avg, losses.avg, eval_time