"""Run Bayesian Optimization on a Deep Learning pipeline."""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Literal

import pandas as pd
import torch
import torchsummary as summary
import yaml
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor

from bayesian_optimization.bayesian_optimization import BayesianOptimization
from bayesian_optimization.constants import DATA_DIR, RESULTS_DIR
from bayesian_optimization.dlmodels.config_space import get_resnet_config_space
from bayesian_optimization.dlmodels.resnet_small import ResNetSmall
from bayesian_optimization.dlmodels.train_and_eval import eval_fn, train_fn

if TYPE_CHECKING:
    from bayesian_optimization.utils import Trial


automl_logger = logging.getLogger(__name__)
automl_logger.setLevel(logging.INFO)


@dataclass
class AutoML:
    """Class to run Bayesian Optimization on a Deep Learning pipeline."""

    num_iterations: int = 10
    "Number of function evaluations for Bayesian Optimization (counted including initial design)."

    initial_design_size: int = 3
    "Number of initial design points to sample."

    train_val_split: float = 0.8
    "Train-validation split ratio."

    def _download_and_prepare_data(
        self,
        seed: int = 0,
    ) -> tuple[tuple[torch.utils.data.Dataset, torch.utils.data.Dataset], int]:
        """Download and prepare the FashionMNIST dataset.

        Returns:
            A tuple containing training and validation datasets, and the number of classes.
        """
        dataset = FashionMNIST(
            root=DATA_DIR,
            train=True,
            download=True,
            transform=ToTensor(),
        )
        num_classes = 10

        train_size = int(self.train_val_split * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(seed)
        )

        return (train_dataset, val_dataset), num_classes


    def optimize(
        self,
        device: Literal["cpu", "cuda"] = "cpu",
        seed: int = 0,
        hp_target: str = "learning_rate",
        batch_size: int = 64,
        epochs: int = 10,
        num_restarts: int = 20,
        *,
        debug: bool = False,
    ) -> None:
        """Run Bayesian Optimization to find the best hyperparameters."""
        config_space = get_resnet_config_space(seed=seed)

        (train_dataset, val_dataset), num_classes = self._download_and_prepare_data(seed=seed)

        bo = BayesianOptimization(
            config_space=config_space,
            num_iterations=self.num_iterations,
            initial_design_size=self.initial_design_size,
            num_restarts=num_restarts,
            device=device,
            eval_fn=partial(
                train_and_evaluate_model,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                num_classes=num_classes,
                batch_size=batch_size,
                epochs=epochs,
            ),
        )

        trials: list[Trial] = bo(seed=seed, debug=debug)

        _df = pd.DataFrame(
            [
                {
                    "trial_id": trial.trial_id,
                    **trial.config,
                    "val_loss": trial.eval_result,
                }
                for trial in trials
            ]
        )

        automl_logger.info("\nBest Found:")
        best_trial = min(trials, key=lambda t: t.eval_result)

        automl_logger.info(f"Trial ID: {best_trial.trial_id}")
        automl_logger.info(f"\nOptimal {hp_target}: {best_trial.config[hp_target]}")
        automl_logger.info(f"Best Validation Loss: {best_trial.eval_result}")

        best_hp_config_path = RESULTS_DIR / "best_config.yaml"
        best_hp_config_path.parent.mkdir(parents=True, exist_ok=True)

        _df.to_csv(RESULTS_DIR / "bo_results.csv", index=False)

        with best_hp_config_path.open("w") as f:
            yaml.dump(best_trial.config, f)

        logging.info(
            f"Best hyperparameter configuration saved to {best_hp_config_path.absolute().resolve()}"
        )


def train_and_evaluate_model(
    train_dataset: torch.utils.data.Dataset,
    val_dataset: torch.utils.data.Dataset,
    num_classes: int,
    hp_configs: dict[str, float],
    batch_size: int = 64,
    device: Literal["cpu", "cuda"] = "cpu",
    seed: int = 0,
    epochs: int = 10,
    *,
    show_summary: bool = False,
) -> float:
    """Train the Resnet model using the prepared dataset.

    Args:
        train_dataset: The training dataset.
        val_dataset: The validation dataset.
        num_classes: Number of output classes.
        hp_configs: Hyperparameter configurations.
        batch_size: Batch size for training and evaluation.
        device: The device (CPU/GPU) to run the training on.
        seed: Random seed for reproducibility.
        epochs: Number of training epochs.
        show_summary: Whether to display the model summary.
    """
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )

    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        warnings.warn("CUDA is not available. Falling back to CPU.", stacklevel=2)
        device = torch.device("cpu")

    input_shape = (1, 32, 32)

    model = ResNetSmall(
        num_classes=num_classes,
        input_shape=input_shape
    ).to(device)

    if show_summary:
        summary.summary(model, input_shape, batch_size=1, device=str(device))
    criterion = torch.nn.CrossEntropyLoss()

    lr = hp_configs["learning_rate"]
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
    )

    total_time = 0.0

    for _ in range(epochs):

        _, _, train_time = train_fn(
            model,
            optimizer,
            criterion,
            train_loader,
            device,
        )

        val_accuracy, val_loss, val_time = eval_fn(
            model,
            criterion,
            val_loader,
            device,
        )

        total_time += train_time + val_time


    automl_logger.info(
        f"Validation Accuracy: {val_accuracy:.4f}, "
        f"Validation Loss: {val_loss:.4f}, "
        f"Total Time: {total_time:.2f} seconds"
    )

    return val_loss