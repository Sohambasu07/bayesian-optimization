from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from ConfigSpace import ConfigurationSpace


@dataclass
class Trial:
    """A dataclass to store information about a single trial."""

    trial_id: int
    """Unique identifier for the trial."""

    config: dict[str, float | int]
    """Hyperparameter configuration used in the trial."""

    eval_result: float | None = field(default=None, init=False)
    """Evaluation result obtained from the trial."""

    complete: bool = field(default=False, init=False)
    """Indicates whether the trial has been completed."""

    def _set_as_complete(
        self,
        eval_result: float,
    ) -> None:
        """Mark the trial as complete and store evaluation results.

        Args:
            eval_result: The evaluation result obtained from the trial.
        """
        self.eval_result = eval_result
        self.complete = True


    @property
    def config_array(self) -> np.ndarray:
        """Get the hyperparameter configuration as a numpy array.

        Returns:
            A numpy array representing the hyperparameter configuration.
        """
        return np.array(list(self.config.values()))


    @classmethod
    def _convert_to_trial(
        cls,
        trial_id: int,
        config_space: ConfigurationSpace,
        config_array: np.ndarray,
    ) -> Trial:
        """Convert a numpy array configuration to a Trial object.

        Args:
            trial_id: The unique identifier for the trial.
            config_space: The ConfigurationSpace object.
            config_array: The numpy array representing the configuration.

        Returns:
            A Trial object with the given configuration.
        """
        config: dict[str, float | int] = {}

        for i, hp in enumerate(config_space.values()):
            config[hp.name] = config_array[i].item()

        return Trial(
            trial_id=trial_id,
            config=config
        )


def get_all_trials_as_arrays(
    trials: list[Trial],
) -> tuple[np.ndarray, np.ndarray]:
    """Get a 2D numpy array of all trial configurations.

    Args:
        trials: A list of Trial objects.

    Returns:
        A tuple containing the trial configs as a 2D numpy array and their corresponding
        evaluation results as a 1D numpy array.
    """
    return (
        np.array([trial.config_array for trial in trials]),
        np.array([trial.eval_result for trial in trials]),
    )


@dataclass
class Average:
    """Class to compute sum and average."""

    sum: float = field(default=0.0, init=False)
    """Sum of some values."""

    count: int = field(default=0, init=False)
    """Number of values."""

    def update(self, value: float, n: int = 1) -> None:  # noqa: D417
        """Update the sum and count.

        Args:
            value: The new value to add.
        """
        self.sum += value * n
        self.count += n

    @property
    def avg(self) -> float:
        """Compute the average.

        Returns:
            The average of the values.
        """
        if self.count == 0:
            return 0.0
        return self.sum / self.count


def accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Compute the accuracy of predictions.

    Args:
        logits: The model output logits.
        labels: The true labels.

    Returns:
        The accuracy as a float.
    """
    return torch.sum(torch.argmax(logits, dim=1) == labels).float() / labels.size(0)