"""Core Bayesian Optimization module."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from bayesian_optimization.gp import GPModel
from bayesian_optimization.initial_design import InitialDesign
from bayesian_optimization.utils import Trial, get_all_trials_as_arrays
from bayesian_optimization.weighted_ei import WeightedExpectedImprovement

if TYPE_CHECKING:
    from ConfigSpace import ConfigurationSpace


bo_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

@dataclass
class BayesianOptimization:
    """Core class for Bayesian Optimization."""

    config_space: ConfigurationSpace
    "The configuration space for the optimization."

    objective: str
    "The objective metric name."

    eval_fn: Callable
    "The evaluation function that takes hyperparameter configurations and "
    "returns their performance."

    num_iterations: int = 10
    "Number of function evaluations for Bayesian Optimization (counted including initial design)."

    initial_design_size: int = 3
    "Number of initial design points to sample."

    num_restarts: int = 20
    "Number of restarts for the GP"

    device: Literal["cpu", "cuda"] = "cpu"
    "Device to use for model training and evaluation."

    def __call__(
        self,
        seed: int = 0,
    ) -> list[Trial]:
        """Run Bayesian Optimization.

        Args:
            eval_fn: The evaluation function that takes hyperparameter configurations
                and returns their performance.
            seed: Random seed for reproducibility.

        Returns:
            A list of completed Trial objects.
        """
        # Run Initial Design

        initial_design = InitialDesign(
            ndims=len(list(self.config_space.values())),
        )

        cs_bounds = np.array(
            [
                np.array([hp.lower, hp.upper])
                for hp in self.config_space.values()
                if hasattr(hp, "lower") and hasattr(hp, "upper")
            ]
        ).T

        initial_points = initial_design.sample(
            n=self.initial_design_size,
            bounds=cs_bounds
        )

        trials: list[Trial] = []

        # Evaluate Initial Design Points

        for i, sample in enumerate(initial_points):
            _trial = Trial._convert_to_trial(
                trial_id=i,
                config_space=self.config_space,
                config_array=sample
            )

            bo_logger.info(f"BO iteration {i+1}\n")
            eval_cost = self.eval_fn(
                hp_configs=_trial.config,
                device=self.device,
                show_summary=i == 0,
            )[self.objective]
            _trial._set_as_complete(eval_cost)
            trials.append(_trial)
            print("===========================================================")

        for _ in range(self.num_iterations - self.initial_design_size):

            # Fit GP Model
            gp = GPModel(
                seed=seed,
                num_restarts=self.num_restarts,
                space=self.config_space,
            )

            X, y = get_all_trials_as_arrays(trials)

            # Define Acquisition Function
            acq_fn = WeightedExpectedImprovement(
                gp=gp,
                xi=0.01,
            )

            # Optimize Acquisition Function
            next_point = acq_fn.optimize(
                x=X,
                y=y,
                seed=seed,
                bounds=cs_bounds,
            )

            # Evaluate new point
            next_trial = Trial._convert_to_trial(
                trial_id=len(trials),
                config_space=self.config_space,
                config_array=next_point,
            )

            # Add new trial to trials list
            bo_logger.info(f"BO iteration {len(trials)+1}\n")
            eval_cost = self.eval_fn(
                hp_configs=next_trial.config,
                device=self.device,
            )[self.objective]
            next_trial._set_as_complete(eval_cost)
            trials.append(next_trial)
            print("===========================================================")

        return trials