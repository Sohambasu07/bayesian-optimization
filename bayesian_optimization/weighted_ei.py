"""Weighted Expected Improvement acquisition function."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import norm

if TYPE_CHECKING:
    from ConfigSpace import Configuration

    from bayesian_optimization.gp import GPModel


@dataclass
class WeightedExpectedImprovement:
    """Weighted Expected Improvement acquisition function."""

    gp: GPModel
    "The Gaussian Process model used for predictions."

    xi: float = 0.01
    "Exploration-exploitation trade-off parameter."

    w: float = 0.35
    "Weighting factor for the weighted expected improvement."

    def weighted_ei(
        self,
        mu: np.ndarray,
        sigma: np.ndarray,
        y_best: float,
    ) -> np.ndarray:
        """Compute the Expected Improvement at points X.

        Args:
            mu: Predicted means at points where the acquisition function should be evaluated.
            sigma: Predicted standard deviations at points where the acquisition function
                should be evaluated.
            y_best: The best observed objective value so far.

        Returns:
            Expected Improvement values at points X.
        """
        if np.any(sigma == 0):
            return np.zeros_like(mu)

        z = (y_best - mu - self.xi) / sigma
        exploration_term = norm.pdf(z)
        exploitation_term = norm.cdf(z)
        return (
            self.w*(y_best - mu - self.xi)*exploitation_term + (1 - self.w)*sigma*exploration_term
        )


    def optimize(
        self,
        x: np.ndarray,
        y: np.ndarray,
        seed: int = 0,
        n_random: int = 10000,
    ) -> np.ndarray:
        """Optimize the acquisition function to find the next best point.

        Args:
            x: Points where the acquisition function should be evaluated.
            y: Observed objective values corresponding to points X.
            seed: Random seed for reproducibility.
            n_random: Number of random samples to evaluate the acquisition function on.

        Returns:
            The point that maximizes the acquisition function.
        """
        self.gp.model.fit(
            x=x,
            y=y,
        )

        self.gp.space.seed(seed)

        random_samples: list[Configuration] = self.gp.space.sample_configuration(
            n_samples=n_random
        )

        random_samples: np.ndarray = np.array(
            [sample.get_array() for sample in random_samples]
        )

        mu, sigma = self.gp.model.predict(random_samples, return_std=True)
        y_best = np.min(y)

        acq_values = self.weighted_ei(
            mu=mu,
            sigma=sigma,
            y_best=y_best,
        )

        next_idx = np.argmax(acq_values)

        return random_samples[next_idx]