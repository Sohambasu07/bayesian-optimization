"""Weighted Expected Improvement acquisition function."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import norm

from bayesian_optimization.plotting import plot_bo_iteration

if TYPE_CHECKING:

    from bayesian_optimization.gp import GPModel


@dataclass
class WeightedExpectedImprovement:
    """Weighted Expected Improvement acquisition function."""

    gp: GPModel
    "The Gaussian Process model used for predictions."

    w: float = 0.5
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

        z = (y_best - mu) / sigma
        exploration_term = norm.pdf(z)
        exploitation_term = norm.cdf(z)
        return (
            self.w*(y_best - mu)*exploitation_term + (1 - self.w)*sigma*exploration_term
        )


    def optimize(
        self,
        x: np.ndarray,
        y: np.ndarray,
        bounds: np.ndarray,
        seed: int = 0,
        n_random: int = 10000,
    ) -> np.ndarray:
        """Optimize the acquisition function to find the next best point.

        Args:
            x: Points where the acquisition function should be evaluated.
            y: Observed objective values corresponding to points X.
            bounds: ndarray of shape (2, ndims) representing the lower and upper bounds
                for each dimension.
            seed: Random seed for reproducibility.
            n_random: Number of random samples to evaluate the acquisition function on.

        Returns:
            The point that maximizes the acquisition function.
        """
        self.gp.model.fit(
            X=x,
            y=y,
        )

        self.gp.space.seed(seed)

        # For better coverage of the whole space,
        # Easy since our search space is 1D
        samples = np.linspace(
            bounds[0],
            bounds[1],
            n_random
        )

        mu, sigma = self.gp.model.predict(samples.reshape(-1, 1), return_std=True)
        y_best = np.min(y)

        acq_values = self.weighted_ei(
            mu=mu,
            sigma=sigma,
            y_best=y_best,
        )

        next_idx = np.argmax(acq_values)

        plot_bo_iteration(
            x=x,
            y=y,
            samples=samples,
            gp_mean=mu,
            gp_std=sigma,
            acq_fn_values=acq_values,
        )

        return samples[next_idx]