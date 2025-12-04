"""Initial design strategy for Bayesian optimization by sampling points
from a Sobol sequence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import qmc


@dataclass
class InitialDesign:
    """Initial design strategy that samples points from a Sobol sequence."""

    ndims: int
    """Number of dimensions of the search space."""

    def sample(
        self,
        n: int,
        bounds: np.ndarray,
        seed: int = 0,
    ) -> np.ndarray:
        """Sample `n` points from a Sobol sequence within the given bounds.

        Args:
            n: Number of points to sample.
            bounds: ndarray of shape (2, ndims) representing the lower and upper bounds
                for each dimension.
            seed: Random seed for reproducibility.

        Returns:
            ndarray of shape (n, ndims) containing the sampled points.
        """
        sampler = qmc.Sobol(d=self.ndims, seed=seed)
        samples = sampler.random_base2(m=int(np.ceil(np.log2(n))))
        return qmc.scale(samples, l_bounds = bounds[0], u_bounds = bounds[1])

