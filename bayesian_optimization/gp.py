"""Gaussian Process Surrogate Model for Bayesian Optimization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

if TYPE_CHECKING:
    from ConfigSpace import ConfigurationSpace
    from sklearn.gaussian_process.kernels import Kernel
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    FloatHyperparameter,
    IntegerHyperparameter,
    OrdinalHyperparameter,
)


@dataclass
class GPModel:
    """Gaussian Process Surrogate Model."""

    space: ConfigurationSpace
    "The configuration space for the input parameters."

    seed: int = 0
    "Random seed for reproducibility."

    kernel: Kernel = field(init=False)
    "Kernel used in the Gaussian Process."

    model: GaussianProcessRegressor = field(init=False)
    "The Gaussian Process Regressor instance."

    num_restarts: int = 20
    "Number of restarts of the optimizer"

    def __post_init__(self):
        self._create_kernel()
        self._get_model()


    def _create_kernel(self) -> None:
        """Create the kernel for the Gaussian Process."""
        num_cats = num_conts = 0

        for hp in list(self.space.values()):
            if isinstance(hp, (CategoricalHyperparameter, OrdinalHyperparameter)):
                num_cats += 1
            elif isinstance(hp, (FloatHyperparameter, IntegerHyperparameter)):
                num_conts += 1

        assert num_cats == 0, (
            "Categorical and Ordinal hyperparameters are not supported yet in the GP kernel."
        )

        self.kernel = Matern(
            nu=2.5,
        )


    def _get_model(self) -> None:
        """Initialize the Gaussian Process Regressor."""
        self.model = GaussianProcessRegressor(
            kernel=self.kernel,
            n_restarts_optimizer=self.num_restarts,
            random_state=self.seed,
        )