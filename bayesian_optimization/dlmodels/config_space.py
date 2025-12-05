"""Resnet hyperparameter configuration space."""

from __future__ import annotations

from ConfigSpace import (
    ConfigurationSpace,
    UniformFloatHyperparameter,
)


def get_resnet_config_space(seed: int = 0) -> ConfigurationSpace:
    """Get the configuration space for Resnet hyperparameters."""
    cs = ConfigurationSpace(seed=seed)

    cs.add(
        UniformFloatHyperparameter(
            name="learning_rate",
            lower=1e-5,
            upper=1e-1,
            log=True,
            default_value=1e-3,
        ),
    )

    return cs
