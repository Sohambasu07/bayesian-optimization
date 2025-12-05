"""Plotting utilities for Bayesian Optimization."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import seaborn as sns

from bayesian_optimization.constants import PLOTS_DIR

if TYPE_CHECKING:
    import numpy as np

sns.set_style("whitegrid")
sns.set_context("paper")

def plot_gp(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    samples: np.ndarray,
    gp_mean: np.ndarray,
    gp_std: np.ndarray,
) -> None:
    """Plot the Gaussian Process Posterior mean and uncertainty estimates.

    Args:
        ax: Matplotlib Axes object to plot on.
        x: Input points where the GP is evaluated.
        y: Observed objective values at the input points.
        samples: Input points where the GP is evaluated.
        gp_mean: Predicted means at the input points.
        gp_std: Predicted standard deviations at the input points.
    """
    ax.plot(samples, gp_mean, label="GP Mean")
    ax.fill_between(
        samples,
        gp_mean - gp_std,
        gp_mean + gp_std,
        alpha=0.2,
        color="blue",
        label="95% Confidence Interval",
    )
    ax.scatter(
        x,
        y,
        c="red",
        s=50,
        zorder=10,
        label="Observed Data Points",
    )
    ax.set_title("Gaussian Process Posterior")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.legend()


def plot_acquisition(
    ax: plt.Axes,
    samples: np.ndarray,
    acq_fn_values: np.ndarray,
) -> None:
    """Plot the acquisition function values for a given iteration.

    Args:
        ax: Matplotlib Axes object to plot on.
        samples: Input points where the acquisition function is evaluated.
        acq_fn_values: Acquisition function values at the input points.
    """
    ax.plot(
        samples,
        acq_fn_values,
        "r-",
        label="Acquisition Function"
    )
    ax.set_title("Acquisition Function")
    ax.set_xlabel("x")
    ax.set_ylabel("Acquisition Value")
    ax.legend()


def plot_bo_iteration(
    x: np.ndarray,
    y: np.ndarray,
    samples: np.ndarray,
    gp_mean: np.ndarray,
    gp_std: np.ndarray,
    acq_fn_values: np.ndarray,
) -> None:
    """Plot the Gaussian Process and Acquisition Function for a single BO iteration.

    Args:
        x: Input points where the GP is evaluated.
        y: Observed objective values at the input points.
        samples: Input points where the GP and acquisition function are evaluated.
        gp_mean: Predicted means from the GP at the input points.
        gp_std: Predicted standard deviations from the GP at the input points.
        acq_fn_values: Acquisition function values at the input points.
    """
    figure, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    x=x.flatten()
    y=y.flatten()
    gp_mean=gp_mean.flatten()
    gp_std=gp_std.flatten()
    samples=samples.flatten()
    acq_fn_values=acq_fn_values.flatten()

    plot_gp(ax1, x, y, samples, gp_mean, gp_std)
    plot_acquisition(
        ax2,
        samples,
        acq_fn_values
    )

    figure.tight_layout()
    figure.suptitle(f"Bayesian Optimization Iteration {len(y)}", y=1.05)

    plot_path = PLOTS_DIR / f"bo_iteration_{len(y)}.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_path, dpi=300)
    plt.close()



