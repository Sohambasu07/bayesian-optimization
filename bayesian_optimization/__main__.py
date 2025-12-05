from __future__ import annotations

import argparse

from bayesian_optimization.automl import AutoML

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num_iterations", "-n",
        type=int,
        default=10,
        help="Number of function evaluations for Bayesian Optimization "
        "(counted including initial design).",
    )
    parser.add_argument(
        "--initial_design_size", "-i",
        type=int,
        default=3,
        help="Number of initial design points to sample.",
    )
    parser.add_argument(
        "--train_val_split",
        type=float,
        default=0.8,
        help="Train-validation split ratio.",
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=0,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use for model training and evaluation.",
    )
    parser.add_argument(
        "--num_restarts", "-r",
        type=int,
        default=20,
        help="Number of restarts for the GP.",
    )
    parser.add_argument(
        "--batch_size", "-b",
        type=int,
        default=128,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=20,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging.",
    )

    args = parser.parse_args()

    automl = AutoML(
        num_iterations=args.num_iterations,
        initial_design_size=args.initial_design_size,
        train_val_split=args.train_val_split,
    )

    automl.optimize(
        device=args.device,
        seed=args.seed,
        hp_target="learning_rate",
        batch_size=args.batch_size,
        epochs=args.epochs,
        num_restarts=args.num_restarts,
        debug=args.debug,
    )