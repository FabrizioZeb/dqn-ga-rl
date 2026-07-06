"""Experiment configuration, reproducibility helpers, and plot label constants."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LunarLander-v3 action semantics (for plot annotations)
ACTION_LABELS = [
    "noop",
    "left engine",
    "main engine",
    "right engine",
]

STATE_LABELS = [
    r"$x$",
    r"$y$",
    r"$v_x$",
    r"$v_y$",
    r"$\theta$",
    r"$\omega$",
    "leg₁",
    "leg₂",
]


@dataclass
class ExperimentConfig:
    """Hyperparameters and I/O paths for reproducibility."""

    seed: int = 42
    env_name: str = "LunarLander-v3"
    solved_threshold: float = 200.0  # Gymnasium "solved" criterion

    # DQN
    dqn_episodes: int = 1000
    dqn_gamma: float = 0.99
    dqn_lr: float = 1e-3
    dqn_batch_size: int = 64
    dqn_buffer_capacity: int = 100_000
    dqn_warmup_steps: int = 5000
    dqn_target_sync: int = 1000
    dqn_epsilon_start: float = 1.0
    dqn_epsilon_decay: float = 0.995
    dqn_epsilon_min: float = 0.05

    # Genetic algorithm
    ga_population_size: int = 100
    ga_n_elite: int = 10
    ga_generations: int = 150
    ga_mutation_std: float = 0.02
    ga_eval_episodes: int = 1  # fitness rollouts per genome

    # Knowledge distillation
    kd_episodes: int = 100
    kd_epochs: int = 50
    kd_lr: float = 1e-3

    # Evaluation & visualization
    eval_episodes: int = 20
    moving_avg_window: int = 20
    results_dir: Path = field(default_factory=lambda: Path("results/lunar_lander"))
    save_figures: bool = True
    show_figures: bool = True
    train_render: bool = False


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
