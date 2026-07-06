"""Neural function approximators used by the DQN agent, the GA, and distillation."""

from __future__ import annotations

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """
    MLP parameterising Q_θ(s, ·) ∈ ℝ^{|A|}.

    Architecture: state_dim → 128 → 128 → |A| with ReLU activations.
    Output heads are *action values* (not a policy); greedy action selection
    uses argmax_a Q_θ(s,a).
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PolicyNetwork(nn.Module):
    """
    Direct policy network for neuroevolution.

    Uses Tanh activations (bounded pre-activations) and argmax over logits
    for deterministic discrete control. Weights are evolved in ℝ^d via GA.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StudentNetwork(nn.Module):
    """Compact network for knowledge distillation from the DQN teacher."""

    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
