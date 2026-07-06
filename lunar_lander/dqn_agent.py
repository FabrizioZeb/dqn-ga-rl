"""Deep Q-Network agent: online/target networks, ε-greedy policy, TD update."""

from __future__ import annotations

import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .config import DEVICE, ExperimentConfig
from .networks import QNetwork
from .replay_buffer import ReplayBuffer


class DQNAgent:
    """
    Deep Q-Network with target network and ε-greedy exploration.

    ``train_step`` implements one SGD update on the MSE loss between
    Q_θ(s,a) and the one-step Bellman target computed with θ⁻.
    """

    def __init__(self, state_dim: int, action_dim: int, config: ExperimentConfig) -> None:
        self.config = config
        self.gamma = config.dqn_gamma
        self.batch_size = config.dqn_batch_size
        self.target_sync = config.dqn_target_sync
        self.warmup_steps = config.dqn_warmup_steps

        self.online_net = QNetwork(state_dim, action_dim).to(DEVICE)
        self.target_net = QNetwork(state_dim, action_dim).to(DEVICE)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=config.dqn_lr)

        self.epsilon = config.dqn_epsilon_start
        self.epsilon_decay = config.dqn_epsilon_decay
        self.epsilon_min = config.dqn_epsilon_min

        self.replay_buffer = ReplayBuffer(config.dqn_buffer_capacity)
        self.total_steps = 0
        self._action_dim = action_dim

    def act(self, state: np.ndarray, training: bool = True) -> int:
        if training and random.random() < self.epsilon:
            return random.randrange(self._action_dim)

        state_tensor = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            q_values = self.online_net(state_tensor)
        return int(q_values.argmax(dim=1).item())

    def train_step(self) -> Optional[float]:
        if len(self.replay_buffer) < self.warmup_steps:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Q_θ(s,a) for taken actions
        current_q = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target: r + γ max_a' Q_{θ⁻}(s', a')  (bootstrap with target net)
        with torch.no_grad():
            next_q = self.target_net(next_states)
            max_next_q = next_q.max(dim=1)[0]
            target_q = rewards + self.gamma * max_next_q * (1.0 - dones)

        loss = nn.functional.mse_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()

        if self.total_steps % self.target_sync == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return float(loss.item())

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
