"""Uniform experience replay for the DQN agent."""

from __future__ import annotations

import random
from collections import deque
from typing import Tuple

import numpy as np
import torch

from .config import DEVICE


class ReplayBuffer:
    """
    Uniform replay buffer D storing transitions (s, a, r, s', done).

    Uniform sampling approximates i.i.d. draws from the visitation distribution,
    reducing variance in the Bellman target estimator compared to on-line TD(0).
    """

    def __init__(self, capacity: int) -> None:
        self.buffer: deque = deque(maxlen=capacity)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.tensor(np.array(states), dtype=torch.float32, device=DEVICE)
        actions_t = torch.tensor(actions, dtype=torch.long, device=DEVICE)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)
        next_states_t = torch.tensor(np.array(next_states), dtype=torch.float32, device=DEVICE)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=DEVICE)

        return states_t, actions_t, rewards_t, next_states_t, dones_t

    def __len__(self) -> int:
        return len(self.buffer)
