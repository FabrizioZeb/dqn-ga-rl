"""Offline knowledge distillation from a DQN teacher into a compact student network."""

from __future__ import annotations

from typing import Callable, List, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .config import DEVICE, ExperimentConfig
from .networks import StudentNetwork


class KnowledgeDistiller:
    """
    Trains a compact ``StudentNetwork`` to match a DQN teacher's Q-values,
    via L_KD = || Q_student(s) − Q_teacher(s) ||² (Hinton et al., 2015).
    """

    def __init__(
        self,
        env: gym.Env,
        teacher: nn.Module,
        state_dim: int,
        action_dim: int,
        config: ExperimentConfig,
    ) -> None:
        self.env = env
        self.teacher = teacher
        self.teacher.eval()
        self.config = config
        self.student = StudentNetwork(state_dim, action_dim).to(DEVICE)
        self.loss_history: List[float] = []

    def _collect_dataset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Roll out the teacher greedily and record (state, teacher Q-values) pairs."""
        states_dataset: List[np.ndarray] = []
        q_dataset: List[np.ndarray] = []

        for _ in range(self.config.kd_episodes):
            state, _ = self.env.reset()
            done = False
            while not done:
                state_tensor = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                with torch.no_grad():
                    teacher_q = self.teacher(state_tensor)
                states_dataset.append(state)
                q_dataset.append(teacher_q.squeeze(0).cpu().numpy())
                action = int(teacher_q.argmax(dim=1).item())
                state, _, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

        states_tensor = torch.tensor(np.array(states_dataset), dtype=torch.float32, device=DEVICE)
        q_tensor = torch.tensor(np.array(q_dataset), dtype=torch.float32, device=DEVICE)
        return states_tensor, q_tensor

    def train(self) -> StudentNetwork:
        states_tensor, q_tensor = self._collect_dataset()
        optimizer = optim.Adam(self.student.parameters(), lr=self.config.kd_lr)

        for epoch in range(self.config.kd_epochs):
            student_q = self.student(states_tensor)
            loss = nn.functional.mse_loss(student_q, q_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.loss_history.append(float(loss.item()))
            if epoch % 10 == 0:
                print(f"  KD epoch {epoch:3d} | MSE {loss.item():.6f}")

        return self.student

    def policy_fn(self) -> Callable[[np.ndarray], int]:
        def _policy(state: np.ndarray) -> int:
            state_tensor = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                q_values = self.student(state_tensor)
            return int(q_values.argmax(dim=1).item())

        return _policy
