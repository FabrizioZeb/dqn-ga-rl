"""Rollout collection and Monte-Carlo policy evaluation, shared by every method."""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import gymnasium as gym
import numpy as np


def collect_episode_trajectory(
    env: gym.Env,
    policy_fn: Callable[[np.ndarray], int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Record (states, actions, rewards) for one episode under ``policy_fn``."""
    states, actions, rewards = [], [], []
    state, _ = env.reset()
    done = False

    while not done:
        action = policy_fn(state)
        states.append(np.array(state, dtype=np.float32))
        actions.append(action)

        state, reward, terminated, truncated, _ = env.step(action)
        rewards.append(reward)
        done = terminated or truncated

    return (
        np.stack(states),
        np.array(actions, dtype=int),
        np.array(rewards, dtype=np.float32),
    )


def collect_action_histogram(
    env: gym.Env,
    policy_fn: Callable[[np.ndarray], int],
    episodes: int,
) -> np.ndarray:
    """Concatenate actions from multiple evaluation episodes."""
    all_actions = []
    for _ in range(episodes):
        _, actions, _ = collect_episode_trajectory(env, policy_fn)
        all_actions.append(actions)
    return np.concatenate(all_actions)


def evaluate_policy(
    env: gym.Env,
    policy_fn: Callable[[np.ndarray], int],
    episodes: int = 20,
) -> Tuple[Dict[str, float], np.ndarray]:
    """Monte-Carlo evaluation: mean, std, best, worst over ``episodes`` rollouts."""
    rewards = []
    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        total = 0.0
        while not done:
            action = policy_fn(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total += reward
        rewards.append(total)

    rewards_arr = np.array(rewards, dtype=float)
    return {
        "mean": float(rewards_arr.mean()),
        "std": float(rewards_arr.std()),
        "best": float(rewards_arr.max()),
        "worst": float(rewards_arr.min()),
    }, rewards_arr
