"""Orchestrates the four experiment phases: DQN, neuroevolution, comparison, distillation."""

from __future__ import annotations

import time
from typing import List

import gymnasium as gym
import numpy as np
import torch

from .config import DEVICE, ExperimentConfig, set_global_seeds
from .distillation import KnowledgeDistiller
from .dqn_agent import DQNAgent
from .evaluation import collect_action_histogram, collect_episode_trajectory, evaluate_policy
from .genetic_algorithm import GeneticAlgorithmTrainer
from .visualizer import LunarLanderVisualizer


class ExperimentRunner:
    """Runs the full DQN vs. GA vs. distillation comparison on LunarLander-v3."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        set_global_seeds(config.seed)

        self.visualizer = LunarLanderVisualizer(config)
        render_mode = "human" if config.train_render else None
        self.env = gym.make(config.env_name, render_mode=render_mode)
        self.env.reset(seed=config.seed)

        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n

        self.dqn_agent: DQNAgent | None = None
        self.ga_trainer: GeneticAlgorithmTrainer | None = None
        self.distiller: KnowledgeDistiller | None = None

        self.reward_history: List[float] = []
        self.dqn_results: dict = {}
        self.ga_results: dict = {}
        self.dqn_eval_rewards: np.ndarray | None = None
        self.ga_eval_rewards: np.ndarray | None = None
        self.dqn_training_time = 0.0
        self.ga_training_time = 0.0

    def run_dqn_phase(self) -> None:
        print("\n" + "=" * 60)
        print("PHASE 1: DQN training")
        print("=" * 60)

        self.dqn_agent = DQNAgent(self.state_dim, self.action_dim, self.config)
        agent = self.dqn_agent

        loss_history: List[float] = []
        epsilon_history: List[float] = []

        start_time = time.time()
        for episode in range(self.config.dqn_episodes):
            state, _ = self.env.reset()
            done = False
            episode_reward = 0.0

            while not done:
                action = agent.act(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                agent.replay_buffer.add(state, action, reward, next_state, done)
                agent.total_steps += 1

                loss = agent.train_step()
                if loss is not None:
                    loss_history.append(loss)

                state = next_state
                episode_reward += reward

            self.reward_history.append(episode_reward)
            epsilon_history.append(agent.epsilon)
            agent.decay_epsilon()

            if episode % 10 == 0:
                mean10 = np.mean(self.reward_history[-10:])
                print(
                    f"Episode {episode:4d} | return {episode_reward:8.2f} | "
                    f"mean10 {mean10:8.2f} | ε {agent.epsilon:.3f}"
                )

        self.dqn_training_time = time.time() - start_time
        self.visualizer.plot_dqn_training(self.reward_history, loss_history, epsilon_history)

        dqn_policy = lambda s: agent.act(s, training=False)
        self.dqn_results, self.dqn_eval_rewards = evaluate_policy(
            self.env, dqn_policy, self.config.eval_episodes
        )

        print("\nDQN evaluation")
        print("-" * 40)
        for key, value in self.dqn_results.items():
            print(f"  {key}: {value:.2f}")
        print(f"  training_time: {self.dqn_training_time:.1f}s")
        print(f"  environment_steps: {agent.total_steps}")

        dqn_states, dqn_actions, dqn_rewards = collect_episode_trajectory(self.env, dqn_policy)
        self.visualizer.plot_trajectory(
            dqn_states, dqn_actions, dqn_rewards,
            title="DQN (greedy policy)",
            filename="04_trajectory_dqn.png",
        )

    def run_ga_phase(self) -> None:
        print("\n" + "=" * 60)
        print("PHASE 2: Genetic algorithm (neuroevolution)")
        print("=" * 60)

        self.ga_trainer = GeneticAlgorithmTrainer(self.env, self.state_dim, self.action_dim, self.config)

        start_time = time.time()
        _, best_history, mean_history, std_history = self.ga_trainer.run()
        self.ga_training_time = time.time() - start_time

        self.visualizer.plot_ga_training(best_history, mean_history, std_history)

        self.ga_results, self.ga_eval_rewards = evaluate_policy(
            self.env, self.ga_trainer.policy_fn, self.config.eval_episodes
        )

        print("\nGA evaluation (best genome)")
        print("-" * 40)
        for key, value in self.ga_results.items():
            print(f"  {key}: {value:.2f}")
        print(f"  training_time: {self.ga_training_time:.1f}s")

        ga_states, ga_actions, ga_rewards = collect_episode_trajectory(self.env, self.ga_trainer.policy_fn)
        self.visualizer.plot_trajectory(
            ga_states, ga_actions, ga_rewards,
            title="GA (best evolved policy)",
            filename="04_trajectory_ga.png",
        )

    def run_comparison_phase(self) -> None:
        print("\n" + "=" * 60)
        print("PHASE 3: Comparative analysis")
        print("=" * 60)

        self.visualizer.plot_method_comparison(
            dqn_train_rewards=self.reward_history,
            ga_best_history=self.ga_trainer.best_history,
            dqn_eval=self.dqn_results,
            ga_eval=self.ga_results,
            dqn_eval_episodes=self.dqn_eval_rewards,
            ga_eval_episodes=self.ga_eval_rewards,
        )

        dqn_policy = lambda s: self.dqn_agent.act(s, training=False)
        dqn_action_hist = collect_action_histogram(self.env, dqn_policy, episodes=10)
        ga_action_hist = collect_action_histogram(self.env, self.ga_trainer.policy_fn, episodes=10)
        self.visualizer.plot_action_distribution(dqn_action_hist, ga_action_hist)

        print("\nFINAL COMPARISON")
        print("=" * 60)
        print(f"{'':18} {'DQN':>12} {'GA':>12}")
        print(f"{'Mean return':18} {self.dqn_results['mean']:12.2f} {self.ga_results['mean']:12.2f}")
        print(f"{'Std return':18} {self.dqn_results['std']:12.2f} {self.ga_results['std']:12.2f}")
        print(
            f"{'Success rate':18} "
            f"{self.visualizer.success_rate(self.dqn_eval_rewards, self.config.solved_threshold):11.1%} "
            f"{self.visualizer.success_rate(self.ga_eval_rewards, self.config.solved_threshold):11.1%}"
        )
        print(f"{'DQN train time':18} {self.dqn_training_time:12.1f}s")
        print(f"{'GA train time':18} {self.ga_training_time:12.1f}s")

    def run_distillation_phase(self) -> None:
        print("\n" + "=" * 60)
        print("PHASE 4: Knowledge distillation (DQN → student)")
        print("=" * 60)

        self.distiller = KnowledgeDistiller(
            self.env, self.dqn_agent.online_net, self.state_dim, self.action_dim, self.config
        )
        self.distiller.train()
        self.visualizer.plot_distillation_loss(self.distiller.loss_history)

        student_results, _ = evaluate_policy(self.env, self.distiller.policy_fn(), self.config.eval_episodes)
        print("\nStudent evaluation")
        for key, value in student_results.items():
            print(f"  {key}: {value:.2f}")

    def save_artifacts(self) -> None:
        torch.save(self.dqn_agent.online_net.state_dict(), self.config.results_dir / "dqn_model.pth")
        torch.save(self.distiller.student.state_dict(), self.config.results_dir / "student_model.pth")
        np.save(self.config.results_dir / "best_genome.npy", self.ga_trainer.best_genome)
        np.save(self.config.results_dir / "dqn_reward_history.npy", np.array(self.reward_history))
        np.save(self.config.results_dir / "ga_best_history.npy", np.array(self.ga_trainer.best_history))
        print(f"\nModels and metrics saved to: {self.config.results_dir.resolve()}")

    def run(self) -> None:
        print("=" * 60)
        print("LunarLander-v3: DQN vs Neuroevolution")
        print("=" * 60)
        print(f"Device: {DEVICE}")
        print(f"Results directory: {self.config.results_dir.resolve()}")
        print(f"State dimension: {self.state_dim}  |  Action dimension: {self.action_dim}")

        self.run_dqn_phase()
        self.run_ga_phase()
        self.run_comparison_phase()
        self.run_distillation_phase()
        self.save_artifacts()

        print("\nExperiment complete.")
        self.env.close()
