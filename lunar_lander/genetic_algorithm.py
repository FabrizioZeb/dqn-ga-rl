"""Direct-encoding neuroevolution: genome codec, genetic operators, and the GA loop."""

from __future__ import annotations

from typing import List, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from .config import DEVICE, ExperimentConfig
from .networks import PolicyNetwork


def flatten_weights(model: nn.Module) -> np.ndarray:
    """Concatenate all parameters into a single genotype vector w ∈ ℝ^d."""
    parts = [p.data.cpu().numpy().flatten() for p in model.parameters()]
    return np.concatenate(parts)


def inject_weights(model: nn.Module, flat_weights: np.ndarray) -> None:
    """Decode genotype w into network parameters (phenotype)."""
    pointer = 0
    for param in model.parameters():
        size = param.data.numel()
        slice_w = flat_weights[pointer : pointer + size]
        param.data.copy_(
            torch.tensor(slice_w.reshape(param.data.shape), dtype=torch.float32, device=DEVICE)
        )
        pointer += size


def tournament_selection(
    population: List[np.ndarray],
    fitnesses: List[float],
    tournament_size: int = 5,
) -> np.ndarray:
    """Select parent with highest fitness among k random contestants."""
    indices = np.random.choice(len(population), tournament_size, replace=False)
    best_idx = max(indices, key=lambda i: fitnesses[i])
    return population[best_idx]


def crossover(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
    """Uniform crossover: each gene inherited independently with p=0.5."""
    mask = np.random.rand(len(parent1)) < 0.5
    return np.where(mask, parent1, parent2)


def mutate(genome: np.ndarray, mutation_std: float) -> np.ndarray:
    """Additive Gaussian mutation: w' = w + σ ε, ε ~ N(0,I)."""
    return genome + np.random.randn(len(genome)) * mutation_std


class GeneticAlgorithmTrainer:
    """
    Evolves a population of flat weight vectors for ``PolicyNetwork`` using
    elitism, tournament selection, uniform crossover, and Gaussian mutation.

    Fitness is the Monte-Carlo return of the decoded policy, estimated with
    ``config.ga_eval_episodes`` rollouts per genome.
    """

    def __init__(self, env: gym.Env, state_dim: int, action_dim: int, config: ExperimentConfig) -> None:
        self.env = env
        self.config = config
        self.policy_model = PolicyNetwork(state_dim, action_dim).to(DEVICE)

        genome_size = len(flatten_weights(self.policy_model))
        self.population: List[np.ndarray] = [
            np.random.randn(genome_size) * 0.5 for _ in range(config.ga_population_size)
        ]

        self.best_history: List[float] = []
        self.mean_history: List[float] = []
        self.std_history: List[float] = []
        self.best_genome: np.ndarray | None = None

    def evaluate_genome(self, genome: np.ndarray, episodes: int | None = None) -> float:
        episodes = episodes if episodes is not None else self.config.ga_eval_episodes
        inject_weights(self.policy_model, genome)
        returns = []
        for _ in range(episodes):
            state, _ = self.env.reset()
            done = False
            total = 0.0
            while not done:
                state_tensor = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                with torch.no_grad():
                    logits = self.policy_model(state_tensor)
                action = int(logits.argmax(dim=1).item())
                state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                total += reward
            returns.append(total)
        return float(np.mean(returns))

    def run(self) -> Tuple[np.ndarray, List[float], List[float], List[float]]:
        """Evolve the population for ``config.ga_generations`` and return the best genome."""
        for generation in range(self.config.ga_generations):
            fitnesses = [self.evaluate_genome(g) for g in self.population]
            best_fitness = float(np.max(fitnesses))
            mean_fitness = float(np.mean(fitnesses))
            std_fitness = float(np.std(fitnesses))

            self.best_history.append(best_fitness)
            self.mean_history.append(mean_fitness)
            self.std_history.append(std_fitness)

            elite_indices = np.argsort(fitnesses)[-self.config.ga_n_elite :]
            new_population = [self.population[i].copy() for i in elite_indices]

            while len(new_population) < self.config.ga_population_size:
                p1 = tournament_selection(self.population, fitnesses)
                p2 = tournament_selection(self.population, fitnesses)
                child = mutate(crossover(p1, p2), self.config.ga_mutation_std)
                new_population.append(child)

            self.population = new_population

            print(
                f"Generation {generation:4d} | best {best_fitness:8.2f} | "
                f"mean {mean_fitness:8.2f} | std {std_fitness:6.2f}"
            )

        fitnesses = [self.evaluate_genome(g) for g in self.population]
        self.best_genome = self.population[int(np.argmax(fitnesses))]
        return self.best_genome, self.best_history, self.mean_history, self.std_history

    def policy_fn(self, state: np.ndarray) -> int:
        """Greedy action under the best evolved genome (call after ``run``)."""
        inject_weights(self.policy_model, self.best_genome)
        state_tensor = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            logits = self.policy_model(state_tensor)
        return int(logits.argmax(dim=1).item())
