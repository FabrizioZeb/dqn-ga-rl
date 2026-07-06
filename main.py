"""
Entry point: DQN vs. Neuroevolution vs. Knowledge Distillation on LunarLander-v3.

See README.md for the theoretical background (MDP formulation, Bellman
equations, GA operators, distillation objective) and result summary.
"""

from lunar_lander.config import ExperimentConfig
from lunar_lander.experiment import ExperimentRunner

if __name__ == "__main__":
    runner = ExperimentRunner(ExperimentConfig())
    runner.run()
