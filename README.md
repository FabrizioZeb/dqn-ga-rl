# DQN vs. Neuroevolution vs. Knowledge Distillation on LunarLander-v3

A comparative study of three approaches to solving the `LunarLander-v3`
control task (Gymnasium):

- **Deep Q-Network (DQN)** — value-based temporal-difference learning with
  experience replay and a target network.
- **Neuroevolution (Genetic Algorithm)** — direct policy search over a flat
  weight vector, using elitism, tournament selection, uniform crossover, and
  Gaussian mutation.
- **Knowledge distillation** — a compact student network trained offline to
  match the DQN teacher's Q-values.

## Results

| Policy | Mean return (100 eval episodes) | Success rate |
|---|---|---|
| DQN | 233.11 | 78.0% |
| GA (neuroevolution) | -18.38 | 23.0% |
| Distilled student | -240.24 | 0% |

DQN converges to a strong, reliable landing policy. The GA policy is far
less sample-efficient under a sparse, Monte-Carlo fitness signal and shows
much higher variance. The distilled student collapses due to covariate
shift / compounding errors typical of offline imitation from a teacher's
on-policy state distribution.

Training curves, trajectories, and the distillation loss are in
`results/lunar_lander/`.

## Files

- `lunar_lander_experiment.py` — standalone, reproducible script that trains
  DQN and the GA, runs the held-out evaluation, distills the student, and
  regenerates every plot in `results/lunar_lander/`.
- `lunar_lander_dqn_vs_ga.ipynb` — the narrative notebook with the
  theoretical background (MDP formulation, Bellman equations, GA operators)
  alongside the same implementation.
- `results/lunar_lander/` — output plots (training curves, DQN vs. GA
  comparison, trajectories, action distributions, distillation loss).

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate   # or: source .venv/bin/activate
pip install -r requirements.txt
```

## Running

```bash
python lunar_lander_experiment.py
```

Trained weights and evaluation histories (`*.pth`, `*.npy`) are not
tracked in this repo — running the script regenerates them locally under
`results/lunar_lander/`.

## References

- Sutton & Barto (2018). *Reinforcement Learning: An Introduction.* MIT Press.
- Mnih et al. (2015). *Human-level control through deep reinforcement learning.* Nature.
- Stanley & Miikkulainen (2002). *Evolving Neural Networks through Augmenting Topologies.*
- Hinton et al. (2015). *Distilling the Knowledge in a Neural Network.*
