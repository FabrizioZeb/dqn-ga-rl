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

## Table of contents

- [Problem setup](#problem-setup)
- [Theory](#theory)
  - [1. The MDP and the control problem](#1-the-mdp-and-the-control-problem)
  - [2. Deep Q-Network (value-based RL)](#2-deep-q-network-value-based-rl)
  - [3. Neuroevolution (direct policy search)](#3-neuroevolution-direct-policy-search)
  - [4. Knowledge distillation](#4-knowledge-distillation)
  - [5. Why the three methods diverge](#5-why-the-three-methods-diverge)
- [Architecture](#architecture)
  - [Module/package diagram](#modulepackage-diagram)
  - [Class diagram (UML)](#class-diagram-uml)
  - [Experiment pipeline (flowchart)](#experiment-pipeline-flowchart)
  - [DQN training step (sequence diagram)](#dqn-training-step-sequence-diagram)
  - [GA generation (sequence diagram)](#ga-generation-sequence-diagram)
  - [Genome encode/decode (state diagram)](#genome-encodedecode-state-diagram)
  - [Knowledge distillation (sequence diagram)](#knowledge-distillation-sequence-diagram)
- [Results](#results)
- [Files](#files)
- [Setup](#setup)
- [Running](#running)
- [References](#references)

## Problem setup

`LunarLander-v3` is an 8-dimensional continuous-state, 4-action discrete
control task: land a rigid body between two flags using a main engine and
two lateral thrusters, under gravity and fuel cost, without crashing or
drifting out of bounds.

| Symbol | Meaning |
|---|---|
| $s = (x, y, v_x, v_y, \theta, \omega, \text{leg}_1, \text{leg}_2)$ | position, velocity, angle, angular velocity, leg-contact flags |
| $a \in \{\text{noop}, \text{left engine}, \text{main engine}, \text{right engine}\}$ | discrete action |
| $r(s,a,s')$ | shaped reward: distance/velocity/angle penalties, leg-contact bonus, fuel cost, ±100 for crash/landing |
| solved threshold | mean return ≥ 200 over 100 episodes |

All three methods are compared on the **same environment, same reward
function, same evaluation protocol** (Monte-Carlo return over held-out
episodes) — only the *learning algorithm* and *policy representation*
change.

## Theory

### 1. The MDP and the control problem

The task is a Markov Decision Process $(\mathcal{S}, \mathcal{A}, P, r, \gamma)$.
A policy $\pi$ selects actions to maximize the expected discounted return:

$$
J(\pi) = \mathbb{E}_{\tau \sim \pi}\left[\sum_{t=0}^{T} \gamma^t r_t\right]
$$

There are two broad families of solution methods, both implemented here:

- **Value-based** (DQN): learn $Q^\pi(s,a)$, the expected return of taking
  $a$ in $s$ and then following $\pi$; derive a policy via
  $\pi(s) = \arg\max_a Q(s,a)$.
- **Direct policy search** (GA / neuroevolution): parameterize a policy
  $\pi_\theta$ directly and search $\theta$-space using a black-box,
  gradient-free optimizer, guided only by episodic return.

### 2. Deep Q-Network (value-based RL)

DQN (Mnih et al., 2015) approximates the optimal action-value function
$Q^*(s,a)$ with a neural network $Q_\theta$, trained by minimizing the
**Bellman residual** on transitions sampled from a replay buffer:

$$
\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}}
\Big[\big(Q_\theta(s,a) - y\big)^2\Big], \qquad
y = r + \gamma (1 - \text{done}) \max_{a'} Q_{\theta^-}(s', a')
$$

Two stabilization tricks are used, both visible in `dqn_agent.py`:

- **Experience replay** (`replay_buffer.py`): transitions are stored in a
  buffer and sampled uniformly at random, breaking the temporal
  correlation between consecutive $(s,a,r,s')$ tuples that would otherwise
  make SGD updates highly correlated and unstable.
- **Target network** $\theta^-$: a periodically-synced copy of $\theta$
  used only to compute the bootstrap target $y$. Without it, $\theta$
  chases a target that moves every step (its own prediction), which is a
  classic source of divergence in bootstrapped, function-approximated TD
  learning ("the deadly triad": bootstrapping + function approximation +
  off-policy data).

Exploration uses **ε-greedy**: with probability $\varepsilon$ take a random
action, otherwise $\arg\max_a Q_\theta(s,a)$; $\varepsilon$ decays
geometrically from `dqn_epsilon_start` to `dqn_epsilon_min`
(`config.py`), trading exploration for exploitation as $Q_\theta$ improves.

### 3. Neuroevolution (direct policy search)

The GA (`genetic_algorithm.py`) treats the policy network's entire weight
vector as a **genotype** $w \in \mathbb{R}^d$ (flattened via
`flatten_weights`/`inject_weights` — a direct encoding, following the
spirit of Stanley & Miikkulainen's NEAT lineage, minus topology evolution).
There is no gradient anywhere in this loop; fitness is the **Monte-Carlo
episodic return** of the decoded policy, a black-box, non-differentiable
signal.

One generation applies:

1. **Fitness evaluation** — roll out every genome's decoded policy for
   `ga_eval_episodes` episode(s); fitness = mean return.
2. **Elitism** — the top `ga_n_elite` genomes survive unchanged, guaranteeing
   monotonic improvement of the population's best fitness.
3. **Tournament selection** — sample `k=5` genomes uniformly, keep the
   fittest as a parent; repeated to fill the mating pool.
4. **Uniform crossover** — child gene $i$ is copied from parent 1 or parent
   2 with probability 0.5 each, independently per gene.
5. **Gaussian mutation** — $w' = w + \sigma \epsilon,\ \epsilon \sim
   \mathcal{N}(0, I)$, injecting variance the crossover step alone cannot
   produce.

This is a genuinely different search paradigm from DQN: it never computes
$\partial \mathcal{L}/\partial \theta$, needs no differentiable reward,
and evolves an entire *population* of candidate policies rather than one.
Its cost is sample efficiency — Monte-Carlo fitness is a very high-variance,
low-information signal per environment interaction compared to per-step TD
updates.

### 4. Knowledge distillation

`distillation.py` implements offline distillation (Hinton et al., 2015):
a small `StudentNetwork` is trained to regress the *teacher's Q-values*
directly, rather than being trained by interacting with the environment:

$$
\mathcal{L}_{\text{KD}} = \mathbb{E}_{s \sim \mathcal{D}_{\text{teacher}}}
\big[\, \lVert Q_{\text{student}}(s) - Q_{\text{teacher}}(s) \rVert_2^2 \,\big]
$$

The dataset $\mathcal{D}_{\text{teacher}}$ is collected by rolling the
**frozen, greedy teacher policy** for `kd_episodes` episodes and recording
its full Q-value vector at every visited state (`_collect_dataset`). The
student then does pure supervised regression (no environment interaction)
for `kd_epochs`.

The critical caveat, and the reason distillation collapses in the results
below: the student is only ever supervised on states the *teacher* visits.
Once deployed, the student's own (imperfect) actions push it into states
slightly outside that support — the teacher never demonstrated the correct
Q-values there, small errors compound step after step, and the trajectory
diverges from anything in the training distribution. This is the classic
**covariate shift / compounding error** problem of offline behavioral
cloning (closely related to why plain behavioral cloning underperforms
interactive imitation learning like DAgger).

### 5. Why the three methods diverge

| | DQN | GA (neuroevolution) | Distilled student |
|---|---|---|---|
| Learning signal | per-step TD error (dense) | episodic return (sparse, Monte-Carlo) | supervised regression to teacher Q-values (dense, but offline) |
| Uses gradients? | yes (backprop through $Q_\theta$) | no (population search) | yes (backprop through student) |
| Interacts with env during training? | yes, on-policy-ish via replay | yes, every genome every generation | no — trained purely offline on teacher rollouts |
| Main failure mode | can diverge without replay/target net | high variance, slow convergence under sparse fitness | covariate shift at deployment time |

## Architecture

### Module/package diagram

```mermaid
%%{init: {'flowchart': {'curve': 'linear'}}}%%
flowchart LR
    MAIN["main.py"] --> EXP["experiment.py<br/>ExperimentRunner"]
    EXP --> DQN["dqn_agent.py"]
    EXP --> GA["genetic_algorithm.py"]
    EXP --> KD["distillation.py"]
    EXP --> VIZ["visualizer.py"]
    DQN --> NET["networks.py"]
    GA --> NET
    KD --> NET
    DQN --> BUFFER["replay_buffer.py"]
    DQN --> GYM["Gymnasium env"]
    GA --> GYM
    KD --> GYM
    ALL["config.py"] -.-> DQN
    ALL -.-> GA
    ALL -.-> KD
```

`config.py` supplies shared hyperparameters/seeding to the DQN, GA, and
distillation modules (dashed edges); `evaluation.py` is used by all three
for Monte-Carlo rollouts and is omitted here for clarity.

### Class diagram (UML)

```mermaid
classDiagram
    direction TB

    class ExperimentConfig {
        +int seed
        +int dqn_episodes
        +int ga_generations
        +int kd_epochs
        +Path results_dir
    }

    class QNetwork {
        +forward(x) Tensor
    }
    class PolicyNetwork {
        +forward(x) Tensor
    }
    class StudentNetwork {
        +forward(x) Tensor
    }

    class ReplayBuffer {
        +add(s, a, r, s2, done)
        +sample(batch_size) Tuple
    }

    class DQNAgent {
        +QNetwork online_net
        +QNetwork target_net
        +ReplayBuffer replay_buffer
        +act(state, training) int
        +train_step() float
        +decay_epsilon()
    }

    class GeneticAlgorithmTrainer {
        +PolicyNetwork policy_model
        +List population
        +ndarray best_genome
        +evaluate_genome(genome) float
        +run() Tuple
        +policy_fn(state) int
    }

    class KnowledgeDistiller {
        +Module teacher
        +StudentNetwork student
        +train() StudentNetwork
        +policy_fn() Callable
    }

    class ExperimentRunner {
        +DQNAgent dqn_agent
        +GeneticAlgorithmTrainer ga_trainer
        +KnowledgeDistiller distiller
        +LunarLanderVisualizer visualizer
        +run()
    }

    class LunarLanderVisualizer {
        +plot_dqn_training(...)
        +plot_ga_training(...)
        +plot_method_comparison(...)
    }

    ExperimentRunner --> ExperimentConfig
    ExperimentRunner --> DQNAgent
    ExperimentRunner --> GeneticAlgorithmTrainer
    ExperimentRunner --> KnowledgeDistiller
    ExperimentRunner --> LunarLanderVisualizer

    DQNAgent --> QNetwork
    DQNAgent --> ReplayBuffer
    GeneticAlgorithmTrainer --> PolicyNetwork
    KnowledgeDistiller --> StudentNetwork
    KnowledgeDistiller --> DQNAgent
```

`ExperimentRunner` owns one instance of each component and wires the DQN's
trained `online_net` in as the `teacher` for `KnowledgeDistiller`. All
four trainer/agent classes also read from `ExperimentConfig` (omitted
above to keep the diagram readable).

### Experiment pipeline (flowchart)

```mermaid
%%{init: {'flowchart': {'curve': 'linear'}}}%%
flowchart TB
    INIT["Setup: seed RNGs, create env"] --> P1["Phase 1: train DQN"]
    P1 --> P2["Phase 2: evolve GA population"]
    P2 --> P3["Phase 3: compare DQN vs. GA"]
    P3 --> P4["Phase 4: distill DQN into student"]
    P4 --> SAVE["Save models, metrics, plots"]
```

Each phase also evaluates and plots its own policy (mean return, success
rate, trajectory) immediately after training, before moving to the next
phase.

### DQN training step (sequence diagram)

```mermaid
sequenceDiagram
    participant Exp as ExperimentRunner
    participant Env as Gymnasium Env
    participant Agent as DQNAgent
    participant Buf as ReplayBuffer
    participant QOn as online_net (θ)
    participant QTgt as target_net (θ⁻)

    loop each episode
        Exp->>Env: reset()
        loop each step until done
            Exp->>Agent: act(state)
            alt random() < ε
                Agent-->>Exp: random action
            else
                Agent->>QOn: forward(state)
                QOn-->>Agent: Q(s, ·)
                Agent-->>Exp: argmax_a Q(s,a)
            end
            Exp->>Env: step(action)
            Env-->>Exp: next_state, reward, done
            Exp->>Buf: add(s, a, r, s', done)
            Exp->>Agent: train_step()
            Buf-->>Agent: sample(batch_size)
            Agent->>QOn: Q(s,a) for taken actions
            Agent->>QTgt: max_a' Q(s', a')  (no grad)
            Agent->>Agent: loss = MSE(Q(s,a), r + γ·max_a'Q⁻·(1-done))
            Agent->>QOn: backward() + optimizer.step()
            alt total_steps % target_sync == 0
                Agent->>QTgt: load_state_dict(online_net)
            end
        end
        Agent->>Agent: decay_epsilon()
    end
```

### GA generation (sequence diagram)

```mermaid
sequenceDiagram
    participant GA as GeneticAlgorithmTrainer
    participant Pop as population (List[genome])
    participant Env as Gymnasium Env
    participant Pol as PolicyNetwork (decoded)

    loop each generation
        loop each genome in population
            GA->>Pol: inject_weights(genome)
            loop ga_eval_episodes
                GA->>Env: rollout with argmax(Pol(state))
                Env-->>GA: episode return
            end
            GA->>GA: fitness[genome] = mean(returns)
        end
        GA->>GA: elite = top-k by fitness (survive unchanged)
        loop until population refilled
            GA->>Pop: tournament_selection() → parent1
            GA->>Pop: tournament_selection() → parent2
            GA->>GA: child = mutate(crossover(parent1, parent2))
            GA->>Pop: append(child)
        end
        GA->>GA: population = elite + children
    end
    GA->>GA: best_genome = argmax fitness (final population)
```

### Genome encode/decode (state diagram)

```mermaid
flowchart LR
    subgraph Genotype["Genotype (search space)"]
        W["flat vector w ∈ ℝ^d<br/>(numpy array)"]
    end
    subgraph Phenotype["Phenotype (behavior space)"]
        NET["PolicyNetwork parameters<br/>(torch tensors)"]
        ACT["greedy action = argmax(logits)"]
    end

    W -- "inject_weights()<br/>reshape + copy_ per layer" --> NET
    NET -- "forward(state)" --> ACT
    NET -- "flatten_weights()<br/>concatenate .flatten() per layer" --> W
```

### Knowledge distillation (sequence diagram)

```mermaid
sequenceDiagram
    participant KD as KnowledgeDistiller
    participant Env as Gymnasium Env
    participant T as teacher (frozen DQN online_net)
    participant S as StudentNetwork

    Note over KD,T: Phase A — dataset collection (offline, no gradients)
    loop kd_episodes
        KD->>Env: reset()
        loop until done
            KD->>T: Q_teacher(state)  [no_grad]
            T-->>KD: Q-values
            KD->>KD: record (state, Q_teacher(state))
            KD->>Env: step(argmax Q_teacher)
        end
    end

    Note over KD,S: Phase B — supervised regression (no env interaction)
    loop kd_epochs
        KD->>S: forward(all states)
        S-->>KD: Q_student
        KD->>KD: loss = MSE(Q_student, Q_teacher)
        KD->>S: backward() + optimizer.step()
    end

    Note over S: Deployment: student acts on its own trajectory —<br/>states outside teacher's visited set are unsupervised (covariate shift)
```

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

- `main.py` — entry point; runs the full experiment end to end.
- `lunar_lander/` — package with the implementation: `config.py`
  (hyperparameters), `networks.py` (Q/policy/student networks),
  `replay_buffer.py`, `dqn_agent.py`, `genetic_algorithm.py` (genome codec,
  genetic operators, GA trainer), `distillation.py`, `evaluation.py`
  (rollout/eval helpers), `visualizer.py`, and `experiment.py` (orchestrates
  the four phases).
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
python main.py
```

Trained weights and evaluation histories (`*.pth`, `*.npy`) are not
tracked in this repo — running the script regenerates them locally under
`results/lunar_lander/`.

## References

- Sutton & Barto (2018). *Reinforcement Learning: An Introduction.* MIT Press.
- Mnih et al. (2015). *Human-level control through deep reinforcement learning.* Nature.
- Stanley & Miikkulainen (2002). *Evolving Neural Networks through Augmenting Topologies.*
- Hinton et al. (2015). *Distilling the Knowledge in a Neural Network.*
