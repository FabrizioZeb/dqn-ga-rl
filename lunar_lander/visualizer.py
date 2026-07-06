"""Publication-oriented figures for training dynamics, evaluation, and trajectories."""

from __future__ import annotations

from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from .config import ACTION_LABELS, STATE_LABELS, ExperimentConfig


class LunarLanderVisualizer:
    """
    Publication-oriented figures for training dynamics, evaluation, and
    physical trajectories on LunarLander-v3.

    All figures are saved under ``config.results_dir`` when ``save_figures`` is True.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.output_dir = config.results_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        try:
            plt.style.use("seaborn-v0_8-whitegrid")
        except OSError:
            plt.style.use("ggplot")

    def finalize(self, fig: plt.Figure, filename: str) -> None:
        fig.tight_layout()
        if self.config.save_figures:
            path = self.output_dir / filename
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"  [figure saved] {path}")
        if self.config.show_figures:
            plt.show()
        else:
            plt.close(fig)

    @staticmethod
    def moving_average(values: np.ndarray, window: int) -> np.ndarray:
        if len(values) < window:
            return np.array(values, dtype=float)
        kernel = np.ones(window) / window
        return np.convolve(values, kernel, mode="valid")

    @staticmethod
    def success_rate(rewards: np.ndarray, threshold: float) -> float:
        """Fraction of episodes with return ≥ threshold (solved proxy)."""
        if len(rewards) == 0:
            return 0.0
        return float(np.mean(rewards >= threshold))

    def plot_dqn_training(
        self,
        reward_history: List[float],
        loss_history: List[float],
        epsilon_history: List[float],
    ) -> None:
        """Episode returns, TD loss, exploration rate ε, and rolling success rate."""
        rewards = np.asarray(reward_history, dtype=float)
        window = self.config.moving_avg_window
        ma = self.moving_average(rewards, window)
        x_ma = np.arange(window - 1, len(rewards))

        # Rolling success rate (fraction of last W episodes above threshold)
        rolling_success = []
        for i in range(len(rewards)):
            start = max(0, i - window + 1)
            rolling_success.append(
                self.success_rate(rewards[start : i + 1], self.config.solved_threshold)
            )

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            "DQN Training on LunarLander-v3\n"
            r"(Bellman TD: $y = r + \gamma \max_{a'} Q_{\theta^-}(s',a')$)",
            fontsize=13,
        )

        ax = axes[0, 0]
        ax.plot(rewards, alpha=0.35, color="steelblue", label="episode return")
        ax.plot(x_ma, ma, color="darkblue", lw=2, label=f"{window}-ep moving avg")
        ax.axhline(
            self.config.solved_threshold,
            color="green",
            ls="--",
            lw=1.5,
            label=f"solved ≥ {self.config.solved_threshold:.0f}",
        )
        ax.set_xlabel("episode")
        ax.set_ylabel("undiscounted return")
        ax.set_title("learning curve")
        ax.legend(loc="lower right", fontsize=8)

        ax = axes[0, 1]
        if loss_history:
            losses = np.asarray(loss_history)
            ax.plot(losses, alpha=0.25, color="coral")
            loss_ma = self.moving_average(losses, min(100, max(10, len(losses) // 20)))
            ax.plot(
                np.arange(len(loss_ma)) + (len(losses) - len(loss_ma)),
                loss_ma,
                color="darkred",
                lw=2,
                label="smoothed MSE",
            )
        ax.set_xlabel("gradient step")
        ax.set_ylabel(r"$\mathcal{L}_{\mathrm{TD}}$ (MSE)")
        ax.set_title("temporal-difference loss")
        ax.legend(fontsize=8)

        ax = axes[1, 0]
        ax.plot(epsilon_history, color="purple", lw=2)
        ax.set_xlabel("episode")
        ax.set_ylabel(r"$\varepsilon$")
        ax.set_title(r"$\varepsilon$-greedy exploration schedule")
        ax.set_ylim(0, 1.05)

        ax = axes[1, 1]
        ax.plot(rolling_success, color="teal", lw=2)
        ax.set_xlabel("episode")
        ax.set_ylabel("success rate")
        ax.set_title(f"rolling {window}-ep success (return ≥ {self.config.solved_threshold:.0f})")
        ax.set_ylim(-0.05, 1.05)

        self.finalize(fig, "01_dqn_training.png")

    def plot_ga_training(
        self,
        best_history: List[float],
        mean_history: List[float],
        std_history: List[float],
    ) -> None:
        """Best / mean / std fitness per generation (population statistics)."""
        gens = np.arange(len(best_history))

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(
            "Neuroevolution (GA) on LunarLander-v3\n"
            r"(fitness $F(w) = \mathbb{E}[\sum_t r_t]$ under policy $\pi_w$)",
            fontsize=13,
        )

        ax = axes[0]
        ax.plot(gens, best_history, label="best", lw=2, color="darkgreen")
        ax.plot(gens, mean_history, label="population mean", lw=2, color="steelblue")
        ax.fill_between(
            gens,
            np.array(mean_history) - np.array(std_history),
            np.array(mean_history) + np.array(std_history),
            alpha=0.25,
            color="steelblue",
            label="±1 std",
        )
        ax.axhline(
            self.config.solved_threshold,
            color="green",
            ls="--",
            lw=1.5,
            label="solved threshold",
        )
        ax.set_xlabel("generation")
        ax.set_ylabel("fitness (episodic return)")
        ax.set_title("population fitness trajectories")
        ax.legend(fontsize=8)

        ax = axes[1]
        improvement = np.diff(best_history, prepend=best_history[0])
        ax.bar(gens, improvement, color=np.where(improvement >= 0, "seagreen", "indianred"), alpha=0.8)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xlabel("generation")
        ax.set_ylabel("Δ best fitness")
        ax.set_title("generation-to-generation improvement")

        self.finalize(fig, "02_ga_training.png")

    def plot_method_comparison(
        self,
        dqn_train_rewards: List[float],
        ga_best_history: List[float],
        dqn_eval: Dict[str, float],
        ga_eval: Dict[str, float],
        dqn_eval_episodes: np.ndarray,
        ga_eval_episodes: np.ndarray,
    ) -> None:
        """Side-by-side comparison of DQN vs neuroevolution."""
        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)
        fig.suptitle("DQN vs Genetic Algorithm — LunarLander-v3", fontsize=14, fontweight="bold")

        # Normalised learning curves (0–1 per method for shape comparison)
        ax1 = fig.add_subplot(gs[0, 0])
        dqn_arr = np.asarray(dqn_train_rewards, dtype=float)
        ga_arr = np.asarray(ga_best_history, dtype=float)

        def normalise(y: np.ndarray) -> np.ndarray:
            lo, hi = y.min(), y.max()
            if hi - lo < 1e-8:
                return np.zeros_like(y)
            return (y - lo) / (hi - lo)

        w = self.config.moving_avg_window
        ax1.plot(
            self.moving_average(normalise(dqn_arr), w),
            label="DQN (episode return, norm.)",
            lw=2,
        )
        ax1.plot(
            self.moving_average(normalise(ga_arr), w),
            label="GA (best fitness, norm.)",
            lw=2,
        )
        ax1.set_xlabel("training index (episodes / generations)")
        ax1.set_ylabel("normalised performance")
        ax1.set_title("learning dynamics (scale-normalised)")
        ax1.legend(fontsize=8)

        # Evaluation boxplots
        ax2 = fig.add_subplot(gs[0, 1])
        bp = ax2.boxplot(
            [dqn_eval_episodes, ga_eval_episodes],
            tick_labels=["DQN", "GA"],
            patch_artist=True,
            widths=0.5,
        )
        colors = ["#4C72B0", "#55A868"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax2.axhline(self.config.solved_threshold, color="green", ls="--", label="solved")
        ax2.set_ylabel("evaluation return")
        ax2.set_title(f"held-out evaluation ({len(dqn_eval_episodes)} episodes each)")
        ax2.legend(fontsize=8)

        # Bar chart of summary statistics
        ax3 = fig.add_subplot(gs[1, 0])
        metrics = ["mean", "std", "best", "worst"]
        x = np.arange(len(metrics))
        width = 0.35
        dqn_vals = [dqn_eval[m] for m in metrics]
        ga_vals = [ga_eval[m] for m in metrics]
        ax3.bar(x - width / 2, dqn_vals, width, label="DQN", color=colors[0], alpha=0.85)
        ax3.bar(x + width / 2, ga_vals, width, label="GA", color=colors[1], alpha=0.85)
        ax3.set_xticks(x)
        ax3.set_xticklabels(metrics)
        ax3.set_ylabel("return")
        ax3.set_title("evaluation statistics")
        ax3.legend()

        # Success rates & training time text panel
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis("off")
        summary = (
            f"{'Metric':<22} {'DQN':>12} {'GA':>12}\n"
            f"{'-' * 48}\n"
            f"{'Mean return':<22} {dqn_eval['mean']:>12.2f} {ga_eval['mean']:>12.2f}\n"
            f"{'Std return':<22} {dqn_eval['std']:>12.2f} {ga_eval['std']:>12.2f}\n"
            f"{'Success rate':<22} "
            f"{self.success_rate(dqn_eval_episodes, self.config.solved_threshold):>11.1%} "
            f"{self.success_rate(ga_eval_episodes, self.config.solved_threshold):>11.1%}\n"
            f"{'Solved threshold':<22} {self.config.solved_threshold:>12.0f}\n"
        )
        ax4.text(
            0.05, 0.95, summary, transform=ax4.transAxes,
            fontsize=11, family="monospace", verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.4),
        )
        ax4.set_title("summary table")

        self.finalize(fig, "03_dqn_vs_ga_comparison.png")

    def plot_trajectory(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        title: str,
        filename: str,
    ) -> None:
        """
        Visualise a single landing episode.

        Panels: (x,y) path with start/land markers, state time series, action sequence,
        cumulative return.
        """
        fig = plt.figure(figsize=(14, 11))
        gs = GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)
        fig.suptitle(f"LunarLander Trajectory — {title}", fontsize=13)

        # 2D path (horizontal position vs altitude)
        ax_xy = fig.add_subplot(gs[0, :])
        x, y = states[:, 0], states[:, 1]
        ax_xy.plot(x, y, "o-", ms=3, lw=1.5, color="navy", alpha=0.8)
        ax_xy.scatter(x[0], y[0], s=120, c="lime", edgecolors="black", zorder=5, label="start")
        ax_xy.scatter(x[-1], y[-1], s=120, c="red", edgecolors="black", zorder=5, label="end")
        # Reference landing pad region (approximate, normalised coords)
        ax_xy.axvspan(-0.2, 0.2, alpha=0.15, color="green", label="pad (approx.)")
        ax_xy.set_xlabel(r"horizontal position $x$")
        ax_xy.set_ylabel(r"altitude $y$")
        ax_xy.set_title("2D trajectory in observation space")
        ax_xy.legend(loc="upper right", fontsize=8)
        ax_xy.grid(True, alpha=0.4)

        # State components over time
        ax_states = fig.add_subplot(gs[1, :])
        timesteps = np.arange(len(states))
        for i in range(min(6, states.shape[1])):  # kinematic states (exclude contacts for clarity)
            ax_states.plot(timesteps, states[:, i], label=STATE_LABELS[i], alpha=0.85)
        ax_states.set_xlabel("timestep")
        ax_states.set_ylabel("state component")
        ax_states.set_title("kinematic state evolution")
        ax_states.legend(ncol=3, fontsize=7, loc="upper right")

        # Actions
        ax_act = fig.add_subplot(gs[2, 0])
        ax_act.step(timesteps, actions, where="post", color="darkorange", lw=2)
        ax_act.set_yticks(range(4))
        ax_act.set_yticklabels(ACTION_LABELS, fontsize=8)
        ax_act.set_xlabel("timestep")
        ax_act.set_title("discrete control sequence")

        # Cumulative reward
        ax_rew = fig.add_subplot(gs[2, 1])
        cum_rew = np.cumsum(rewards)
        ax_rew.plot(timesteps, cum_rew, color="teal", lw=2)
        ax_rew.axhline(self.config.solved_threshold, color="green", ls="--", alpha=0.7)
        ax_rew.set_xlabel("timestep")
        ax_rew.set_ylabel("cumulative return")
        ax_rew.set_title(f"episodic return = {cum_rew[-1]:.1f}")

        self.finalize(fig, filename)

    def plot_action_distribution(
        self,
        dqn_actions: np.ndarray,
        ga_actions: np.ndarray,
    ) -> None:
        """Compare marginal action frequencies between policies."""
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        fig.suptitle("Marginal action distribution during evaluation rollouts", fontsize=12)

        for ax, actions, name in zip(
            axes, [dqn_actions, ga_actions], ["DQN", "GA"]
        ):
            counts = np.bincount(actions, minlength=4)
            probs = counts / counts.sum()
            bars = ax.bar(ACTION_LABELS, probs, color=["#4C72B0", "#DD8452", "#55A868", "#C44E52"], alpha=0.85)
            ax.set_ylabel("probability")
            ax.set_title(name)
            ax.set_ylim(0, max(0.5, probs.max() * 1.2))
            for bar, p in zip(bars, probs):
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{p:.2%}", ha="center", fontsize=9,
                )

        self.finalize(fig, "05_action_distributions.png")

    def plot_distillation_loss(self, kd_loss_history: List[float]) -> None:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(kd_loss_history, color="purple", lw=2)
        ax.set_xlabel("epoch")
        ax.set_ylabel(r"$\mathcal{L}_{\mathrm{KD}} = \|Q_s - Q_t\|^2$")
        ax.set_title("Knowledge distillation loss")
        self.finalize(fig, "06_knowledge_distillation.png")
