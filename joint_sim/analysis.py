"""
Post-hoc analysis and visualization of simulation results.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import rel_entr

from config import Config, StateSpace
from organism import compute_uniform_joint


class Analyzer:
    """Analyze simulation results and generate plots."""

    def __init__(self, config: Config):
        self.config = config
        self.base_dir = config.output.base_dir
        self.plots_dir = self.base_dir / config.output.plots_dir
        self.plots_dir.mkdir(exist_ok=True)

        # Load logs
        self.logs = pd.read_csv(self.base_dir / config.output.logs_file)

        # Reference uniform joint
        self.uniform_joint = compute_uniform_joint()

    def run_all_analyses(self) -> None:
        """Run all analysis plots."""
        print("Running analysis...")

        self.plot_population_over_time()
        self.plot_kl_divergence_over_time()
        self.plot_h_marginal_comparison()
        self.plot_conditional_slices()
        self.plot_diversity_histogram()

        print(f"Plots saved to {self.plots_dir}")

    def plot_population_over_time(self) -> None:
        """Plot population size over time."""
        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(self.logs["tick"], self.logs["population"], linewidth=0.5)
        ax.set_xlabel("Tick")
        ax.set_ylabel("Population")
        ax.set_title("Population Over Time")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.plots_dir / "population_over_time.png", dpi=150)
        plt.close()

    def plot_kl_divergence_over_time(self) -> None:
        """Plot KL divergence from uniform over time."""
        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(self.logs["tick"], self.logs["kl_divergence"], linewidth=0.5)
        ax.set_xlabel("Tick")
        ax.set_ylabel("KL Divergence")
        ax.set_title("KL Divergence (Population Joint || Uniform) Over Time")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.plots_dir / "kl_divergence_over_time.png", dpi=150)
        plt.close()

    def plot_h_marginal_comparison(self) -> None:
        """Compare marginal P(H) at start vs end."""
        snapshots_dir = self.base_dir / "joint_snapshots"
        snapshot_files = sorted(snapshots_dir.glob("joint_tick_*.npy"))

        if len(snapshot_files) < 2:
            print("Not enough snapshots for H marginal comparison")
            return

        # Load first and last
        start_joint = np.load(snapshot_files[0])
        end_joint = np.load(snapshot_files[-1])

        # Compute marginals: P(H) = sum over X, O
        start_h_marginal = start_joint.sum(axis=(0, 2))  # Sum over X and O
        end_h_marginal = end_joint.sum(axis=(0, 2))

        # Normalize
        start_h_marginal = start_h_marginal / start_h_marginal.sum()
        end_h_marginal = end_h_marginal / end_h_marginal.sum()

        # Plot
        fig, ax = plt.subplots(figsize=(10, 5))

        x = np.arange(StateSpace.NUM_H)
        width = 0.35

        ax.bar(x - width / 2, start_h_marginal, width, label="Start", alpha=0.7)
        ax.bar(x + width / 2, end_h_marginal, width, label="End", alpha=0.7)

        ax.set_xlabel("Energy Level (H)")
        ax.set_ylabel("Probability")
        ax.set_title("Marginal P(H) at Start vs End")
        ax.set_xticks(x)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.plots_dir / "h_marginal_comparison.png", dpi=150)
        plt.close()

    def plot_conditional_slices(self) -> None:
        """Plot specific P(O|X,H) slices at start vs end."""
        snapshots_dir = self.base_dir / "joint_snapshots"
        snapshot_files = sorted(snapshots_dir.glob("joint_tick_*.npy"))

        if len(snapshot_files) < 2:
            print("Not enough snapshots for conditional slice analysis")
            return

        start_joint = np.load(snapshot_files[0])
        end_joint = np.load(snapshot_files[-1])

        # Define interesting slices
        slices_to_plot = [
            {
                "name": "On Food (H=3)",
                "description": "X=(empty, food, empty), H=3\nHypothesis: P(STAY) rises",
                "x_idx": StateSpace.encode_x(StateSpace.EMPTY, StateSpace.FOOD, StateSpace.EMPTY),
                "h_idx": 3
            },
            {
                "name": "Food to Left (H=5)",
                "description": "X=(food, empty, empty), H=5\nHypothesis: P(MOVE_LEFT) rises",
                "x_idx": StateSpace.encode_x(StateSpace.FOOD, StateSpace.EMPTY, StateSpace.EMPTY),
                "h_idx": 5
            },
            {
                "name": "Surrounded by Occupied (H=8)",
                "description": "X=(occupied, empty, occupied), H=8\nHypothesis: P(STAY) rises",
                "x_idx": StateSpace.encode_x(StateSpace.OCCUPIED, StateSpace.EMPTY, StateSpace.OCCUPIED),
                "h_idx": 8
            },
            {
                "name": "Low Energy, Food Right (H=2)",
                "description": "X=(empty, empty, food), H=2\nHypothesis: P(MOVE_RIGHT) rises",
                "x_idx": StateSpace.encode_x(StateSpace.EMPTY, StateSpace.EMPTY, StateSpace.FOOD),
                "h_idx": 2
            }
        ]

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        action_names = ["MOVE_LEFT", "STAY", "MOVE_RIGHT"]

        for ax, slice_info in zip(axes, slices_to_plot):
            x_idx = slice_info["x_idx"]
            h_idx = slice_info["h_idx"]

            # Get P(O|X,H) by normalizing the slice
            start_slice = start_joint[x_idx, h_idx, :]
            end_slice = end_joint[x_idx, h_idx, :]

            start_probs = start_slice / start_slice.sum() if start_slice.sum() > 0 else start_slice
            end_probs = end_slice / end_slice.sum() if end_slice.sum() > 0 else end_slice

            x = np.arange(3)
            width = 0.35

            ax.bar(x - width / 2, start_probs, width, label="Start", alpha=0.7)
            ax.bar(x + width / 2, end_probs, width, label="End", alpha=0.7)

            ax.set_xlabel("Action")
            ax.set_ylabel("P(O|X,H)")
            ax.set_title(slice_info["name"])
            ax.set_xticks(x)
            ax.set_xticklabels(action_names)
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Add description
            ax.text(
                0.02, 0.98, slice_info["description"],
                transform=ax.transAxes,
                fontsize=8,
                verticalalignment="top",
                fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
            )

        plt.tight_layout()
        plt.savefig(self.plots_dir / "conditional_slices.png", dpi=150)
        plt.close()

    def plot_diversity_histogram(self) -> None:
        """Plot histogram of pairwise KL divergence among survivors."""
        final_joints_dir = self.base_dir / "final_joints"
        all_joints_path = final_joints_dir / "all_joints.npy"

        if not all_joints_path.exists():
            print("Final joints not found for diversity analysis")
            return

        all_joints = np.load(all_joints_path)

        if len(all_joints) < 2:
            print("Need at least 2 organisms for diversity analysis")
            return

        # Compute pairwise KL divergences
        n = len(all_joints)
        kl_values = []

        for i in range(n):
            for j in range(i + 1, n):
                p = all_joints[i].flatten()
                q = all_joints[j].flatten()

                # Add epsilon and normalize
                eps = 1e-12
                p = (p + eps) / (p + eps).sum()
                q = (q + eps) / (q + eps).sum()

                # Symmetric KL (average of both directions)
                kl = (np.sum(rel_entr(p, q)) + np.sum(rel_entr(q, p))) / 2
                kl_values.append(kl)

        # Plot histogram
        fig, ax = plt.subplots(figsize=(10, 5))

        ax.hist(kl_values, bins=50, edgecolor="black", alpha=0.7)
        ax.axvline(np.mean(kl_values), color="red", linestyle="--", label=f"Mean: {np.mean(kl_values):.4f}")
        ax.axvline(np.median(kl_values), color="orange", linestyle="--", label=f"Median: {np.median(kl_values):.4f}")

        ax.set_xlabel("Symmetric KL Divergence")
        ax.set_ylabel("Count")
        ax.set_title(f"Pairwise Joint Diversity Among {n} Survivors")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.plots_dir / "diversity_histogram.png", dpi=150)
        plt.close()

    def print_summary_statistics(self) -> None:
        """Print summary statistics to console."""
        print("\n" + "=" * 60)
        print("SIMULATION SUMMARY")
        print("=" * 60)

        print(f"\nTotal ticks: {self.logs['tick'].max()}")
        print(f"Final population: {self.logs['population'].iloc[-1]}")
        print(f"Peak population: {self.logs['population'].max()}")
        print(f"Min population: {self.logs['population'].min()}")

        print(f"\nFinal KL divergence: {self.logs['kl_divergence'].iloc[-1]:.6f}")
        print(f"Max KL divergence: {self.logs['kl_divergence'].max():.6f}")

        print(f"\nTotal births: {self.logs['births_interval'].sum()}")
        print(f"Total starvation deaths: {self.logs['deaths_starvation_interval'].sum()}")
        print(f"Total collision deaths: {self.logs['deaths_collision_interval'].sum()}")

        print(f"\nFinal mean H: {self.logs['mean_h'].iloc[-1]:.2f}")
        print(f"Final median H: {self.logs['median_h'].iloc[-1]:.1f}")

        print("=" * 60 + "\n")


def run_analysis(config: Config) -> None:
    """Run full post-hoc analysis."""
    analyzer = Analyzer(config)
    analyzer.print_summary_statistics()
    analyzer.run_all_analyses()


if __name__ == "__main__":
    from config import default_config
    run_analysis(default_config)
