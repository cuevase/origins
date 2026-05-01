"""
Logging and serialization for simulation data.
"""
import csv
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.special import rel_entr

from config import Config, StateSpace
from organism import compute_uniform_joint

if TYPE_CHECKING:
    from simulation import Simulation, TickStats


class Logger:
    """
    Handles logging of simulation state and statistics.

    Logs scalars to CSV and joint distributions to .npy files.
    """

    def __init__(self, config: Config):
        self.config = config
        self.base_dir = config.output.base_dir
        self.logs_path = self.base_dir / config.output.logs_file
        self.snapshots_dir = self.base_dir / config.output.joint_snapshots_dir

        # Reference uniform joint for KL divergence
        self.uniform_joint = compute_uniform_joint()

        # Interval tracking
        self.interval_births = 0
        self.interval_deaths_starvation = 0
        self.interval_deaths_collision = 0

        # Initialize CSV
        self._init_csv()

    def _init_csv(self) -> None:
        """Initialize the CSV log file with headers."""
        with open(self.logs_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "tick",
                "population",
                "mean_h",
                "median_h",
                "births_interval",
                "deaths_starvation_interval",
                "deaths_collision_interval",
                "food_count",
                "kl_divergence"
            ])

    def accumulate_tick_stats(self, stats: "TickStats") -> None:
        """Accumulate stats within a logging interval."""
        self.interval_births += stats.births
        self.interval_deaths_starvation += stats.deaths_starvation
        self.interval_deaths_collision += stats.deaths_collision

    def log(self, sim: "Simulation") -> None:
        """
        Log current state to CSV and save joint snapshot.

        Should be called every log_interval ticks.
        """
        tick = sim.tick
        population = sim.get_population_size()
        mean_h = sim.get_mean_h()
        median_h = sim.get_median_h()
        food_count = sim.world.count_food()

        # Compute population-averaged joint and KL divergence
        pop_joint = sim.get_population_averaged_joint()
        kl_div = self._compute_kl_divergence(pop_joint)

        # Write to CSV
        with open(self.logs_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                tick,
                population,
                f"{mean_h:.4f}",
                f"{median_h:.1f}",
                self.interval_births,
                self.interval_deaths_starvation,
                self.interval_deaths_collision,
                food_count,
                f"{kl_div:.6f}"
            ])

        # Save joint snapshot
        snapshot_path = self.snapshots_dir / f"joint_tick_{tick:07d}.npy"
        np.save(snapshot_path, pop_joint)

        # Reset interval counters
        self.interval_births = 0
        self.interval_deaths_starvation = 0
        self.interval_deaths_collision = 0

    def _compute_kl_divergence(self, pop_joint: np.ndarray) -> float:
        """
        Compute KL divergence from population joint to uniform joint.

        KL(pop || uniform) = sum(pop * log(pop / uniform))
        """
        # Flatten for computation
        p = pop_joint.flatten()
        q = self.uniform_joint.flatten()

        # Add small epsilon to avoid log(0)
        eps = 1e-12
        p = p + eps
        q = q + eps

        # Renormalize
        p = p / p.sum()
        q = q / q.sum()

        # Compute KL divergence
        kl = np.sum(rel_entr(p, q))
        return float(kl)

    def save_final_state(self, sim: "Simulation") -> None:
        """Save final population joints and lineage samples."""
        living = sim.get_living_organisms()

        if not living:
            print("No living organisms to save.")
            return

        # Save all final joints
        if self.config.output.save_final_joints:
            final_dir = self.base_dir / "final_joints"
            final_dir.mkdir(exist_ok=True)

            all_joints = np.stack([o.joint for o in living])
            np.save(final_dir / "all_joints.npy", all_joints)

            # Also save individual joints with IDs
            for org in living:
                np.save(final_dir / f"organism_{org.id}.npy", org.joint)

        # Save lineage sample
        if self.config.output.save_lineage_sample > 0:
            lineages_dir = self.base_dir / "lineages"
            lineages_dir.mkdir(exist_ok=True)

            # Sample organisms
            sample_size = min(self.config.output.save_lineage_sample, len(living))
            sample = np.random.choice(living, size=sample_size, replace=False)

            for org in sample:
                lineage = self._trace_lineage(org, sim.organisms)
                lineage_data = {
                    "organism_id": org.id,
                    "joints": [o.joint for o in lineage],
                    "ids": [o.id for o in lineage]
                }
                np.savez(
                    lineages_dir / f"lineage_{org.id}.npz",
                    **{f"joint_{i}": j for i, j in enumerate(lineage_data["joints"])},
                    ids=lineage_data["ids"]
                )

    def _trace_lineage(self, org, all_organisms: dict) -> list:
        """Trace an organism's lineage back through parents."""
        lineage = [org]
        current = org

        while current.parent_id is not None and current.parent_id in all_organisms:
            parent = all_organisms[current.parent_id]
            lineage.append(parent)
            current = parent

        return lineage


def print_progress(sim: "Simulation", stats: "TickStats") -> None:
    """Print a simple progress update."""
    pop = sim.get_population_size()
    food = sim.world.count_food()
    mean_h = sim.get_mean_h()

    print(
        f"Tick {sim.tick:>7} | "
        f"Pop: {pop:>4} | "
        f"Food: {food:>4} | "
        f"Mean H: {mean_h:.2f} | "
        f"B/D: +{stats.births}/-{stats.deaths_starvation + stats.deaths_collision}"
    )
