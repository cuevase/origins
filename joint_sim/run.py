#!/usr/bin/env python3
"""
Entry point for the Joint Distribution Evolution Simulation.

Usage:
    python run.py                    # Run with default config
    python run.py --ticks 10000      # Run for 10k ticks
    python run.py --visualize        # Show ASCII world periodically
    python run.py --analyze          # Only run analysis on existing data
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

from config import Config, default_config
from simulation import Simulation
from logger import Logger, print_progress
from analysis import run_analysis


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Joint Distribution Evolution Simulation"
    )

    # Run modes
    parser.add_argument(
        "--analyze", action="store_true",
        help="Only run analysis on existing data (skip simulation)"
    )

    # Simulation parameters
    parser.add_argument(
        "--ticks", type=int, default=None,
        help="Number of ticks to simulate (default: 100000)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility"
    )

    # World parameters
    parser.add_argument(
        "--world-size", type=int, default=None,
        help="World size in spots (default: 1000)"
    )
    parser.add_argument(
        "--initial-food", type=int, default=None,
        help="Initial food count (default: 200)"
    )
    parser.add_argument(
        "--food-rate", type=int, default=None,
        help="Food spawn rate per tick (default: 2)"
    )

    # Organism parameters
    parser.add_argument(
        "--population", type=int, default=None,
        help="Initial population size (default: 50)"
    )
    parser.add_argument(
        "--repro-threshold", type=int, default=None,
        help="Energy needed to reproduce (default: 8)"
    )

    # Metabolism parameters
    parser.add_argument(
        "--decay-interval", type=int, default=None,
        help="Ticks between metabolic decay (default: 5)"
    )

    # Mutation parameters
    parser.add_argument(
        "--mutation-rate", type=float, default=None,
        help="Mutation rate per joint entry (default: 0.05)"
    )
    parser.add_argument(
        "--mutation-sigma", type=float, default=None,
        help="Mutation perturbation std dev (default: 0.05)"
    )

    # Output parameters
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: outputs)"
    )
    parser.add_argument(
        "--log-interval", type=int, default=None,
        help="Ticks between log snapshots (default: 100)"
    )

    # Display options
    parser.add_argument(
        "--visualize", action="store_true",
        help="Show ASCII world visualization periodically"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output"
    )
    parser.add_argument(
        "--progress-interval", type=int, default=1000,
        help="Ticks between progress updates (default: 1000)"
    )

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    """Build config from command line args, falling back to defaults."""
    config = Config()

    # Apply overrides
    if args.ticks is not None:
        config.simulation.total_ticks = args.ticks
    if args.seed is not None:
        config.simulation.random_seed = args.seed
    if args.log_interval is not None:
        config.simulation.log_interval = args.log_interval

    if args.world_size is not None:
        config.world.size = args.world_size
    if args.initial_food is not None:
        config.world.initial_food = args.initial_food
    if args.food_rate is not None:
        config.world.food_spawn_rate = args.food_rate

    if args.population is not None:
        config.organism.initial_population = args.population
    if args.repro_threshold is not None:
        config.organism.reproduction_threshold = args.repro_threshold

    if args.decay_interval is not None:
        config.metabolism.decay_interval = args.decay_interval

    if args.mutation_rate is not None:
        config.reproduction.mutation_rate = args.mutation_rate
    if args.mutation_sigma is not None:
        config.reproduction.mutation_sigma = args.mutation_sigma

    if args.output_dir is not None:
        config.output.base_dir = Path(args.output_dir)

    # Ensure directories exist
    config.__post_init__()

    return config


def run_simulation(config: Config, args: argparse.Namespace) -> None:
    """Run the main simulation loop."""
    # Set random seed if specified
    if config.simulation.random_seed is not None:
        np.random.seed(config.simulation.random_seed)

    # Print config summary
    print("\n" + "=" * 60)
    print("JOINT DISTRIBUTION EVOLUTION SIMULATION")
    print("=" * 60)
    print(f"World size: {config.world.size}")
    print(f"Initial population: {config.organism.initial_population}")
    print(f"Initial food: {config.world.initial_food}")
    print(f"Food spawn rate: {config.world.food_spawn_rate}/tick")
    print(f"Decay interval: {config.metabolism.decay_interval} ticks")
    print(f"Repro threshold: H >= {config.organism.reproduction_threshold}")
    print(f"Mutation rate: {config.reproduction.mutation_rate}")
    print(f"Total ticks: {config.simulation.total_ticks}")
    print(f"Output dir: {config.output.base_dir}")
    print("=" * 60 + "\n")

    # Initialize
    sim = Simulation(config)
    logger = Logger(config)

    # Log initial state
    logger.log(sim)

    start_time = time.time()
    last_progress_tick = 0

    # Main loop
    try:
        while sim.tick < config.simulation.total_ticks:
            stats = sim.run_tick()
            logger.accumulate_tick_stats(stats)

            # Log at intervals
            if sim.tick % config.simulation.log_interval == 0:
                logger.log(sim)

            # Progress update
            if not args.quiet and (sim.tick - last_progress_tick) >= args.progress_interval:
                print_progress(sim, stats)
                last_progress_tick = sim.tick

                if args.visualize:
                    print("\n" + sim.world.render_ascii(width=100) + "\n")

            # Check for extinction
            if sim.is_extinct():
                print(f"\nPopulation extinct at tick {sim.tick}!")
                break

    except KeyboardInterrupt:
        print(f"\nInterrupted at tick {sim.tick}")

    # Final stats
    elapsed = time.time() - start_time
    print(f"\nSimulation complete: {sim.tick} ticks in {elapsed:.1f}s")
    print(f"Final population: {sim.get_population_size()}")
    print(f"Ticks/sec: {sim.tick / elapsed:.1f}")

    # Save final state
    logger.save_final_state(sim)
    print(f"Results saved to {config.output.base_dir}")


def main() -> None:
    """Main entry point."""
    args = parse_args()
    config = build_config(args)

    if args.analyze:
        # Only run analysis
        run_analysis(config)
    else:
        # Run simulation then analysis
        run_simulation(config, args)
        print("\nRunning post-hoc analysis...")
        run_analysis(config)


if __name__ == "__main__":
    main()
