"""Tests for collision resolution logic."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from simulation import Simulation
from organism import Organism
from config import Config, StateSpace


def make_deterministic_organism(position: int, h: int, action: int) -> Organism:
    """Create an organism that always takes the specified action."""
    org = Organism(position=position, h=h)
    # Set joint to always choose the specified action
    org.joint[:, :, :] = 0.0
    org.joint[:, :, action] = 1.0
    org.joint = org.joint / org.joint.sum()
    return org


def test_mover_into_stationary():
    """Test that mover into occupied spot dies, stationary survives."""
    config = Config()
    config.world.size = 10
    config.world.initial_food = 0
    config.world.food_spawn_rate = 0
    config.organism.initial_population = 0  # We'll add manually
    config.metabolism.decay_interval = 1000  # Disable decay

    sim = Simulation(config)

    # Clear any auto-created organisms
    sim.organisms.clear()

    # Create two organisms: one at position 5 (staying), one at 4 (moving right)
    stationary = make_deterministic_organism(position=5, h=5, action=StateSpace.STAY)
    mover = make_deterministic_organism(position=4, h=5, action=StateSpace.MOVE_RIGHT)

    sim.organisms[stationary.id] = stationary
    sim.organisms[mover.id] = mover
    sim.world.place_organism(5, stationary.id)
    sim.world.place_organism(4, mover.id)

    # Run one tick
    stats = sim.run_tick()

    # Mover should be dead, stationary should be alive
    assert mover.alive == False
    assert stationary.alive == True
    assert stats.deaths_collision == 1


def test_two_movers_same_destination():
    """Test that two organisms moving to same spot both die."""
    config = Config()
    config.world.size = 10
    config.world.initial_food = 0
    config.world.food_spawn_rate = 0
    config.organism.initial_population = 0
    config.metabolism.decay_interval = 1000

    sim = Simulation(config)
    sim.organisms.clear()

    # Position 4 moves right to 5, position 6 moves left to 5
    mover1 = make_deterministic_organism(position=4, h=5, action=StateSpace.MOVE_RIGHT)
    mover2 = make_deterministic_organism(position=6, h=5, action=StateSpace.MOVE_LEFT)

    sim.organisms[mover1.id] = mover1
    sim.organisms[mover2.id] = mover2
    sim.world.place_organism(4, mover1.id)
    sim.world.place_organism(6, mover2.id)

    # Run one tick
    stats = sim.run_tick()

    # Both should be dead
    assert mover1.alive == False
    assert mover2.alive == False
    assert stats.deaths_collision == 2


def test_stay_is_safe():
    """Test that staying in place never causes collision."""
    config = Config()
    config.world.size = 10
    config.world.initial_food = 0
    config.world.food_spawn_rate = 0
    config.organism.initial_population = 0
    config.metabolism.decay_interval = 1000

    sim = Simulation(config)
    sim.organisms.clear()

    # All organisms stay in place
    org1 = make_deterministic_organism(position=3, h=5, action=StateSpace.STAY)
    org2 = make_deterministic_organism(position=5, h=5, action=StateSpace.STAY)
    org3 = make_deterministic_organism(position=7, h=5, action=StateSpace.STAY)

    sim.organisms[org1.id] = org1
    sim.organisms[org2.id] = org2
    sim.organisms[org3.id] = org3
    sim.world.place_organism(3, org1.id)
    sim.world.place_organism(5, org2.id)
    sim.world.place_organism(7, org3.id)

    # Run multiple ticks
    for _ in range(10):
        stats = sim.run_tick()
        assert stats.deaths_collision == 0

    # All should still be alive
    assert org1.alive == True
    assert org2.alive == True
    assert org3.alive == True


def test_safe_movement():
    """Test that moving to empty spot is safe."""
    config = Config()
    config.world.size = 10
    config.world.initial_food = 0
    config.world.food_spawn_rate = 0
    config.organism.initial_population = 0
    config.metabolism.decay_interval = 1000

    sim = Simulation(config)
    sim.organisms.clear()

    # Organism at 5 moves right to 6 (empty)
    mover = make_deterministic_organism(position=5, h=5, action=StateSpace.MOVE_RIGHT)
    sim.organisms[mover.id] = mover
    sim.world.place_organism(5, mover.id)

    # Run one tick
    stats = sim.run_tick()

    # Should be alive at new position
    assert mover.alive == True
    assert mover.position == 6
    assert stats.deaths_collision == 0


def test_circular_collision():
    """Test collision at world wrap boundary."""
    config = Config()
    config.world.size = 10
    config.world.initial_food = 0
    config.world.food_spawn_rate = 0
    config.organism.initial_population = 0
    config.metabolism.decay_interval = 1000

    sim = Simulation(config)
    sim.organisms.clear()

    # Position 9 moves right (to 0), position 0 stays
    mover = make_deterministic_organism(position=9, h=5, action=StateSpace.MOVE_RIGHT)
    stationary = make_deterministic_organism(position=0, h=5, action=StateSpace.STAY)

    sim.organisms[mover.id] = mover
    sim.organisms[stationary.id] = stationary
    sim.world.place_organism(9, mover.id)
    sim.world.place_organism(0, stationary.id)

    # Run one tick
    stats = sim.run_tick()

    # Mover should die, stationary survives
    assert mover.alive == False
    assert stationary.alive == True
    assert stats.deaths_collision == 1


def test_starvation_death():
    """Test death from H reaching 0."""
    config = Config()
    config.world.size = 10
    config.world.initial_food = 0
    config.world.food_spawn_rate = 0
    config.organism.initial_population = 0
    config.metabolism.decay_interval = 1  # Decay every tick

    sim = Simulation(config)
    sim.organisms.clear()

    org = make_deterministic_organism(position=5, h=1, action=StateSpace.STAY)
    sim.organisms[org.id] = org
    sim.world.place_organism(5, org.id)

    # First tick - no decay (tick starts at 0)
    stats1 = sim.run_tick()
    assert org.alive == True  # Still alive after first tick

    # Second tick - decay happens (tick is now 1)
    stats2 = sim.run_tick()

    # Should be dead from starvation
    assert org.alive == False
    assert stats2.deaths_starvation == 1


def test_food_consumption():
    """Test that organism eats food when moving to food spot."""
    config = Config()
    config.world.size = 10
    config.world.initial_food = 0
    config.world.food_spawn_rate = 0
    config.organism.initial_population = 0
    config.metabolism.decay_interval = 1000

    sim = Simulation(config)
    sim.organisms.clear()

    # Place food at position 6
    sim.world.has_food[6] = True

    # Organism at 5 moves right to food
    mover = make_deterministic_organism(position=5, h=5, action=StateSpace.MOVE_RIGHT)
    sim.organisms[mover.id] = mover
    sim.world.place_organism(5, mover.id)

    # Run one tick
    stats = sim.run_tick()

    # Should have eaten food
    assert mover.h == 6
    assert stats.food_consumed == 1
    assert sim.world.has_food_at(6) == False


if __name__ == "__main__":
    test_mover_into_stationary()
    test_two_movers_same_destination()
    test_stay_is_safe()
    test_safe_movement()
    test_circular_collision()
    test_starvation_death()
    test_food_consumption()
    print("All collision tests passed!")
