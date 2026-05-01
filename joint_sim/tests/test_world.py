"""Tests for World class."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from world import World
from config import WorldConfig, StateSpace


def test_world_creation():
    """Test world initializes correctly."""
    config = WorldConfig(size=100, initial_food=20, food_spawn_rate=2)
    world = World(config)

    assert world.size == 100
    assert world.count_food() == 20
    assert world.count_occupied() == 0


def test_circular_topology():
    """Test that world wraps around correctly."""
    config = WorldConfig(size=10, initial_food=0, food_spawn_rate=0)
    world = World(config)

    # Test left neighbor of position 0 is position 9
    x_idx = world.sense_neighborhood(0)
    left, self_, right = StateSpace.decode_x(x_idx)
    # All should be empty
    assert left == StateSpace.EMPTY
    assert self_ == StateSpace.EMPTY
    assert right == StateSpace.EMPTY


def test_spot_states():
    """Test spot state tracking."""
    config = WorldConfig(size=10, initial_food=0, food_spawn_rate=0)
    world = World(config)

    # Initially empty
    assert world.get_spot_state(5) == StateSpace.EMPTY

    # Add food
    world.has_food[5] = True
    assert world.get_spot_state(5) == StateSpace.FOOD

    # Add organism - should override food
    world.place_organism(5, organism_id=1)
    assert world.get_spot_state(5) == StateSpace.OCCUPIED

    # Food still there
    assert world.has_food_at(5) == True

    # Remove organism
    world.remove_organism(5)
    assert world.get_spot_state(5) == StateSpace.FOOD


def test_food_consumption():
    """Test food consumption."""
    config = WorldConfig(size=10, initial_food=0, food_spawn_rate=0)
    world = World(config)

    world.has_food[3] = True
    assert world.count_food() == 1

    # Consume food
    result = world.consume_food(3)
    assert result == True
    assert world.count_food() == 0

    # Try to consume again
    result = world.consume_food(3)
    assert result == False


def test_sense_neighborhood():
    """Test neighborhood sensing with correct encoding."""
    config = WorldConfig(size=10, initial_food=0, food_spawn_rate=0)
    world = World(config)

    # Set up: food at 4, organism at 6, query position 5
    world.has_food[4] = True
    world.place_organism(6, organism_id=1)

    x_idx = world.sense_neighborhood(5)
    left, self_, right = StateSpace.decode_x(x_idx)

    assert left == StateSpace.FOOD
    assert self_ == StateSpace.EMPTY
    assert right == StateSpace.OCCUPIED


def test_food_spawning():
    """Test food spawn rate."""
    config = WorldConfig(size=100, initial_food=0, food_spawn_rate=5)
    world = World(config)

    assert world.count_food() == 0

    world.spawn_food()
    assert world.count_food() == 5

    world.spawn_food()
    assert world.count_food() == 10


def test_empty_neighbors():
    """Test finding empty neighbors."""
    config = WorldConfig(size=10, initial_food=0, food_spawn_rate=0)
    world = World(config)

    # Position 5, both neighbors empty
    empty = world.get_empty_neighbors(5)
    assert 4 in empty
    assert 6 in empty
    assert len(empty) == 2

    # Occupy position 4
    world.place_organism(4, organism_id=1)
    empty = world.get_empty_neighbors(5)
    assert 4 not in empty
    assert 6 in empty
    assert len(empty) == 1


def test_position_calculation():
    """Test action-to-position calculation."""
    config = WorldConfig(size=10, initial_food=0, food_spawn_rate=0)
    world = World(config)

    # Move left from 0 should wrap to 9
    assert world.get_new_position(0, StateSpace.MOVE_LEFT) == 9

    # Move right from 9 should wrap to 0
    assert world.get_new_position(9, StateSpace.MOVE_RIGHT) == 0

    # Stay should not change position
    assert world.get_new_position(5, StateSpace.STAY) == 5


if __name__ == "__main__":
    test_world_creation()
    test_circular_topology()
    test_spot_states()
    test_food_consumption()
    test_sense_neighborhood()
    test_food_spawning()
    test_empty_neighbors()
    test_position_calculation()
    print("All world tests passed!")
