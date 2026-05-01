"""Tests for Organism class."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from organism import Organism, mutate_joint, compute_uniform_joint
from config import StateSpace, ReproductionConfig


def test_organism_creation():
    """Test organism initializes correctly."""
    org = Organism(position=50, h=5)

    assert org.position == 50
    assert org.h == 5
    assert org.alive == True
    assert org.joint.shape == (27, 11, 3)
    assert np.isclose(org.joint.sum(), 1.0)
    assert (org.joint >= 0).all()


def test_joint_normalization():
    """Test that joint is a valid probability distribution."""
    org = Organism(position=0, h=5)

    # Should sum to 1
    assert np.isclose(org.joint.sum(), 1.0)

    # All entries should be non-negative
    assert (org.joint >= 0).all()


def test_sample_action():
    """Test action sampling."""
    # Create organism with known joint
    org = Organism(position=0, h=5)

    # Set up joint to always choose STAY for a specific (X, H)
    org.joint[:, :, :] = 0.001  # Small base
    x_idx = 0
    h_idx = 5
    org.joint[x_idx, h_idx, StateSpace.STAY] = 1000  # Heavily favor STAY
    org.joint = org.joint / org.joint.sum()

    org.h = h_idx

    # Sample multiple times - should almost always be STAY
    actions = [org.sample_action(x_idx) for _ in range(100)]
    assert all(a == StateSpace.STAY for a in actions)


def test_eat():
    """Test eating increases energy."""
    org = Organism(position=0, h=5)

    org.eat(energy_gain=1, max_h=10)
    assert org.h == 6

    # Test cap at max
    org.h = 9
    org.eat(energy_gain=3, max_h=10)
    assert org.h == 10


def test_decay():
    """Test metabolic decay."""
    org = Organism(position=0, h=5)

    org.decay(amount=1, min_h=0)
    assert org.h == 4

    # Test floor at min
    org.h = 1
    org.decay(amount=3, min_h=0)
    assert org.h == 0


def test_death():
    """Test organism death."""
    org = Organism(position=0, h=5)
    assert org.alive == True

    org.die()
    assert org.alive == False


def test_can_reproduce():
    """Test reproduction eligibility."""
    org = Organism(position=0, h=5)
    assert org.can_reproduce(max_h=10) == False

    org.h = 10
    assert org.can_reproduce(max_h=10) == True

    org.die()
    assert org.can_reproduce(max_h=10) == False


def test_reproduction():
    """Test reproduction creates offspring correctly."""
    parent = Organism(position=50, h=10)
    config = ReproductionConfig()

    offspring = parent.reproduce(
        offspring_position=51,
        parent_h_after=5,
        offspring_h=5,
        reproduction_config=config
    )

    # Parent should have reduced energy
    assert parent.h == 5

    # Offspring should be at correct position with correct energy
    assert offspring.position == 51
    assert offspring.h == 5
    assert offspring.alive == True

    # Offspring should have parent reference
    assert offspring.parent_id == parent.id

    # Offspring joint should be similar but not identical (mutation)
    assert offspring.joint.shape == parent.joint.shape
    # Could be identical if no mutations hit, but joint should be valid
    assert np.isclose(offspring.joint.sum(), 1.0)


def test_mutation():
    """Test joint mutation."""
    original = np.random.rand(27, 11, 3)
    original = original / original.sum()

    mutated = mutate_joint(
        original,
        mutation_rate=1.0,  # Mutate everything
        sigma=0.1,
        min_value=1e-9
    )

    # Should still be normalized
    assert np.isclose(mutated.sum(), 1.0)

    # Should be different from original
    assert not np.allclose(original, mutated)

    # Should have correct shape
    assert mutated.shape == original.shape


def test_mutation_keeps_positive():
    """Test mutation keeps all values positive."""
    original = np.ones((27, 11, 3)) * 0.001
    original = original / original.sum()

    # High sigma to potentially drive values negative
    mutated = mutate_joint(
        original,
        mutation_rate=1.0,
        sigma=0.5,
        min_value=1e-9
    )

    # All values should be positive
    assert (mutated > 0).all()
    assert np.isclose(mutated.sum(), 1.0)


def test_uniform_joint():
    """Test uniform joint creation."""
    uniform = compute_uniform_joint()

    assert uniform.shape == (27, 11, 3)
    assert np.isclose(uniform.sum(), 1.0)

    # All entries should be equal
    expected_value = 1.0 / (27 * 11 * 3)
    assert np.allclose(uniform, expected_value)


def test_unique_ids():
    """Test organisms get unique IDs."""
    org1 = Organism(position=0, h=5)
    org2 = Organism(position=1, h=5)
    org3 = Organism(position=2, h=5)

    assert org1.id != org2.id
    assert org2.id != org3.id
    assert org1.id != org3.id


if __name__ == "__main__":
    test_organism_creation()
    test_joint_normalization()
    test_sample_action()
    test_eat()
    test_decay()
    test_death()
    test_can_reproduce()
    test_reproduction()
    test_mutation()
    test_mutation_keeps_positive()
    test_uniform_joint()
    test_unique_ids()
    print("All organism tests passed!")
