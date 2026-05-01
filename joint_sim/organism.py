"""
Organism class with joint probability distribution over (X, H, O).
"""
import numpy as np
from config import StateSpace, ReproductionConfig


class Organism:
    """
    An organism with position, energy (H), and a joint distribution P(X, H, O).

    The joint distribution is a (27, 11, 3) array where:
    - 27 = possible neighborhood configurations (X)
    - 11 = energy levels 0-10 (H)
    - 3 = actions: move_left, stay, move_right (O)
    """

    _id_counter = 0

    def __init__(
        self,
        position: int,
        h: int,
        joint: np.ndarray | None = None,
        parent_id: int | None = None
    ):
        Organism._id_counter += 1
        self.id = Organism._id_counter
        self.position = position
        self.h = h
        self.alive = True
        self.parent_id = parent_id

        if joint is None:
            self.joint = self._random_joint()
        else:
            self.joint = joint

    @staticmethod
    def _random_joint() -> np.ndarray:
        """Create a random joint distribution (uniform over simplex)."""
        joint = np.random.rand(StateSpace.NUM_X, StateSpace.NUM_H, StateSpace.NUM_O)
        joint = joint / joint.sum()
        return joint

    def sample_action(self, x_idx: int) -> int:
        """
        Sample an action given current neighborhood (X) and energy (H).

        Args:
            x_idx: Encoded neighborhood state (0-26)

        Returns:
            Action index: MOVE_LEFT=0, STAY=1, MOVE_RIGHT=2
        """
        h_idx = self.h  # H is already 0-10
        slice_ = self.joint[x_idx, h_idx, :]

        # Normalize to get P(O | X, H)
        total = slice_.sum()
        if total <= 0:
            # Fallback to uniform if somehow all zero
            probs = np.ones(StateSpace.NUM_O) / StateSpace.NUM_O
        else:
            probs = slice_ / total

        return np.random.choice(StateSpace.NUM_O, p=probs)

    def eat(self, energy_gain: int, max_h: int) -> None:
        """Consume food and gain energy."""
        self.h = min(self.h + energy_gain, max_h)

    def decay(self, amount: int, min_h: int) -> None:
        """Metabolic decay - lose energy."""
        self.h = max(self.h - amount, min_h)

    def die(self) -> None:
        """Mark organism as dead."""
        self.alive = False

    def can_reproduce(self, threshold: int) -> bool:
        """Check if organism can reproduce (H >= threshold)."""
        return self.alive and self.h >= threshold

    def reproduce(
        self,
        offspring_position: int,
        parent_h_after: int,
        offspring_h: int,
        reproduction_config: ReproductionConfig
    ) -> "Organism":
        """
        Create offspring with mutated joint distribution.

        Args:
            offspring_position: Position for the new organism
            parent_h_after: Parent's H after reproduction
            offspring_h: Offspring's starting H
            reproduction_config: Mutation parameters

        Returns:
            New Organism instance
        """
        # Parent loses energy
        self.h = parent_h_after

        # Mutate joint for offspring
        offspring_joint = mutate_joint(
            self.joint,
            mutation_rate=reproduction_config.mutation_rate,
            sigma=reproduction_config.mutation_sigma,
            min_value=reproduction_config.min_joint_value
        )

        return Organism(
            position=offspring_position,
            h=offspring_h,
            joint=offspring_joint,
            parent_id=self.id
        )

    def __repr__(self) -> str:
        status = "alive" if self.alive else "dead"
        return f"Organism(id={self.id}, pos={self.position}, H={self.h}, {status})"


def mutate_joint(
    joint: np.ndarray,
    mutation_rate: float = 0.05,
    sigma: float = 0.05,
    min_value: float = 1e-9
) -> np.ndarray:
    """
    Apply per-entry point mutation to a joint distribution.

    Args:
        joint: Original joint distribution (27, 11, 3)
        mutation_rate: Probability of mutating each entry
        sigma: Standard deviation of Gaussian perturbation
        min_value: Minimum value to keep distribution positive

    Returns:
        Mutated and renormalized joint distribution
    """
    # Create mutation mask
    mask = np.random.rand(*joint.shape) < mutation_rate

    # Apply Gaussian noise where mask is True
    noise = np.random.normal(0, sigma, joint.shape) * mask
    new_joint = joint + noise

    # Keep all values positive
    new_joint = np.maximum(new_joint, min_value)

    # Renormalize
    new_joint = new_joint / new_joint.sum()

    return new_joint


def create_initial_joint() -> np.ndarray:
    """Create a random initial joint distribution."""
    return Organism._random_joint()


def compute_uniform_joint() -> np.ndarray:
    """Create a perfectly uniform joint distribution for comparison."""
    joint = np.ones((StateSpace.NUM_X, StateSpace.NUM_H, StateSpace.NUM_O))
    return joint / joint.sum()
