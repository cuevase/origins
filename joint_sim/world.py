"""
World class - 1D circular array with spot states and food dynamics.
"""
import numpy as np
from config import StateSpace, WorldConfig


class World:
    """
    A 1D circular world where organisms live.

    Each spot can be: EMPTY (0), FOOD (1), or OCCUPIED (2).
    OCCUPIED overrides FOOD for sensing purposes.
    """

    def __init__(self, config: WorldConfig):
        self.size = config.size
        self.food_spawn_rate = config.food_spawn_rate

        # Track spot states and occupancy separately
        # spots[i] = True if food is present (organism may or may not be there)
        # occupied[i] = organism.id if occupied, else 0
        self.has_food = np.zeros(self.size, dtype=bool)
        self.occupant_id = np.zeros(self.size, dtype=int)  # 0 means empty

        # Spawn initial food
        self._spawn_food_at_random(config.initial_food)

    def _spawn_food_at_random(self, count: int) -> None:
        """Spawn food at random empty spots."""
        empty_spots = np.where(~self.has_food & (self.occupant_id == 0))[0]
        if len(empty_spots) == 0:
            return

        count = min(count, len(empty_spots))
        chosen = np.random.choice(empty_spots, size=count, replace=False)
        self.has_food[chosen] = True

    def spawn_food(self) -> None:
        """Spawn new food per tick according to spawn rate."""
        self._spawn_food_at_random(self.food_spawn_rate)

    def place_organism(self, position: int, organism_id: int) -> None:
        """Place an organism at a position."""
        self.occupant_id[position] = organism_id

    def remove_organism(self, position: int) -> None:
        """Remove an organism from a position."""
        self.occupant_id[position] = 0

    def is_occupied(self, position: int) -> bool:
        """Check if a position has an organism."""
        return self.occupant_id[position] != 0

    def has_food_at(self, position: int) -> bool:
        """Check if a position has food (may be under an organism)."""
        return self.has_food[position]

    def consume_food(self, position: int) -> bool:
        """
        Consume food at position if present.

        Returns:
            True if food was consumed, False otherwise.
        """
        if self.has_food[position]:
            self.has_food[position] = False
            return True
        return False

    def get_spot_state(self, position: int) -> int:
        """
        Get the apparent state of a spot for sensing.
        OCCUPIED overrides FOOD.
        """
        if self.occupant_id[position] != 0:
            return StateSpace.OCCUPIED
        elif self.has_food[position]:
            return StateSpace.FOOD
        else:
            return StateSpace.EMPTY

    def sense_neighborhood(self, position: int) -> int:
        """
        Sense the 3-spot neighborhood around a position.

        Args:
            position: The organism's current position

        Returns:
            Encoded X index (0-26)
        """
        left_pos = (position - 1) % self.size
        right_pos = (position + 1) % self.size

        left = self.get_spot_state(left_pos)
        self_ = self.get_spot_state(position)
        right = self.get_spot_state(right_pos)

        return StateSpace.encode_x(left, self_, right)

    def get_empty_neighbors(self, position: int) -> list[int]:
        """Get list of empty neighboring positions."""
        left_pos = (position - 1) % self.size
        right_pos = (position + 1) % self.size

        empty = []
        if not self.is_occupied(left_pos):
            empty.append(left_pos)
        if not self.is_occupied(right_pos):
            empty.append(right_pos)
        return empty

    def get_new_position(self, position: int, action: int) -> int:
        """Calculate new position after taking an action."""
        if action == StateSpace.MOVE_LEFT:
            return (position - 1) % self.size
        elif action == StateSpace.MOVE_RIGHT:
            return (position + 1) % self.size
        else:  # STAY
            return position

    def count_food(self) -> int:
        """Count total food in the world."""
        return int(self.has_food.sum())

    def count_occupied(self) -> int:
        """Count occupied spots."""
        return int((self.occupant_id != 0).sum())

    def get_random_empty_positions(self, count: int) -> list[int]:
        """Get random empty (not occupied, no food) positions."""
        empty = np.where(~self.has_food & (self.occupant_id == 0))[0]
        if len(empty) == 0:
            return []
        count = min(count, len(empty))
        return list(np.random.choice(empty, size=count, replace=False))

    def render_ascii(self, width: int = 100) -> str:
        """
        Render the world as ASCII art.

        Args:
            width: Characters per line (wraps if world is larger)

        Returns:
            ASCII representation string
        """
        chars = []
        for i in range(self.size):
            if self.occupant_id[i] != 0:
                chars.append("O")  # Organism
            elif self.has_food[i]:
                chars.append("*")  # Food
            else:
                chars.append(".")  # Empty

        # Wrap into lines
        lines = []
        for i in range(0, len(chars), width):
            lines.append("".join(chars[i:i + width]))

        return "\n".join(lines)
