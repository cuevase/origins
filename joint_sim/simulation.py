"""
Simulation tick loop with action resolution, death, and reproduction.
"""
import random
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from config import Config, StateSpace
from organism import Organism
from world import World


@dataclass
class TickStats:
    """Statistics for a single tick."""
    births: int = 0
    deaths_starvation: int = 0
    deaths_collision: int = 0
    food_consumed: int = 0


class Simulation:
    """
    Main simulation controller.

    Handles the tick loop, organism actions, collisions, death, and reproduction.
    """

    def __init__(self, config: Config):
        self.config = config
        self.tick = 0
        self.world = World(config.world)
        self.organisms: dict[int, Organism] = {}

        # Initialize population
        self._initialize_population()

        # Track stats
        self.total_births = 0
        self.total_deaths_starvation = 0
        self.total_deaths_collision = 0

    def _initialize_population(self) -> None:
        """Create initial population with random positions and joints."""
        positions = self.world.get_random_empty_positions(
            self.config.organism.initial_population
        )

        # If not enough empty spots, use what we have
        if len(positions) < self.config.organism.initial_population:
            # Find any unoccupied spots
            unoccupied = [i for i in range(self.world.size)
                          if not self.world.is_occupied(i)]
            positions = random.sample(
                unoccupied,
                min(len(unoccupied), self.config.organism.initial_population)
            )

        for pos in positions:
            org = Organism(
                position=pos,
                h=self.config.organism.initial_h
            )
            self.organisms[org.id] = org
            self.world.place_organism(pos, org.id)

    def run_tick(self) -> TickStats:
        """
        Execute one tick of the simulation.

        Returns:
            TickStats for this tick
        """
        stats = TickStats()

        # 1. Get living organisms and shuffle
        living = [o for o in self.organisms.values() if o.alive]
        random.shuffle(living)

        # 2. Collect all intended moves
        moves: dict[int, tuple[int, int]] = {}  # org_id -> (old_pos, new_pos)
        actions: dict[int, int] = {}  # org_id -> action

        for org in living:
            x_idx = self.world.sense_neighborhood(org.position)
            action = org.sample_action(x_idx)
            actions[org.id] = action

            old_pos = org.position
            new_pos = self.world.get_new_position(old_pos, action)
            moves[org.id] = (old_pos, new_pos)

        # 3. Resolve collisions
        deaths_this_tick = set()

        # Group moves by destination
        dest_to_orgs: dict[int, list[int]] = defaultdict(list)
        for org_id, (old_pos, new_pos) in moves.items():
            if old_pos != new_pos:  # Only movers
                dest_to_orgs[new_pos].append(org_id)

        # Check for multi-collision (two+ organisms moving to same spot)
        for dest, org_ids in dest_to_orgs.items():
            if len(org_ids) > 1:
                # All die
                for org_id in org_ids:
                    deaths_this_tick.add(org_id)
                    stats.deaths_collision += 1

        # Check for mover-into-occupied
        for org_id, (old_pos, new_pos) in moves.items():
            if org_id in deaths_this_tick:
                continue
            if old_pos != new_pos:
                # Check if destination is currently occupied by a non-moving organism
                occupant_id = self.world.occupant_id[new_pos]
                if occupant_id != 0:
                    # Check if occupant is staying or also moving
                    if occupant_id in moves:
                        occ_old, occ_new = moves[occupant_id]
                        if occ_old == occ_new:  # Occupant is staying
                            deaths_this_tick.add(org_id)
                            stats.deaths_collision += 1
                    else:
                        # Occupant not in moves (shouldn't happen if alive, but safety)
                        deaths_this_tick.add(org_id)
                        stats.deaths_collision += 1

        # 4. Apply deaths from collisions
        for org_id in deaths_this_tick:
            org = self.organisms[org_id]
            self.world.remove_organism(org.position)
            org.die()

        # 5. Apply valid moves
        for org_id, (old_pos, new_pos) in moves.items():
            if org_id in deaths_this_tick:
                continue

            org = self.organisms[org_id]
            if old_pos != new_pos:
                self.world.remove_organism(old_pos)
                self.world.place_organism(new_pos, org_id)
                org.position = new_pos

            # 6. Check for food at new position
            if self.world.consume_food(org.position):
                org.eat(
                    self.config.metabolism.food_energy,
                    self.config.organism.max_h
                )
                stats.food_consumed += 1

        # 7. Metabolic decay (every k ticks)
        if self.tick > 0 and self.tick % self.config.metabolism.decay_interval == 0:
            for org in self.organisms.values():
                if org.alive:
                    org.decay(
                        self.config.metabolism.decay_amount,
                        self.config.organism.min_h
                    )

        # 8. Resolve deaths from starvation
        for org in list(self.organisms.values()):
            if org.alive and org.h <= self.config.organism.min_h:
                self.world.remove_organism(org.position)
                org.die()
                stats.deaths_starvation += 1

        # 9. Resolve reproductions
        for org in list(self.organisms.values()):
            if org.can_reproduce(self.config.organism.reproduction_threshold):
                empty_neighbors = self.world.get_empty_neighbors(org.position)
                if empty_neighbors:
                    offspring_pos = random.choice(empty_neighbors)
                    offspring = org.reproduce(
                        offspring_position=offspring_pos,
                        parent_h_after=self.config.reproduction.parent_h_after,
                        offspring_h=self.config.reproduction.offspring_h,
                        reproduction_config=self.config.reproduction
                    )
                    self.organisms[offspring.id] = offspring
                    self.world.place_organism(offspring_pos, offspring.id)
                    stats.births += 1

        # 10. Spawn new food
        self.world.spawn_food()

        # Update totals
        self.total_births += stats.births
        self.total_deaths_starvation += stats.deaths_starvation
        self.total_deaths_collision += stats.deaths_collision

        self.tick += 1
        return stats

    def get_living_organisms(self) -> list[Organism]:
        """Get list of all living organisms."""
        return [o for o in self.organisms.values() if o.alive]

    def get_population_size(self) -> int:
        """Get current population count."""
        return len(self.get_living_organisms())

    def get_mean_h(self) -> float:
        """Get mean energy across living organisms."""
        living = self.get_living_organisms()
        if not living:
            return 0.0
        return np.mean([o.h for o in living])

    def get_median_h(self) -> float:
        """Get median energy across living organisms."""
        living = self.get_living_organisms()
        if not living:
            return 0.0
        return float(np.median([o.h for o in living]))

    def get_population_averaged_joint(self) -> np.ndarray:
        """Compute elementwise mean of joints across living organisms."""
        living = self.get_living_organisms()
        if not living:
            return np.zeros((StateSpace.NUM_X, StateSpace.NUM_H, StateSpace.NUM_O))

        joints = np.stack([o.joint for o in living])
        return joints.mean(axis=0)

    def is_extinct(self) -> bool:
        """Check if population has died out."""
        return self.get_population_size() == 0
