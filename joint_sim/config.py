"""
Configuration file for Joint Distribution Evolution Simulation.
All tunable hyperparameters are centralized here for easy experimentation.
"""
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class WorldConfig:
    """World topology and food dynamics."""
    size: int = 1000                    # Number of spots in circular world
    initial_food: int = 400             # Food spots at simulation start
    food_spawn_rate: int = 10           # New food items per tick


@dataclass
class OrganismConfig:
    """Organism parameters."""
    initial_population: int = 50        # Starting number of organisms
    initial_h: int = 5                  # Starting energy for new organisms
    max_h: int = 10                     # Maximum energy cap
    reproduction_threshold: int = 8     # Energy needed to reproduce (lowered from 10)
    min_h: int = 0                      # Minimum energy (death threshold)


@dataclass
class MetabolismConfig:
    """Energy dynamics."""
    decay_interval: int = 10            # Ticks between metabolic decay
    decay_amount: int = 1               # Energy lost per decay event
    food_energy: int = 1                # Energy gained from eating food


@dataclass
class ReproductionConfig:
    """Reproduction and mutation parameters."""
    parent_h_after: int = 5             # Parent's H after reproduction
    offspring_h: int = 5                # Offspring's starting H
    mutation_rate: float = 0.05         # Probability per joint entry
    mutation_sigma: float = 0.05        # Std dev of perturbation
    min_joint_value: float = 1e-9       # Floor to keep joint positive


@dataclass
class SimulationConfig:
    """Simulation run parameters."""
    total_ticks: int = 100_000          # Total simulation length
    log_interval: int = 100             # Ticks between log snapshots
    random_seed: int | None = None      # For reproducibility (None = random)


@dataclass
class OutputConfig:
    """Output paths and settings."""
    base_dir: Path = field(default_factory=lambda: Path("outputs"))
    logs_file: str = "logs.csv"
    joint_snapshots_dir: str = "joint_snapshots"
    plots_dir: str = "plots"
    save_final_joints: bool = True
    save_lineage_sample: int = 10       # Number of lineages to save


@dataclass
class Config:
    """Master configuration combining all sub-configs."""
    world: WorldConfig = field(default_factory=WorldConfig)
    organism: OrganismConfig = field(default_factory=OrganismConfig)
    metabolism: MetabolismConfig = field(default_factory=MetabolismConfig)
    reproduction: ReproductionConfig = field(default_factory=ReproductionConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def __post_init__(self):
        """Ensure output directories exist."""
        self.output.base_dir = Path(self.output.base_dir)
        (self.output.base_dir / self.output.joint_snapshots_dir).mkdir(parents=True, exist_ok=True)
        (self.output.base_dir / self.output.plots_dir).mkdir(parents=True, exist_ok=True)


# State space constants (derived, not tunable)
class StateSpace:
    """Fixed encoding for X, H, O state spaces."""
    # Spot states
    EMPTY = 0
    FOOD = 1
    OCCUPIED = 2
    SPOT_STATES = 3

    # Actions
    MOVE_LEFT = 0
    STAY = 1
    MOVE_RIGHT = 2
    NUM_ACTIONS = 3

    # State space sizes
    NUM_X = 27          # 3^3 neighborhood configurations
    NUM_H = 11          # Energy levels 0-10
    NUM_O = 3           # Actions

    @staticmethod
    def encode_x(left: int, self_: int, right: int) -> int:
        """Encode 3-spot neighborhood as single index."""
        return left * 9 + self_ * 3 + right

    @staticmethod
    def decode_x(x_idx: int) -> tuple[int, int, int]:
        """Decode neighborhood index to (left, self, right)."""
        left = x_idx // 9
        self_ = (x_idx % 9) // 3
        right = x_idx % 3
        return left, self_, right


# Default configuration instance
default_config = Config()
