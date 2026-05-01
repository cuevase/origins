"""
Pygame visualization for the Joint Distribution Evolution Simulation.

Usage:
    python3 visualizer.py                    # Default settings
    python3 visualizer.py --speed 10         # 10 ticks per frame
    python3 visualizer.py --paused           # Start paused
"""
import argparse
import sys

import numpy as np

try:
    import pygame
except ImportError:
    print("Pygame not installed. Run: pip install pygame")
    sys.exit(1)

from config import Config, StateSpace
from simulation import Simulation


# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (34, 197, 94)       # Food
RED = (239, 68, 68)         # Organism
DARK_GREEN = (22, 163, 74)  # Food being eaten
BLUE = (59, 130, 246)       # Organism with high H
YELLOW = (250, 204, 21)     # Organism about to reproduce
GRAY = (75, 85, 99)         # Grid lines
DARK_BG = (17, 24, 39)      # Background


class Visualizer:
    """Pygame-based visualization of the simulation."""

    def __init__(
        self,
        config: Config,
        cell_size: int = 8,
        grid_width: int = 100,
        ticks_per_frame: int = 1,
        start_paused: bool = False
    ):
        self.config = config
        self.cell_size = cell_size
        self.grid_width = grid_width  # Cells per row
        self.grid_height = (config.world.size + grid_width - 1) // grid_width
        self.ticks_per_frame = ticks_per_frame
        self.paused = start_paused

        # Window dimensions
        self.world_width = grid_width * cell_size
        self.world_height = self.grid_height * cell_size
        self.sidebar_width = 260
        self.window_width = self.world_width + self.sidebar_width
        self.window_height = max(self.world_height, 520)  # Minimum height for sidebar

        # Initialize pygame
        pygame.init()
        pygame.display.set_caption("Joint Distribution Evolution")
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 14)
        self.font_large = pygame.font.SysFont("monospace", 18, bold=True)

        # Initialize simulation
        if config.simulation.random_seed is not None:
            np.random.seed(config.simulation.random_seed)
        self.sim = Simulation(config)

        # Stats tracking
        self.fps = 60
        self.actual_tps = 0  # Ticks per second
        self.last_stats = None

    def world_index_to_pixel(self, idx: int) -> tuple[int, int]:
        """Convert world index to pixel coordinates."""
        row = idx // self.grid_width
        col = idx % self.grid_width
        x = col * self.cell_size
        y = row * self.cell_size
        return x, y

    def draw_world(self) -> None:
        """Draw the world grid."""
        # Background
        world_rect = pygame.Rect(0, 0, self.world_width, self.world_height)
        pygame.draw.rect(self.screen, DARK_BG, world_rect)

        # Draw food
        for i in range(self.config.world.size):
            if self.sim.world.has_food[i]:
                x, y = self.world_index_to_pixel(i)
                rect = pygame.Rect(x + 1, y + 1, self.cell_size - 2, self.cell_size - 2)
                pygame.draw.rect(self.screen, GREEN, rect)

        # Draw organisms
        for org in self.sim.get_living_organisms():
            x, y = self.world_index_to_pixel(org.position)
            rect = pygame.Rect(x + 1, y + 1, self.cell_size - 2, self.cell_size - 2)

            # Color based on energy
            repro_thresh = self.config.organism.reproduction_threshold
            if org.h >= repro_thresh:
                color = YELLOW  # Ready to reproduce
            elif org.h >= repro_thresh - 2:
                color = BLUE    # High energy (close to repro)
            elif org.h >= 4:
                color = RED     # Normal
            else:
                color = (180, 80, 80)  # Low energy (dimmer red)

            pygame.draw.rect(self.screen, color, rect)

    def draw_sidebar(self) -> None:
        """Draw statistics sidebar."""
        # Sidebar background
        sidebar_rect = pygame.Rect(self.world_width, 0, self.sidebar_width, self.window_height)
        pygame.draw.rect(self.screen, (31, 41, 55), sidebar_rect)

        x = self.world_width + 15
        y = 15
        line_height = 22

        # Title
        title = self.font_large.render("SIMULATION STATS", True, WHITE)
        self.screen.blit(title, (x, y))
        y += 35

        # Paused indicator
        if self.paused:
            pause_text = self.font_large.render("[ PAUSED ]", True, YELLOW)
            self.screen.blit(pause_text, (x, y))
            y += 30
        else:
            y += 5

        # Stats
        stats = [
            ("Tick", f"{self.sim.tick:,}"),
            ("Population", f"{self.sim.get_population_size()}"),
            ("Food", f"{self.sim.world.count_food():,}"),
            ("Mean H", f"{self.sim.get_mean_h():.2f}"),
            ("Median H", f"{self.sim.get_median_h():.1f}"),
        ]

        for label, value in stats:
            text = self.font.render(f"{label}:", True, (156, 163, 175))
            self.screen.blit(text, (x, y))
            val_text = self.font.render(value, True, WHITE)
            self.screen.blit(val_text, (x + 130, y))
            y += line_height

        y += 10

        # Deaths/births
        stats2 = [
            ("Births", f"{self.sim.total_births:,}"),
            ("Deaths (starve)", f"{self.sim.total_deaths_starvation:,}"),
            ("Deaths (collide)", f"{self.sim.total_deaths_collision:,}"),
        ]

        for label, value in stats2:
            text = self.font.render(f"{label}:", True, (156, 163, 175))
            self.screen.blit(text, (x, y))
            val_text = self.font.render(value, True, WHITE)
            self.screen.blit(val_text, (x + 130, y))
            y += line_height

        y += 10

        # Speed info
        speed_text = self.font.render(f"Speed: {self.ticks_per_frame} tick/frame", True, WHITE)
        self.screen.blit(speed_text, (x, y))
        y += line_height
        tps_text = self.font.render(f"TPS: {self.actual_tps:.0f}", True, WHITE)
        self.screen.blit(tps_text, (x, y))
        y += line_height + 15

        # Legend
        pygame.draw.line(self.screen, GRAY, (x, y), (x + 240, y))
        y += 10
        legend_text = self.font.render("LEGEND:", True, WHITE)
        self.screen.blit(legend_text, (x, y))
        y += line_height

        repro_thresh = self.config.organism.reproduction_threshold
        legend_items = [
            (GREEN, "Food"),
            (RED, "Organism"),
            (BLUE, f"High H (>={repro_thresh-2})"),
            (YELLOW, f"Repro (H>={repro_thresh})"),
        ]
        for color, label in legend_items:
            pygame.draw.rect(self.screen, color, (x, y + 3, 12, 12))
            text = self.font.render(label, True, (200, 200, 200))
            self.screen.blit(text, (x + 20, y))
            y += line_height

        y += 10

        # Controls
        pygame.draw.line(self.screen, GRAY, (x, y), (x + 240, y))
        y += 10
        ctrl_text = self.font.render("CONTROLS:", True, WHITE)
        self.screen.blit(ctrl_text, (x, y))
        y += line_height

        controls = [
            "SPACE - Pause/Resume",
            "UP/DOWN - Speed +/-",
            "R - Reset",
            "Q/ESC - Quit",
        ]
        for line in controls:
            text = self.font.render(line, True, (156, 163, 175))
            self.screen.blit(text, (x, y))
            y += line_height

    def handle_events(self) -> bool:
        """Handle pygame events. Returns False to quit."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    return False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_UP:
                    self.ticks_per_frame = min(100, self.ticks_per_frame + 1)
                elif event.key == pygame.K_DOWN:
                    self.ticks_per_frame = max(1, self.ticks_per_frame - 1)
                elif event.key == pygame.K_r:
                    # Reset simulation
                    if self.config.simulation.random_seed is not None:
                        np.random.seed(self.config.simulation.random_seed)
                    self.sim = Simulation(self.config)
        return True

    def run(self) -> None:
        """Main visualization loop."""
        running = True
        tick_counter = 0

        print("Visualization started. Press Q or ESC to quit.")

        while running:
            # Handle events
            running = self.handle_events()

            # Update simulation
            if not self.paused and not self.sim.is_extinct():
                for _ in range(self.ticks_per_frame):
                    self.sim.run_tick()
                    tick_counter += 1

            # Calculate TPS
            self.actual_tps = tick_counter * self.clock.get_fps() / max(1, self.ticks_per_frame)

            # Draw
            self.screen.fill(BLACK)
            self.draw_world()
            self.draw_sidebar()

            # Extinction message
            if self.sim.is_extinct():
                ext_text = self.font_large.render("POPULATION EXTINCT", True, RED)
                text_rect = ext_text.get_rect(center=(self.world_width // 2, self.world_height // 2))
                pygame.draw.rect(self.screen, BLACK, text_rect.inflate(20, 10))
                self.screen.blit(ext_text, text_rect)

            pygame.display.flip()
            self.clock.tick(self.fps)

            tick_counter = 0

        pygame.quit()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Visualize Joint Distribution Evolution")

    # Visualization settings
    parser.add_argument("--cell-size", type=int, default=6, help="Pixel size per cell (default: 6)")
    parser.add_argument("--grid-width", type=int, default=100, help="Cells per row (default: 100)")
    parser.add_argument("--speed", type=int, default=1, help="Ticks per frame (default: 1)")
    parser.add_argument("--paused", action="store_true", help="Start paused")
    parser.add_argument("--fps", type=int, default=60, help="Target FPS (default: 60)")

    # Simulation parameters (same as run.py)
    parser.add_argument("--world-size", type=int, default=None)
    parser.add_argument("--population", type=int, default=None)
    parser.add_argument("--initial-food", type=int, default=None)
    parser.add_argument("--food-rate", type=int, default=None)
    parser.add_argument("--decay-interval", type=int, default=None)
    parser.add_argument("--mutation-rate", type=float, default=None)
    parser.add_argument("--mutation-sigma", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    """Build config from args."""
    config = Config()

    if args.world_size is not None:
        config.world.size = args.world_size
    if args.population is not None:
        config.organism.initial_population = args.population
    if args.initial_food is not None:
        config.world.initial_food = args.initial_food
    if args.food_rate is not None:
        config.world.food_spawn_rate = args.food_rate
    if args.decay_interval is not None:
        config.metabolism.decay_interval = args.decay_interval
    if args.mutation_rate is not None:
        config.reproduction.mutation_rate = args.mutation_rate
    if args.mutation_sigma is not None:
        config.reproduction.mutation_sigma = args.mutation_sigma
    if args.seed is not None:
        config.simulation.random_seed = args.seed

    return config


def main():
    args = parse_args()
    config = build_config(args)

    viz = Visualizer(
        config=config,
        cell_size=args.cell_size,
        grid_width=args.grid_width,
        ticks_per_frame=args.speed,
        start_paused=args.paused
    )
    viz.fps = args.fps
    viz.run()


if __name__ == "__main__":
    main()
