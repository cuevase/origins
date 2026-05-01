# Origins — Joint Distribution Evolution Simulation

A minimal evolutionary simulation where each organism maintains a **full joint probability distribution** over local neighborhood (**X**), internal energy (**H**), and action (**O**). Behavior is sampled from that joint; offspring inherit a **mutated, renormalized** joint. The environment is a **one-dimensional circular world** with food spawning, metabolic decay, and collision death.

Useful for experimenting with how selection pressure reshapes high-dimensional categorical policies without hand-specifying neural nets or explicit fitness functions beyond survival and reproduction.

## What gets simulated

- **State space**: 27 neighborhood encodings (`3³` spots: empty / food / occupied), 11 discrete energy levels (0–10), 3 actions (move left, stay, move right).
- **Organism**: Position, energy `H`, and a `27 × 11 × 3` joint array (normalized). Actions are drawn as `P(O | X, H)` from the appropriate slice.
- **World**: `World` wraps toroidally; each tick organisms choose moves; **collisions** (multiple movers to one cell, or moving into someone who stays) remove organisms; eating increases `H`; periodic **decay** lowers `H`; reproduction requires energy above a threshold and places offspring in an empty neighbor slot with a mutated joint.
- **Outputs**: Time series in `logs.csv` (population, energy stats, births/deaths intervals, KL divergence vs. uniform joint), periodic `.npy` joint snapshots, and matplotlib plots under `outputs/plots/` after each run.

## Requirements

Python 3.10+ recommended (uses `int | None` style hints). Dependencies are listed in [`joint_sim/requirements.txt`](joint_sim/requirements.txt):

- `numpy`, `scipy`, `matplotlib`, `pandas`
- `pygame` — optional; used by [`joint_sim/visualizer.py`](joint_sim/visualizer.py)

## Quick start

Install and run **from** the `joint_sim` directory (imports assume that working directory):

```bash
cd joint_sim
python3 -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
python3 run.py
```

By default this uses values from [`joint_sim/config.py`](joint_sim/config.py) (`100_000` ticks, world size `1000`, etc.). Override with CLI flags or edit `Config` dataclasses in code.

Analysis (population curve, KL vs. uniform, marginals and other plots) runs **automatically** after the simulation. To regenerate plots from existing logs without rerunning:

```bash
python3 run.py --analyze
```

Optional **ASCII** world printout during runs:

```bash
python3 run.py --visualize --progress-interval 2000
```

Optional **Pygame** window:

```bash
python3 visualizer.py
```

## CLI reference

| Flag | Role |
|------|------|
| `--ticks` | Simulation length |
| `--seed` | RNG seed |
| `--world-size` | Cells on the ring |
| `--initial-food`, `--food-rate` | Initial food count and spawn rate per tick |
| `--population` | Starting organism count |
| `--repro-threshold` | Minimum `H` to reproduce |
| `--decay-interval` | Ticks between `-1` energy metabolic steps |
| `--mutation-rate`, `--mutation-sigma` | Per-entry mutation probability and Gaussian noise scale |
| `--output-dir` | Base output directory (default `outputs`) |
| `--log-interval` | Tick spacing for CSV / snapshot writes |
| `--quiet`, `--progress-interval`, `--visualize` | Console noise and ASCII preview |

Example long run matching a typical exploratory profile:

```bash
python3 run.py --ticks 100000 --population 50 --food-rate 10 --initial-food 400 \
  --decay-interval 15 --progress-interval 5000 \
  --mutation-rate 0.01 --mutation-sigma 0.02
```

For a tabular rundown of hyperparameters and intuition, see [`hyperparameter_reference.txt`](hyperparameter_reference.txt) at the repo root.

## Repository layout

| Path | Purpose |
|------|---------|
| `joint_sim/run.py` | CLI entrypoint: simulate → log → analyze |
| `joint_sim/config.py` | Dataclass configs + `StateSpace` constants |
| `joint_sim/organism.py` | Joint representation, sampling, reproduction / mutation |
| `joint_sim/world.py` | Circular grid, food, sensing, collisions helpers |
| `joint_sim/simulation.py` | Tick loop |
| `joint_sim/logger.py` | CSV + snapshot + KL metrics |
| `joint_sim/analysis.py` | Matplotlib plots from logs and snapshots |
| `joint_sim/visualizer.py` | Live pygame view |
| `joint_sim/tests/` | Unit tests (`pytest` style) |

## Tests

From `joint_sim`:

```bash
pip install pytest
pytest tests/
```

## License

No license file is included in this repository; add one if you plan to publish or accept contributions.
