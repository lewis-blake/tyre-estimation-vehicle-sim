# Tyre estimation — vehicle simulator

This repo is the 15-DOF vehicle simulator I split out from a larger tyre-parameter project.

## Setup

```bash
pip install -r requirements.txt
```

You need NumPy, SciPy, Matplotlib, Pandas, and PyYAML.

## Running the main simulator

From the `simulation` folder:

```bash
cd simulation

# Track following (edit configs or pass your own YAML)
python run_simulation.py --config configs/simulation_config.yaml

# Open-loop excitation mode
python run_simulation.py --excitation

# Live debug plot (or set live_debug_plot in the YAML)
python run_simulation.py --config configs/simulation_config.yaml --live-debug
```

YAML configs live in `simulation/configs/`. The default file at the top of `run_simulation.py` controls what runs if you don't pass `--config`.

Outputs go under `results/simulation/` (timestamped folders). Some runs also drop a copy under `data/synthetic/` if you use the export path wired into the analysis scripts.

## Acceleration event demo

`acceleration_event.py` is a small straight-line run that compares a few traction-control style approaches using the same vehicle model. Run it from `simulation/`:

```bash
python acceleration_event.py
```

## Repo layout

- `simulation/` — scripts, vehicle model, helpers, configs
- `shared/` — tyre and vehicle parameter presets (`run_simulation` loads tyre presets from `shared/tyre_configs/`)
- `estimator/` — contains the tyre model dispatcher and Pacejka implementations the simulator imports (`core.tyre_model` etc.). 
