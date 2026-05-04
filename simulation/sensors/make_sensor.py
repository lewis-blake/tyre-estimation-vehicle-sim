#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert simulation output (sim_full.json) into a sensorized CSV file
suitable for the estimator. Configures which signals to extract, noise
levels, biases, and output frequency.

Usage:
    python make_sensor.py
    python make_sensor.py --input path/to/sim_full.json --output path/to/sensorized.csv
"""

from pathlib import Path
import sys
import argparse

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from sensorize import export_sensor_file

# Default data directory (relative to repo root)
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "synthetic"
DEFAULT_INPUT = DEFAULT_DATA_DIR / "sim_full.json"
DEFAULT_OUTPUT = DEFAULT_DATA_DIR / "sensorized.csv"

# Signals to extract from the simulation
SIGNALS = [
    "r_radps", "r_dot_radps", "vx_mps", "vy_mps",
    "steer_cmd_rad", "ay_mps2", "ax_mps2",
    "wheel_w_radps_0", "wheel_w_radps_1", "wheel_w_radps_2", "wheel_w_radps_3",
    "T_FL", "T_FR", "T_RL", "T_RR",
]

# Per-signal noise standard deviation (set to 0 for clean data)
NOISE_STD = {
    "ay_mps2": 0,
    "ax_mps2": 0,
    "vx_mps": 0,
    "r_radps": 0,
    "T_FL": 0,
    "T_FR": 0,
    "T_RL": 0,
    "T_RR": 0,
}

# Per-signal constant bias
BIAS = {
    "vx_mps": 0.0,
}

# Per-signal dropout probability (per sample)
DROPOUT_PROB = {
    "vx_mps": 0,
}

# Output frequency in Hz
OUT_FREQ_HZ = 100


def main():
    parser = argparse.ArgumentParser(description="Generate sensorized CSV from simulation JSON")
    parser.add_argument("--input", type=str, default=None,
                        help=f"Path to sim_full.json (default: {DEFAULT_INPUT})")
    parser.add_argument("--output", type=str, default=None,
                        help=f"Path to output CSV (default: {DEFAULT_OUTPUT})")
    args = parser.parse_args()

    input_file = Path(args.input) if args.input else DEFAULT_INPUT
    output_csv = Path(args.output) if args.output else DEFAULT_OUTPUT

    if not input_file.exists():
        raise FileNotFoundError(
            f"sim_full.json not found at: {input_file}\n"
            f"Run the simulation first (simulation/run_simulation.py) to generate it."
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    export_sensor_file(
        str(input_file), str(output_csv),
        signals=SIGNALS,
        out_freq_hz=OUT_FREQ_HZ,
        noise_std=NOISE_STD,
        bias=BIAS,
        dropout_prob=DROPOUT_PROB,
    )
    print(f"Wrote: {output_csv}")


if __name__ == "__main__":
    main()
