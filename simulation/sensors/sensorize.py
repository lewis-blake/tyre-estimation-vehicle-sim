#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Subsample / add noise / bias / dropout to a simulation export (JSON or CSV).
Produces a time-aligned CSV with chosen channels.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import warnings


def load_sim_json_or_csv(path, default_dt=0.01):
    """
    Robust loader for simulation output.
    Supports CSV (must contain 'time' or 't') and JSON with either:
      - top-level 'time' list
      - 'time_series' dict-of-arrays (as in sim_full.json)
    Returns pandas.DataFrame with a 'time' column.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    if p.suffix.lower() == '.json':
        with open(p, 'r') as fh:
            doc = json.load(fh)

        if isinstance(doc, dict) and 'time' in doc and isinstance(doc['time'], list):
            return pd.DataFrame(doc)

        if isinstance(doc, dict) and 'time_series' in doc and isinstance(doc['time_series'], dict):
            ts = doc['time_series']
            flattened_ts = {}
            for k, v in ts.items():
                arr = np.array(v)
                if arr.ndim > 1:
                    if arr.shape[1] == 1:
                        flattened_ts[k] = arr.flatten()
                    else:
                        flattened_ts[k] = arr[:, 0]
                else:
                    flattened_ts[k] = arr
            df = pd.DataFrame(flattened_ts)
            if 'time' in df.columns:
                return df

            dt = None
            sim_meta = doc.get('simulation_meta') or {}
            for dt_key in ('dt', 'time_step', 'sample_dt', 'dt_s'):
                if dt_key in sim_meta and sim_meta[dt_key] is not None:
                    try:
                        dt = float(sim_meta[dt_key])
                        break
                    except Exception:
                        dt = None
            if dt is None:
                for cand in ('t', 'time_s', 'timestamp', 'time_sec'):
                    if cand in df.columns:
                        df = df.rename(columns={cand: 'time'})
                        return df
                warnings.warn(
                    "No explicit 'time' array found in JSON and simulation_meta.dt is missing. "
                    f"Using default_dt={default_dt} s."
                )
                dt = default_dt

            n = len(next(iter(ts.values())))
            time = np.arange(0, n * dt, dt, dtype=float)
            if len(time) != len(df):
                time = np.linspace(0.0, dt * (len(df)-1), len(df))
            df.insert(0, 'time', time)
            return df

        lists = {k: v for k, v in doc.items() if isinstance(v, list)}
        if lists:
            if 'time' in lists:
                return pd.DataFrame(lists)
            max_len = max(len(v) for v in lists.values())
            candidate = {k: np.array(v) for k, v in lists.items() if len(v) == max_len}
            df = pd.DataFrame(candidate)
            warnings.warn(
                "Loaded JSON lists fallback; no explicit 'time' found. "
                f"Using default_dt={default_dt} s to construct time column."
            )
            df.insert(0, 'time', np.arange(0, len(df)*default_dt, default_dt))
            return df

        raise ValueError("Unrecognized JSON structure. Expected 'time' or 'time_series' keys.")

    else:
        df = pd.read_csv(p)
        if 'time' not in df.columns:
            if 't' in df.columns:
                df = df.rename(columns={'t': 'time'})
            else:
                warnings.warn(
                    "CSV has no 'time' or 't' column -- inserting 'time' using default_dt"
                )
                df.insert(0, 'time', np.arange(0, len(df) * default_dt, default_dt))
        return df


def sensorize_dataframe(df, signals, out_freq_hz=100,
                        noise_std=None, bias=None, dropout_prob=None, timecol='time'):
    """Resample signals to a fixed output frequency and apply noise/bias/dropout."""
    t = df[timecol].values.astype(float)
    t0, t1 = t[0], t[-1]
    dt_out = 1.0 / out_freq_hz
    t_out = np.arange(t0, t1 + dt_out/2, dt_out)
    df_out = pd.DataFrame({timecol: t_out})
    for sig in signals:
        if sig not in df.columns:
            raise KeyError(f"Signal {sig} not in dataframe columns")
        vals = np.interp(t_out, t, df[sig].values)
        b = bias.get(sig, 0.0) if isinstance(bias, dict) else (bias or 0.0)
        vals = vals + b
        std = noise_std.get(sig, 0.0) if isinstance(noise_std, dict) else (noise_std or 0.0)
        if std > 0:
            vals = vals + np.random.normal(scale=std, size=vals.shape)
        p = dropout_prob.get(sig, 0.0) if isinstance(dropout_prob, dict) else (dropout_prob or 0.0)
        if p > 0:
            mask = np.random.rand(len(vals)) < p
            vals[mask] = np.nan
        df_out[sig] = vals
    return df_out


def export_sensor_file(input_path, output_csv, signals=None,
                       out_freq_hz=100, noise_std=None, bias=None, dropout_prob=None):
    """Load simulation data, sensorize it, and write to CSV."""
    df = load_sim_json_or_csv(input_path)
    if signals is None:
        default_signals = [
            'vx_mps', 'vy_mps', 'r_rad_s', 'ax_mps2', 'ay_mps2',
            'steer_cmd_rad', 'wheel_speed_FL', 'wheel_speed_FR', 'Sum_Fz'
        ]
        signals = [s for s in default_signals if s in df.columns]
    df_out = sensorize_dataframe(df, signals, out_freq_hz, noise_std, bias, dropout_prob)
    df_out.to_csv(output_csv, index=False)
    print(f"Saved sensorized file: {output_csv}")
    return output_csv


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Generate noisy sensor CSV from sim CSV/JSON")
    p.add_argument("input_file")
    p.add_argument("output_csv")
    p.add_argument("--signals", nargs="+", default=None)
    p.add_argument("--freq", type=float, default=100.0)
    args = p.parse_args()
    export_sensor_file(args.input_file, args.output_csv, signals=args.signals, out_freq_hz=args.freq)
