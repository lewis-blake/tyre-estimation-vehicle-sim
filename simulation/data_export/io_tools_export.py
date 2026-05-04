#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 22:53:39 2025

@author: lewisblake
"""

# io_tools_export.py
import json, os, numpy as np
import types


def sanitize_for_json(obj, _stack=None):
    """
    Convert `obj` into JSON-serializable structures.

    Improvements over earlier versions:
      * Detect cycles using the current recursion stack only (so shared references
        are duplicated rather than treated as cycles),
      * Do NOT treat primitives (ints/floats/bools/str/np scalars) as container objects for cycle detection.
      * Preserve interp1d-like objects (x/y) and provide helpful fallbacks.
    """
    if _stack is None:
        _stack = []

    # primitives: do NOT add to stack
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()

    # Determine if object is a container for cycle detection
    is_container = isinstance(obj, (dict, list, tuple, set, np.ndarray)) or hasattr(obj, "__dict__")

    if is_container:
        oid = id(obj)
        # If object is already on the current recursion stack -> real recursion/cycle
        if oid in _stack:
            return {"__cycle__": True, "__type__": getattr(obj, "__class__", type(obj)).__name__}
        # push to stack
        _stack.append(oid)

    # numpy arrays
    if isinstance(obj, np.ndarray):
        try:
            result = [sanitize_for_json(v, _stack) for v in obj.tolist()]
        finally:
            if is_container:
                _stack.pop()
        return result

    # dicts
    if isinstance(obj, dict):
        out = {}
        try:
            for k, v in obj.items():
                out[str(k)] = sanitize_for_json(v, _stack)
        finally:
            if is_container:
                _stack.pop()
        return out

    # lists/tuples/sets
    if isinstance(obj, (list, tuple, set)):
        try:
            return [sanitize_for_json(v, _stack) for v in obj]
        finally:
            if is_container:
                _stack.pop()

    # interp1d-like (objects exposing x/y or xp/fp)
    try:
        x = getattr(obj, "x", None) or getattr(obj, "xp", None)
        y = getattr(obj, "y", None) or getattr(obj, "fp", None)
        if x is not None and y is not None:
            try:
                x_list = np.array(x).tolist()
                y_list = np.array(y).tolist()
            except Exception:
                x_list = list(x)
                y_list = list(y)
            if is_container:
                _stack.pop()
            return {
                "__interp1d__": True,
                "class": getattr(obj, "__class__", type(obj)).__name__,
                "x": sanitize_for_json(x_list, _stack),
                "y": sanitize_for_json(y_list, _stack)
            }
    except Exception:
        # continue to next fallbacks
        pass

    # callables/functions
    if isinstance(obj, types.FunctionType) or callable(obj):
        try:
            rep = {"__callable__": True,
                   "name": getattr(obj, "__name__", None),
                   "module": getattr(obj, "__module__", None),
                   "repr": repr(obj)}
        except Exception:
            rep = {"__callable__": True, "repr": str(obj)}
        if is_container:
            _stack.pop()
        return rep

    # objects with __dict__: shallow snapshot
    if hasattr(obj, "__dict__"):
        state = {}
        try:
            for k, v in vars(obj).items():
                try:
                    state[str(k)] = sanitize_for_json(v, _stack)
                except Exception:
                    state[str(k)] = {"__repr__": repr(v), "__type__": getattr(v, "__class__", type(v)).__name__}
        finally:
            if is_container:
                _stack.pop()
        return {"__object__": True, "class": getattr(obj, "__class__", type(obj)).__name__, "state": state}

    # fallback: repr + typename
    try:
        rep = {"__repr__": repr(obj), "__type__": getattr(obj, "__class__", type(obj)).__name__}
    except Exception:
        rep = {"__unserializable__": True}
    if is_container:
        _stack.pop()
    return rep




def _np_to_py(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x

def save_full_export_json(out_path, df, corners_list, params, per_tyre_time_series=None, simulation_meta=None, indent=2):
    """
    Exports:
     - full vehicle state (every df column)
     - per-tyre arrays (Fx,Fy,Fz,SA,SR) either taken from df or per_tyre_time_series
     - drive torques (Tq_drive_RL, Tq_drive_RR) from df
     - corner->phase index lists and phase summaries
     - vehicle params under 'vehicle_params' (single copy)
     - tyre params under 'tyre_params'  (single copy)
    """
    # ensure output dir exists
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)

    # 1) top-level meta
    doc = {}
    doc['simulation_meta'] = simulation_meta or {}
     # sanitize params for JSON
    raw_vehicle_params = params.get('vehicle', {})
    raw_tyre_front = params.get('tyre_front', {})
    raw_tyre_rear  = params.get('tyre_rear', {})
    
    doc['vehicle_params'] = sanitize_for_json(raw_vehicle_params)
    doc['tyre_params'] = {
        'front': sanitize_for_json(raw_tyre_front),
        'rear' : sanitize_for_json(raw_tyre_rear)
    }
    # 2) full time series (all DataFrame columns)
    if hasattr(df, 'to_dict'):
        time_series = {col: _np_to_py(df[col].values) for col in df.columns}
    elif isinstance(df, dict):
        time_series = {k: _np_to_py(np.array(v)) for k, v in df.items()}
    else:
        raise ValueError("df must be a pandas DataFrame or dict-of-arrays")

    doc['time_series'] = time_series

    # 3) explicit per-tyre top-level arrays (guarantee availability even if you didn't pass per_tyre_time_series)
    tyres = ['FL','FR','RL','RR']
    per_tyre = {}
    for t in tyres:
        # canonical keys we expect: Fx_T, Fy_T, Fz_T, SA_T, SR_T
        # prefer explicit per_tyre_time_series if provided, otherwise try df columns
        per_tyre[t] = {}
        if per_tyre_time_series and t in per_tyre_time_series:
            for k, arr in per_tyre_time_series[t].items():
                per_tyre[t][k] = _np_to_py(np.array(arr))
        else:
            # fallback to df column names
            for metric, colfmt in [('Fx','Fx_{}'), ('Fy','Fy_{}'), ('Fz','Fz_{}'),
                                   ('SA_rad','SA_{}_rad'), ('SR','SR_{}')]:
                col = colfmt.format(t)
                if col in time_series:
                    per_tyre[t][metric] = time_series[col]
    doc['per_tyre_time_series'] = per_tyre

    # 4) drive torques, diff details (if present in df)
    for c in ('Tq_drive_RL','Tq_drive_RR','Tq_diff_transfer','lock_factor'):
        if c in time_series:
            doc.setdefault('diff_time_series', {})[c] = time_series[c]

    # 5) path (if present in params)
    path = params.get('path', {})
    if path:
        doc['path'] = {k: _np_to_py(v) for k, v in path.items()}

    # 6) corners: store indices & phase summaries, but do NOT repeat tyre_params here (they're at top-level)
    doc['corners'] = []
    def _metric_summary(metric, indices):
        arr = np.array(time_series.get(metric, []), dtype=float) if metric in time_series else None
        if arr is None or arr.size == 0 or len(indices) == 0:
            return None
        vals = arr[indices]
        rel = int(np.nanargmax(np.abs(vals)))
        return {'max_abs': float(np.abs(vals[rel])), 'max_idx': int(indices[rel]), 'value_at_max': float(vals[rel])}

    for c in corners_list:
        entry = {
            'corner_id': int(c.get('corner', c.get('id', -1))),
            'start_idx': int(c['start_idx']),
            'end_idx': int(c['end_idx']),
            'entry_indices': [int(i) for i in c.get('entry_indices', [])],
            'mid_indices':   [int(i) for i in c.get('mid_indices', [])],
            'exit_indices':  [int(i) for i in c.get('exit_indices', [])],
            'phases': {}
        }
        for phase_name, idxs in (('Entry', entry['entry_indices']),
                                 ('Mid', entry['mid_indices']),
                                 ('Exit', entry['exit_indices'])):
            ph = {'indices': idxs}
            # common summaries - you can extend the metric list
            summaries = {}
            for metric in ['ay_g','ay_lat','vx_mps','vy_mps','r_rad_s','Sum_Fz','Sum_Fx']:
                s = _metric_summary(metric, idxs)
                if s:
                    summaries[metric] = s
            ph['summaries'] = summaries

            entry['phases'][phase_name] = ph
        doc['corners'].append(entry)

    # 7) write file
    doc_to_dump = sanitize_for_json(doc)
    with open(out_path, 'w') as fh:
        json.dump(doc_to_dump, fh, indent=indent)
    print(f"Saved full export JSON to: {out_path}")
    return out_path
