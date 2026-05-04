#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 18:43:47 2025

@author: lewisblake
"""

# state_manager.py
"""
Manages the vehicle's state vector.
"""
import numpy as np

# Updated state vector layout for 7-DOF model
# Replaced single roll (phi) and roll rate (p) with front/rear pairs.
STATE_VECTOR_LAYOUT = {
    # Sprung Mass (Chassis) States
    'X_m':       {'size': 1, 'dims': (1,)},
    'Y_m':       {'size': 1, 'dims': (1,)},
    'Z_m':       {'size': 1, 'dims': (1,)},
    'phi_f_rad': {'size': 1, 'dims': (1,)}, # Front roll angle
    'phi_r_rad': {'size': 1, 'dims': (1,)}, # Rear roll angle
    'theta_rad': {'size': 1, 'dims': (1,)},
    'psi_rad':   {'size': 1, 'dims': (1,)},
    'vx_mps':    {'size': 1, 'dims': (1,)},
    'vy_mps':    {'size': 1, 'dims': (1,)},
    'vz_mps':    {'size': 1, 'dims': (1,)},
    'p_f_radps':   {'size': 1, 'dims': (1,)}, # Front roll rate
    'p_r_radps':   {'size': 1, 'dims': (1,)}, # Rear roll rate
    'q_radps':   {'size': 1, 'dims': (1,)},
    'r_radps':   {'size': 1, 'dims': (1,)},
    'r_dot_radps': {'size': 1, 'dims': (1,)},

    # Unsprung Mass (Wheel) States
    'wheel_Z_m':       {'size': 4, 'dims': (4,)},
    'wheel_Z_dot_mps': {'size': 4, 'dims': (4,)},

    # Wheel Rotational States
    'wheel_w_radps':   {'size': 4, 'dims': (4,)}
}


_current_idx = 0
for state_name, info in STATE_VECTOR_LAYOUT.items():
    info['slice'] = slice(_current_idx, _current_idx + info['size'])
    _current_idx += info['size']

STATE_VECTOR_SIZE = _current_idx

def pack_state_dict(state_dict):
    """Converts a state dictionary to a 1D NumPy state vector."""
    y = np.zeros(STATE_VECTOR_SIZE)
    for name, info in STATE_VECTOR_LAYOUT.items():
        if name in state_dict:
            y[info['slice']] = np.ravel(state_dict[name])
    return y

def unpack_state_vector(y):
    """Converts a 1D NumPy state vector back to a state dictionary."""
    state_dict = {}
    for name, info in STATE_VECTOR_LAYOUT.items():
        state_dict[name] = np.reshape(y[info['slice']], info['dims'])
    return state_dict

def get_initial_state_vector(initial_conditions_dict):
    """Creates the initial state vector from a dictionary of initial values."""
    return pack_state_dict(initial_conditions_dict)
