#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 18:40:52 2025

@author: lewisblake
"""

import numpy as np
import sys
from pathlib import Path

# Resolve config path from argv (same logic as main()) so we can read live_debug_plot before importing matplotlib
_script_dir = Path(__file__).resolve().parent
_config_path = _script_dir / "configs" / "simulation_config.yaml"  # default = track config (same as main())
if "--excitation" in sys.argv:
    _config_path = _script_dir / "configs" / "simulation_config_excitation.yaml"
elif "--config" in sys.argv:
    try:
        _idx = sys.argv.index("--config")
        if _idx + 1 < len(sys.argv):
            _config_path = Path(sys.argv[_idx + 1])
            if not _config_path.is_absolute():
                _config_path = _script_dir / _config_path
    except (ValueError, IndexError):
        pass
_live_debug_wanted = "--live-debug" in sys.argv
if not _live_debug_wanted and _config_path.exists():
    try:
        import yaml as _yaml
        with open(_config_path, "r") as _f:
            _cfg = _yaml.safe_load(_f)
        _live_debug_wanted = _cfg.get("live_debug_plot", False) if _cfg else False
    except Exception:
        pass

# Force interactive matplotlib backend before ANY matplotlib import (so live plot window can show)
if _live_debug_wanted:
    import os
    import platform
    _preferred = "macosx" if platform.system() == "Darwin" else "TkAgg"
    os.environ["MPLBACKEND"] = _preferred  # override Agg so the window can show
    import matplotlib as _mpl
    _backends = ("macosx", "TkAgg", "Qt5Agg", "Qt4Agg", "WXAgg", "GTK4Agg", "GTK3Agg") if platform.system() == "Darwin" else ("TkAgg", "Qt5Agg", "Qt4Agg", "macosx", "WXAgg", "GTK4Agg", "GTK3Agg")
    for _b in _backends:
        try:
            _mpl.use(_b)
            break
        except Exception:
            continue
else:
    import matplotlib as _mpl

import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import copy
import yaml

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from helpers.track_tools import generate_parametric_track, find_closest_point_on_path
from helpers.speed_profile_generator import generate_dynamic_speed_profile
from helpers.live_debug_plot import LiveDebugPlot, LIVE_UPDATE_INTERVAL_SEC
from vehicle.vehicle_dynamics import calculate_derivatives_for_solver, get_vehicle_forces
from helpers.controls import DriverModel, ExcitationController
from vehicle import state_manager as sm
from analysis.analysis_tools import plot_simulation_results, calculate_and_print_suspension_analytics

# Simulation Frequency Settings
CONTROL_HZ = 20
LOG_HZ = 100
TRACK_WIDTH_M = 30

# Default config file (relative to this script's directory). Overridden by --config / --excitation.
DEFAULT_CONFIG = "configs/simulation_config_excitation.yaml"  # or "configs/simulation_config.yaml" for track mode

def print_debug_stats(sim_time, state_dict, controls_dict, debug_info):
    """Prints a formatted block of debug information to the console."""
    vx = np.ravel(state_dict['vx_mps'])[0]
    X = np.ravel(state_dict['X_m'])[0]
    Y = np.ravel(state_dict['Y_m'])[0]
    r = np.ravel(state_dict['r_radps'])[0]
    # Display front and rear roll angles
    phi_f = np.ravel(state_dict['phi_f_rad'])[0]
    phi_r = np.ravel(state_dict['phi_r_rad'])[0]
    theta = np.ravel(state_dict['theta_rad'])[0]
    steer = controls_dict.get('steer_cmd_rad', 0.0)
    throttle_brake = controls_dict.get('throttle_brake_cmd', 0.0)
    target_v = controls_dict.get('target_v_mps', 0.0)
    sum_fx = debug_info.get('Sum_Fx', 0.0)
    sum_fy = debug_info.get('Sum_Fy', 0.0)
    slip_ratios = debug_info.get('slip_ratios', [0.0, 0.0, 0.0, 0.0])
    slip_angles = debug_info.get('slip_angles', [0.0, 0.0, 0.0, 0.0])

    print(f"--- DEBUG @ Sim Time: {sim_time:.3f}s ---")
    print(f"  Speed (vx):       {vx * 3.6:.1f} km/h  |  Target: {target_v * 3.6:.1f} km/h")
    print(f"  Position (X,Y):   ({X:.1f}, {Y:.1f}) m")
    print(f"  Roll (F/R):       {np.rad2deg(phi_f):.2f}/{np.rad2deg(phi_r):.2f} deg   |  Pitch (theta): {np.rad2deg(theta):.2f} deg")
    print(f"  Yaw Rate:         {np.rad2deg(r):.2f} deg/s")
    print(f"  Throttle/Brake:   {throttle_brake:.2f}  |  Steer Angle: {np.rad2deg(steer):.2f} deg")
    print(f"  Sum Forces (Fx,Fy): ({sum_fx:.1f}, {sum_fy:.1f}) N")
    print(f"  Slip Ratios (%)   FL: {slip_ratios[0]*100:6.2f}  FR: {slip_ratios[1]*100:6.2f}  RL: {slip_ratios[2]*100:6.2f}  RR: {slip_ratios[3]*100:6.2f}")
    print(f"  Slip Angles (deg) FL: {np.rad2deg(slip_angles[0]):6.2f}  FR: {np.rad2deg(slip_angles[1]):6.2f}  RL: {np.rad2deg(slip_angles[2]):6.2f}  RR: {np.rad2deg(slip_angles[3]):6.2f}")
    print("-" * 50)


def _get_per_wheel_slips(state_dict, controls, params):
    """Extract per-wheel slip angles and slip ratios from current state."""
    p_v = params.get('vehicle', {})
    vx = max(float(np.ravel(state_dict['vx_mps'])[0]), 1.0)
    vy = float(np.ravel(state_dict['vy_mps'])[0])
    r = float(np.ravel(state_dict['r_radps'])[0])
    steer = controls.get('steer_cmd_rad', 0.0)
    lf, lr = p_v.get('lf', 0.8), p_v.get('lr', 0.8)
    tw_f = p_v.get('track_width_f', 1.25) / 2.0
    tw_r = p_v.get('track_width_r', 1.25) / 2.0
    R_eff = p_v.get('R_eff', 0.32)

    omega = state_dict.get('wheel_w_radps')
    if omega is None:
        ow = np.full(4, vx / R_eff)
    else:
        ow = np.ravel(omega)

    # Per-wheel lateral velocity at contact patch
    vy_fl = vy + lf * r
    vy_fr = vy + lf * r
    vy_rl = vy - lr * r
    vy_rr = vy - lr * r
    vx_fl = vx - tw_f * r
    vx_fr = vx + tw_f * r
    vx_rl = vx - tw_r * r
    vx_rr = vx + tw_r * r

    alphas = np.array([
        abs(steer - np.arctan2(vy_fl, max(vx_fl, 0.5))),
        abs(steer - np.arctan2(vy_fr, max(vx_fr, 0.5))),
        abs(np.arctan2(vy_rl, max(vx_rl, 0.5))),
        abs(np.arctan2(vy_rr, max(vx_rr, 0.5))),
    ])
    kappas = np.array([
        abs((float(ow[0]) * R_eff - max(vx_fl, 0.5)) / max(vx_fl, 0.5)),
        abs((float(ow[1]) * R_eff - max(vx_fr, 0.5)) / max(vx_fr, 0.5)),
        abs((float(ow[2]) * R_eff - max(vx_rl, 0.5)) / max(vx_rl, 0.5)),
        abs((float(ow[3]) * R_eff - max(vx_rr, 0.5)) / max(vx_rr, 0.5)),
    ])
    return alphas, kappas


def get_tyre_temperatures(sim_time, state_dict, dt, controls, params):
    """Compute tyre temperatures [FL, FR, RL, RR] from config (sine/ode/lumped/none).
    For ODE and lumped modes, updates params['_tyre_temp_state'] in place for next step.
    Returns tuple (T_FL, T_FR, T_RL, T_RR, T_bulk_FL, T_bulk_FR, T_bulk_RL, T_bulk_RR) in °C.
    For non-lumped modes, bulk temps equal tread temps.
    """
    tc = params.get('tyre_temperature') or {}
    mode = tc.get('mode', 'none')
    p_v = params.get('vehicle', {})
    T_opt = params.get('tyre_front', {}).get('T_opt', 60.0)

    if mode == 'sine':
        T_f = tc.get('temp_sine_base_f', 60.0) + tc.get('temp_sine_amplitude_f', 0.0) * np.sin(
            2 * np.pi * tc.get('temp_sine_freq_f', 0.02) * sim_time)
        T_r = tc.get('temp_sine_base_r', 60.0) + tc.get('temp_sine_amplitude_r', 0.0) * np.sin(
            2 * np.pi * tc.get('temp_sine_freq_r', 0.015) * sim_time)
        return (float(T_f), float(T_f), float(T_r), float(T_r),
                float(T_f), float(T_f), float(T_r), float(T_r))

    if mode == 'ode':
        if '_tyre_temp_state' not in params:
            init = float(tc.get('temp_ode_initial', 25.0))
            params['_tyre_temp_state'] = np.array([init, init, init, init], dtype=float)
        state = params['_tyre_temp_state']
        vx = max(float(np.ravel(state_dict['vx_mps'])[0]), 1.0)
        vy = float(np.ravel(state_dict['vy_mps'])[0])
        r = float(np.ravel(state_dict['r_radps'])[0])
        steer = controls.get('steer_cmd_rad', 0.0)
        lf, lr = p_v.get('lf', 0.8), p_v.get('lr', 0.8)
        R_eff = p_v.get('R_eff', 0.32)
        omega = state_dict.get('wheel_w_radps')
        if omega is None:
            omega_f = omega_r = vx / R_eff
        else:
            ow = np.ravel(omega)
            omega_f = (float(ow[0]) + float(ow[1])) / 2.0
            omega_r = (float(ow[2]) + float(ow[3])) / 2.0
        alpha_f = abs(steer - np.arctan2(vy + lf * r, vx))
        alpha_r = abs(np.arctan2(vy - lr * r, vx))
        kappa_f = abs((omega_f * R_eff - vx) / vx)
        kappa_r = abs((omega_r * R_eff - vx) / vx)
        Fz_static = p_v.get('mass', 320) * 9.81 / 4.0
        h_alpha = tc.get('temp_ode_heat_rate_alpha', 0.0)
        h_kappa = tc.get('temp_ode_heat_rate_kappa', 0.0)
        if h_alpha == 0.0 and h_kappa == 0.0:
            h_combined = tc.get('temp_ode_heat_rate', 0.5)
            heat_f = h_combined * np.sqrt(alpha_f**2 + kappa_f**2) * Fz_static
            heat_r = h_combined * np.sqrt(alpha_r**2 + kappa_r**2) * Fz_static
        else:
            heat_f = (h_alpha * alpha_f + h_kappa * kappa_f) * Fz_static
            heat_r = (h_alpha * alpha_r + h_kappa * kappa_r) * Fz_static
        T_amb = tc.get('temp_ode_T_ambient', 25.0)
        cooling = tc.get('temp_ode_cooling_rate', 0.02)
        for i, heat in enumerate([heat_f, heat_f, heat_r, heat_r]):
            dT = heat - cooling * (state[i] - T_amb)
            state[i] += dT * dt
        t = (float(state[0]), float(state[1]), float(state[2]), float(state[3]))
        return t + t  # bulk = tread for simple ODE

    if mode == 'lumped':
        # Two-state lumped thermal model per wheel (from thermal_fit.py):
        # dTt/dt = (1/c_tread) * [heat - k_bt*(Tt-Tb) - k_ta*(Tt-T_air) - k_tg*(Tt-T_ground)]
        # dTb/dt = (1/c_bulk)  * k_bt * (Tt - Tb)
        if '_tyre_lumped_state' not in params:
            init = float(tc.get('lumped_T_initial', 25.0))
            params['_tyre_lumped_state'] = np.array(
                [init]*4 + [init]*4, dtype=float)  # [Tt_FL..Tt_RR, Tb_FL..Tb_RR]
        ls = params['_tyre_lumped_state']

        h_alpha = tc.get('lumped_h_alpha', 0.03493)
        h_kappa = tc.get('lumped_h_kappa', 0.0151)
        k_bt    = tc.get('lumped_k_bt', 0.423864)
        k_ta    = tc.get('lumped_k_ta', 0.037442)
        k_tg    = tc.get('lumped_k_tg', 0.001)
        c_tread = max(tc.get('lumped_c_tread', 1.0), 0.01)
        c_bulk  = max(tc.get('lumped_c_bulk', 9.278181), 0.01)
        T_air   = tc.get('lumped_T_air', 25.0)
        T_ground = tc.get('lumped_T_ground', 30.0)

        inv_ct = 1.0 / c_tread
        inv_cb = 1.0 / c_bulk

        alphas, kappas = _get_per_wheel_slips(state_dict, controls, params)
        Fz_static = p_v.get('mass', 320) * 9.81 / 4.0

        # Sub-stepping for numerical stability
        k_eff_tread = (k_bt + k_ta + k_tg) * inv_ct
        k_eff_bulk  = k_bt * inv_cb
        max_dt = 0.5 / max(k_eff_tread, k_eff_bulk, 1e-6)
        n_sub = max(1, int(np.ceil(dt / max_dt)))
        dt_sub = dt / n_sub

        for i in range(4):
            heat = (h_alpha * alphas[i] + h_kappa * kappas[i]) * Fz_static
            Tt = ls[i]
            Tb = ls[4 + i]
            for _ in range(n_sub):
                dTt = inv_ct * (heat - k_bt * (Tt - Tb) - k_ta * (Tt - T_air) - k_tg * (Tt - T_ground))
                dTb = inv_cb * k_bt * (Tt - Tb)
                Tt += dTt * dt_sub
                Tb += dTb * dt_sub
            ls[i] = Tt
            ls[4 + i] = Tb

        return (float(ls[0]), float(ls[1]), float(ls[2]), float(ls[3]),
                float(ls[4]), float(ls[5]), float(ls[6]), float(ls[7]))

    # none: constant from tyre T_opt
    return (T_opt, T_opt, T_opt, T_opt, T_opt, T_opt, T_opt, T_opt)


def calculate_initial_preload(params):
    """Calculates required spring preload to achieve target static ride height."""
    p_v = params['vehicle']
    g = params['environment']['g']
    m, L, lf, lr = p_v['m_s'], p_v['wheelbase'], p_v['lf'], p_v['lr']
    
    # Static vertical load per corner
    Fz_static_f_corner = (m * g * lr / L) / 2.0
    Fz_static_r_corner = (m * g * lf / L) / 2.0
    
    # The suspension force at the wheel must equal the static load
    
    preload_f = Fz_static_f_corner * p_v['motion_ratio_f']
    preload_r = Fz_static_r_corner * p_v['motion_ratio_r']
    
    print(f"Calculated Preload -> Front: {preload_f:.1f} N, Rear: {preload_r:.1f} N")
    return preload_f, preload_r

def main(config_path=None, live_debug_plot=None):
    # Load simulation config (optional)
    script_dir = Path(__file__).resolve().parent
    if config_path is None:
        config_path = script_dir / DEFAULT_CONFIG
    else:
        config_path = Path(config_path)
        if not config_path.is_absolute():
            config_path = script_dir / config_path
    if config_path.exists():
        with open(config_path, 'r') as f:
            sim_config = yaml.safe_load(f)
        print(f"Loaded config: {config_path.resolve()}")
    else:
        sim_config = {'mode': 'track', 'track': {'num_laps': 4, 'segments': []}}
        print(f"Config not found: {config_path.resolve()} — using defaults (track mode).")

    # Tyre temperature model from config (same structure as MPC)
    params_from_config = {}
    if sim_config.get('tyre_temperature'):
        params_from_config['tyre_temperature'] = sim_config['tyre_temperature']

    mode = sim_config.get('mode', 'track')
    if mode not in ('track', 'excitation'):
        raise ValueError(f"config mode must be 'track' or 'excitation', got '{mode}'")

    # Live debug plot: enable in config with live_debug_plot: true or via CLI --live-debug
    live_debug_plot_enabled = (
        live_debug_plot if live_debug_plot is not None
        else sim_config.get('live_debug_plot', False)
    )

    # Turn numpy warnings into errors that stop the program
    np.seterr(all='raise')
    
    plt.close('all')

    # Number of laps / duration from config
    if mode == 'track':
        track_cfg = sim_config.get('track', {})
        num_laps = track_cfg.get('num_laps', 4)
        track_segments = track_cfg.get('segments', [
            {'type': 'straight', 'length': 300},
            {'type': 'arc', 'radius': 15, 'angle': -120},
            {'type': 'straight', 'length': 300},
            {'type': 'arc', 'radius': 25, 'angle': -90},
            {'type': 'straight', 'length': 100},
        ])
    else:
        num_laps = 1
        exc_cfg = sim_config.get('excitation', {})
        track_segments = [{'type': 'straight', 'length': exc_cfg.get('path_length_m', 2000)}]

    path = generate_parametric_track(track_segments, step=0.5)
    
    params = {}
    params['vehicle'] = {
        'mass': 320, 
        'wheelbase': 1.6, 'track_width_f': 1.25, 'track_width_r': 1.25, 
        'COG_z': 0.2, 'COG_ratio_x': 0.5, # COG from bottom of chassis, NOT ground plane
        'Ix': 60, 'Iy': 100, 'Iz': 200, # Mass moments of inertia
        
        # Suspension Parameters
        'm_us_f': 9, 'm_us_r': 12, # Unsprung mass front/rear per corner
        'K_s_f': 20000, 'K_s_r': 30000, # Spring rates N/m
        'C_s_f': 1500, 'C_s_r': 2500, # Damping rates Ns/m
        'K_arb_f': 10000, 'K_arb_r': 0000, # Anti-roll bar stiffness Nm/rad
        # Heave Spring/Damper Parameters (at the shock)
        'K_heave_f': 0, 'K_heave_r': 0, # Heave spring rate (N/m)
        'C_heave_f': 0,  'C_heave_r': 0,  # Heave damper rate (Ns/m)
        'motion_ratio_f': 1,  # Wheel travel / Shock travel
        'motion_ratio_r': 1, # Wheel travel / Shock travel
        'motion_ratio_rate_f': 0, # % change in MR per meter of shock travel (positive for rising rate)
        'motion_ratio_rate_r': 0,
        'shock_stroke_max_m': 0.035, # 57mm max shock travel
        'shock_stroke_rebound_max_m': 0.025,
        'K_bump_stop': 1000000, # Very stiff bump stop spring (N/m)
        'ride_height_f': 0.05, # Target ride height from ground to chassis at front axle (m)
        'ride_height_r': 0.05, # Target ride height at rear axle (m)
        'K_torsion_Nm_rad': 800000, # Torsional stiffness of the chassis between axles

        'RCH_f_m': 0.0, # Front roll center height above ground (m)
        'RCH_r_m': 0.0, # Rear roll center height above ground (m)
        'PCH_f_m': 0, # Front pitch center height above ground (m)
        'PCH_r_m': 0, # Rear pitch center height above ground (m)
        'caster_f_deg': 0.0, 'caster_r_deg': 0.0, # Caster angle (deg) - rear is typically 0
        'kpi_f_deg': 0.0, 'kpi_r_deg': 0.0,       # Kingpin Inclination (deg) - rear is typically 0
        'scrub_radius_f': 0.0, # Scrub radius (m), positive for wheel center outside kingpin axis
        'static_toe_f_deg': -0, # Static toe angle (deg), negative for toe-out
        'static_toe_r_deg': 0, # Positive for toe-in
        'bump_steer_f_deg_per_mm': -0, # deg toe change per mm of jounce. Positive = toe IN.
        'bump_steer_r_deg_per_mm': -0,  # deg toe change per mm of jounce. Positive = toe IN.
        'static_camber_f_deg': -0.0, # Static camber angle (deg), negative for camber in
        'static_camber_r_deg': -0.0,
        'camber_roll_gain_f': 0, # deg camber change per deg of chassis roll
        'camber_roll_gain_r': 0,
        
        # Tire Parameters (vertical properties only)
        'K_tire': 200000, # Tire vertical stiffness N/m
        'R_eff': 0.22, 'I_wheel': 1.0,

        # Drivetrain and Brakes
        'brake_bias': 0.63, 'max_brake_torque': 2000,
        'motor_torque_curve': ([0, 1000, 4000, 8000, 12000], [230, 230, 220, 180, 100]),
        'gear_ratio': 3.5,
        'max_power_W': 80000,
        'diff_lock_torque_accel': 400, # Max locking torque difference under accel (Nm)
        'diff_lock_torque_coast': 100, # Max locking torque difference on coast/decel (Nm)
        'diff_preload': 50,           # Static preload on the differential clutch (Nm)
        
        # Aero
        'ClA': 0, 'CdA': 0.0, 'COP_z': 0.2, 'COP_x_ratio': 0.5,

        # Steering
        'ackerman_percentage': 0,
        
        'yaw_damping_coeff': 0.0,
        'max_speed_mps': 120
    }
    # Derived parameters
    p_v = params['vehicle']
    p_v['lf'] = p_v['wheelbase'] * (1 - p_v['COG_ratio_x'])
    p_v['lr'] = p_v['wheelbase'] * p_v['COG_ratio_x']
    m_us_total = 2 * (p_v['m_us_f'] + p_v['m_us_r'])
    p_v['m_s'] = p_v['mass'] - m_us_total # Sprung mass

    p_v['suspension_static_length_f_m'] = p_v['R_eff'] - p_v['ride_height_f']
    p_v['suspension_static_length_r_m'] = p_v['R_eff'] - p_v['ride_height_r']
    
    params['environment'] = {'g': 9.81, 'rho_air': 1.225}


    # Tyre Configuration Selection
    # Options: 'standard_config', 'ttc_config'
    #TYRE_CONFIG_NAME = 'standard_config'
    TYRE_CONFIG_NAME = 'ttc_config_paper'
    
    print(f"Loading '{TYRE_CONFIG_NAME}' tyre configuration...")
    
    # Load Tyre Config
    config_path = Path(__file__).parent.parent / "shared" / "tyre_configs" / f"{TYRE_CONFIG_NAME}.py"
    
    import importlib.util
    spec = importlib.util.spec_from_file_location("tyre_config", config_path)
    tyre_config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tyre_config_module)
    
    tyre_params_base = tyre_config_module.get_config()
    tyre_params_base.setdefault('traction_ellipse', 'current')
    params['tyre_front'] = copy.deepcopy(tyre_params_base)
    params['tyre_rear'] = copy.deepcopy(tyre_params_base)

    # Merge in config-driven params (e.g. tyre_temperature)
    params.update(params_from_config)

    
    # Driver / safety tuning params
    params['driver'] = {
    'tc_throttle_sr_threshold': 0.20,
    'tc_brake_sr_threshold': 0.20,
    'tc_gain': 5,
    'tc_min_throttle': 0.0,
    'tc_min_brake': -0.1,
    'target_speed_safety_factor': 1.01,  # 10% safety margin globally for stable driving
    'target_speed_smoothing_tau': 0.05,
    'speed_profile_stability_margin': 0.95,  # Use full TC capability
    'lap_boundary_margin': 0.5,  # 25% slower through closing segments
    'lap_boundary_zone_fraction': 0.20,  # Apply margin over last 30% of lap to cover all closing segments (26.4%)
    # Steering Stability Options
    'use_yaw_damping': True,
    'yaw_damping_gain': 0.12
    }

    
    # Calculate and add preload to params
    preload_f, preload_r = calculate_initial_preload(params)
    params['vehicle']['preload_f'] = preload_f
    params['vehicle']['preload_r'] = preload_r

    calculate_and_print_suspension_analytics(params)

    if mode == 'track':
        print("Generating optimal speed profile dynamically from vehicle physics...")
        path = generate_dynamic_speed_profile(path, params)
    else:
        exc_cfg = sim_config.get('excitation', {})
        path['target_speed'] = np.full(len(path['s']), exc_cfg.get('initial_speed_mps', 15.0))
        params['excitation'] = exc_cfg
    params['path'] = path

    initial_vx = (sim_config.get('excitation', {}).get('initial_speed_mps', 15.0) if mode == 'excitation' else 10)  # ms^-1
    
    ride_height_f = p_v['ride_height_f']
    ride_height_r = p_v['ride_height_r']
    g = params['environment']['g']
    lf, lr = p_v['lf'], p_v['lr']
    track_f = p_v['track_width_f']
    track_r = p_v['track_width_r']

    z_track_start = path['z_interp'](path['s'][0])
    
    initial_phi = 0.0 # Vehicle starts level
    initial_theta = np.arctan((ride_height_f - ride_height_r) / p_v['wheelbase'])

    avg_ride_height_at_cog = (ride_height_f * lr + ride_height_r * lf) / p_v['wheelbase']
    initial_Z_m = z_track_start + avg_ride_height_at_cog + p_v['COG_z']

    # Note: Using a single initial_phi as both front and rear start at the same angle
    initial_chassis_corner_Z = np.array([
        ride_height_f + lf * np.sin(initial_theta) - (track_f/2) * np.sin(initial_phi), # FL
        ride_height_f + lf * np.sin(initial_theta) + (track_f/2) * np.sin(initial_phi), # FR
        ride_height_r - lr * np.sin(initial_theta) - (track_r/2) * np.sin(initial_phi), # RL
        ride_height_r - lr * np.sin(initial_theta) + (track_r/2) * np.sin(initial_phi)  # RR
    ])

    # Set initial wheel height based on the static suspension length
    # To start with zero shock travel, the initial distance between the chassis and wheel
    # must be equal to our defined static length.
    initial_wheel_Z = np.array([
        initial_chassis_corner_Z[0] + p_v['suspension_static_length_f_m'], # FL
        initial_chassis_corner_Z[1] + p_v['suspension_static_length_f_m'], # FR
        initial_chassis_corner_Z[2] + p_v['suspension_static_length_r_m'], # RL
        initial_chassis_corner_Z[3] + p_v['suspension_static_length_r_m'], # RR
    ])
    
    # Calculate static tire compression ("squash")
    Fz_static_f = (p_v['mass'] * g * lr / p_v['wheelbase']) / 2.0
    Fz_static_r = (p_v['mass'] * g * lf / p_v['wheelbase']) / 2.0
    p_v['static_tire_squash_f'] = Fz_static_f / p_v['K_tire']
    p_v['static_tire_squash_r'] = Fz_static_r / p_v['K_tire']
    static_tire_squash = np.array([p_v['static_tire_squash_f'], p_v['static_tire_squash_f'], 
                                   p_v['static_tire_squash_r'], p_v['static_tire_squash_r']])
    
    initial_wheel_Z = initial_wheel_Z - static_tire_squash
    
    p_v['suspension_static_length_f_m'] = p_v['suspension_static_length_f_m'] - p_v['static_tire_squash_f']
    p_v['suspension_static_length_r_m'] = p_v['suspension_static_length_r_m'] - p_v['static_tire_squash_r']
    
    initial_conditions_dict = {
        'X_m': path['x'][0], 'Y_m': path['y'][0], 'psi_rad': path['heading'][0],
        'vx_mps': initial_vx, 'wheel_w_radps': np.full(4, initial_vx / p_v['R_eff']),
        'Z_m': initial_Z_m,
        'phi_f_rad': initial_phi, # Front roll
        'phi_r_rad': initial_phi, # Rear roll
        'theta_rad': initial_theta,
        'wheel_Z_m': initial_wheel_Z,
        'wheel_Z_dot_mps': np.zeros(4)
    }

    initial_conditions = sm.get_initial_state_vector(initial_conditions_dict)

    state_vector = initial_conditions
    if mode == 'excitation':
        controller = ExcitationController(params)
    else:
        controller = DriverModel(params)
    
    dt_control = 1.0 / CONTROL_HZ
    params['dt_log'] = 1.0 / LOG_HZ;
    max_time_per_lap = 120
    if mode == 'excitation':
        exc_phases = sim_config.get('excitation', {}).get('phases', [])
        exc_repeats = int(sim_config.get('excitation', {}).get('repeat_phases', 1))
        max_time = sum(p['duration_s'] for p in exc_phases) * exc_repeats
    else:
        max_time = max_time_per_lap * num_laps  # Scale total time by number of laps
    DEBUG_INTERVAL = 0.1 # Slower debug prints

    num_steps = int(max_time / params['dt_log']) + 2
    history = np.zeros((num_steps, sm.STATE_VECTOR_SIZE))
    control_history = np.zeros((num_steps, 3)) 
    force_history = np.zeros((num_steps, 4))
    tire_history = np.zeros((num_steps, 12)) # Fz, SA, SR for 4 wheels
    
    # Increased size to log heave spring forces
    suspension_history = np.zeros((num_steps, 18))


    tire_force_history = np.zeros((num_steps, 8))
    camber_history = np.zeros((num_steps, 4))
    # History for jacking forces
    jacking_force_history = np.zeros((num_steps, 4))
    
    # NEW: History for differential torques
    diff_history = np.zeros((num_steps, 4))
    
    # NEW: History for tyre temperatures
    temperature_history = np.zeros((num_steps, 4))  # T_FL, T_FR, T_RL, T_RR
    bulk_temperature_history = np.zeros((num_steps, 4))  # Tb_FL, Tb_FR, Tb_RL, Tb_RR
    
    # NEW: History for r_dot (yaw acceleration)
    r_dot_history = np.zeros(num_steps)  # r_dot_radps

    history[0, :] = state_vector

    # Live debug plot (optional)
    live_plot = LiveDebugPlot(params, enabled=live_debug_plot_enabled)
    if live_debug_plot_enabled:
        import matplotlib
        print("Live debug plot enabled (updates every {:.1f} s). Backend: {}.".format(LIVE_UPDATE_INTERVAL_SEC, matplotlib.get_backend()))
        if matplotlib.get_backend().lower() == 'agg':
            print("  -> Backend is Agg (no window). Run from a terminal with --live-debug so MPLBACKEND can be set to macosx/TkAgg.")

    # Lap tracking variables
    current_lap = 1
    lap_times = []
    lap_tire_data = []  # Store tire temps/pressures at end of each lap
    in_finish_zone = False  # Debounce flag to prevent multiple lap detections

    sim_time, log_index = 0.0, 1
    next_control_time, last_debug_print_time = 0.0, -1.0
    last_live_plot_time = -1.0
    
    controls = {'throttle_brake_cmd': 0.0, 'steer_cmd_rad': 0.0, 'target_v_mps': 0.0}
    debug_info = {}
    
    print("Starting simulation with dynamically generated speed profile...")
    if mode == 'excitation':
        print("Excitation mode: open-loop steering and throttle/brake from config.")
    while sim_time < max_time and log_index < num_steps:
        current_state_dict = sm.unpack_state_vector(state_vector)

        if sim_time >= next_control_time:
            if mode == 'excitation':
                controls = controller.get_controls(current_state_dict, dt_control, sim_time)
            else:
                controls = controller.get_controls(current_state_dict, dt_control)
            next_control_time += dt_control
        
        if sim_time >= last_debug_print_time + DEBUG_INTERVAL:
            if params.get('tyre_temperature'):
                T_FL, T_FR, T_RL, T_RR, Tb_FL, Tb_FR, Tb_RL, Tb_RR = get_tyre_temperatures(sim_time, current_state_dict, params['dt_log'], controls, params)
                forces, _, _, slip_angles, slip_ratios, _, _, _, _, jacking_forces, _, r_dot, T_FL, T_FR, T_RL, T_RR = get_vehicle_forces( #type: ignore
                    current_state_dict, controls, params, path, sim_time=sim_time, T_FL=T_FL, T_FR=T_FR, T_RL=T_RL, T_RR=T_RR)
            else:
                forces, _, _, slip_angles, slip_ratios, _, _, _, _, jacking_forces, _, r_dot, T_FL, T_FR, T_RL, T_RR = get_vehicle_forces( #type: ignore
                    current_state_dict, controls, params, path, sim_time=sim_time)
                Tb_FL, Tb_FR, Tb_RL, Tb_RR = T_FL, T_FR, T_RL, T_RR
            debug_info.update({
                'Sum_Fx': forces['Fx'],
                'Sum_Fy': forces['Fy'],
                'slip_ratios': slip_ratios,
                'slip_angles': slip_angles
            })
            print_debug_stats(sim_time, current_state_dict, controls, debug_info)
            last_debug_print_time = sim_time
            
        t_span = [sim_time, sim_time + params['dt_log']]
        result = solve_ivp(
            fun=calculate_derivatives_for_solver,
            t_span=t_span, y0=state_vector, method='LSODA', 
            args=(params, controls, path),
            dense_output=True, max_step=0.001
        )

        if result.status != 0:
            print(f"Solver failed at t={sim_time:.2f} with message: {result.message}")
            break

        sim_time = result.t[-1]
        state_vector = result.y[:, -1]
        
        final_state_dict = sm.unpack_state_vector(state_vector)
        if params.get('tyre_temperature'):
            T_FL, T_FR, T_RL, T_RR, Tb_FL, Tb_FR, Tb_RL, Tb_RR = get_tyre_temperatures(sim_time, final_state_dict, params['dt_log'], controls, params)
            final_forces, fz_tires, _, slip_angles, slip_ratios, susp_details, fx_tires, fy_tires, camber_angles, jacking_forces, diff_details, r_dot, T_FL, T_FR, T_RL, T_RR = get_vehicle_forces(
                final_state_dict, controls, params, path, sim_time=sim_time, T_FL=T_FL, T_FR=T_FR, T_RL=T_RL, T_RR=T_RR)
        else:
            final_forces, fz_tires, _, slip_angles, slip_ratios, susp_details, fx_tires, fy_tires, camber_angles, jacking_forces, diff_details, r_dot, T_FL, T_FR, T_RL, T_RR = get_vehicle_forces(
                final_state_dict, controls, params, path, sim_time=sim_time)
            Tb_FL, Tb_FR, Tb_RL, Tb_RR = T_FL, T_FR, T_RL, T_RR
        
        history[log_index, :] = state_vector
        control_history[log_index-1, :] = [controls['throttle_brake_cmd'], controls['steer_cmd_rad'], controls['target_v_mps']]
        force_history[log_index-1, :] = [final_forces['Fx'], final_forces['Fy'], np.sum(fz_tires), final_forces['Mz']]
        tire_history[log_index-1, :] = np.concatenate([fz_tires, slip_angles, slip_ratios])
        
        # Log heave spring forces
        suspension_history[log_index-1, :] = np.concatenate([
            susp_details['spring'], susp_details['damper'], susp_details['arb'], susp_details['heave'], susp_details['bump']
        ])
        tire_force_history[log_index-1, :] = np.concatenate([fx_tires, fy_tires])
        camber_history[log_index-1, :] = camber_angles
        # Log jacking forces
        jacking_force_history[log_index-1, :] = jacking_forces


        # Log differential details
        diff_history[log_index-1, :] = diff_details
        
        # Log tyre temperatures
        temperature_history[log_index-1, :] = [T_FL, T_FR, T_RL, T_RR]
        bulk_temperature_history[log_index-1, :] = [Tb_FL, Tb_FR, Tb_RL, Tb_RR]
        
        # Log r_dot (yaw acceleration)
        r_dot_history[log_index-1] = r_dot

        # Live debug plot update
        if live_debug_plot_enabled and (sim_time - last_live_plot_time >= LIVE_UPDATE_INTERVAL_SEC or last_live_plot_time < 0):
            last_live_plot_time = sim_time
            live_plot.update(
                sim_time=sim_time,
                log_index=log_index,
                state_dict=final_state_dict,
                history=history,
                tire_history=tire_history,
                tire_force_history=tire_force_history,
                temperature_history=temperature_history,
                bulk_temperature_history=bulk_temperature_history,
                total_sim_time=max_time,
                controls=controls,
            )

        log_index += 1
        
        cte, path_idx = find_closest_point_on_path(final_state_dict['X_m'], final_state_dict['Y_m'], path)
        vx_final = np.ravel(final_state_dict['vx_mps'])[0]
        sideslip_angle_rad = np.arctan2(np.ravel(final_state_dict['vy_mps'])[0], vx_final + 1e-6)
        
        SPIN_LIMIT_RAD, STALL_SPEED_MPS = np.deg2rad(45), 1.0

        if mode == 'track' and abs(cte) > TRACK_WIDTH_M:
            print(f"Car went off track at t={sim_time:.2f}s. Halting.")
            break
        if mode == 'track' and abs(sideslip_angle_rad) > SPIN_LIMIT_RAD:
            print(f"Car spun out at t={sim_time:.2f}s. Halting.")
            break
        if mode == 'track' and sim_time > 10 and abs(vx_final) < STALL_SPEED_MPS:
            print(f"Car stalled at t={sim_time:.2f}s. Halting.")
            break
        # Multi-lap completion logic with debounce
        # Check if we're currently in the finish zone
        # Calculate distance to actual start position (not origin)
        dist_to_start = np.sqrt((final_state_dict['X_m'] - path['x'][0])**2 +
                                (final_state_dict['Y_m'] - path['y'][0])**2)
        is_in_finish_zone = (mode == 'track' and
                             path_idx > len(path['x']) * 0.98 and
                             log_index > 100 and
                             dist_to_start < 10.0)

        # Only count a lap if we just entered the finish zone (wasn't there before)
        if is_in_finish_zone and not in_finish_zone:
            # Record lap time (time for this lap only)
            lap_time = sim_time - sum(lap_times)
            lap_times.append(lap_time)

            # Store tire data at end of lap
            tire_data = {
                'lap': current_lap,
                'time': sim_time,
                'temperatures': [T_FL, T_FR, T_RL, T_RR],
                'vertical_loads': fz_tires.tolist()
            }
            lap_tire_data.append(tire_data)

            print(f"Lap {current_lap} completed in {lap_time:.2f}s (total time: {sim_time:.2f}s)")
            print(f"  Tire temps: FL={T_FL:.1f}°C, FR={T_FR:.1f}°C, RL={T_RL:.1f}°C, RR={T_RR:.1f}°C")

            current_lap += 1
            if current_lap > num_laps:
                print(f"\nAll {num_laps} laps completed!")
                break
            # Otherwise continue to next lap

        # Update the finish zone flag for next iteration
        in_finish_zone = is_in_finish_zone
    
    print("\n" + "="*60)
    print("SIMULATION FINISHED")
    print("="*60)

    # Print lap summary
    if len(lap_times) > 0:
        print(f"\nCompleted {len(lap_times)} of {num_laps} laps:")
        print("-" * 60)
        for i, lap_time in enumerate(lap_times, 1):
            print(f"  Lap {i}: {lap_time:.2f}s")
        print(f"  Total time: {sum(lap_times):.2f}s")
        if len(lap_times) > 1:
            print(f"  Average lap time: {np.mean(lap_times):.2f}s")
            print(f"  Best lap: {min(lap_times):.2f}s")

        # Print tire degradation summary
        if len(lap_tire_data) > 0:
            print("\nTire Temperature Progression:")
            print("-" * 60)
            print(f"{'Lap':<6} {'FL (°C)':<10} {'FR (°C)':<10} {'RL (°C)':<10} {'RR (°C)':<10} {'Avg (°C)':<10}")
            for data in lap_tire_data:
                temps = data['temperatures']
                avg_temp = np.mean(temps)
                print(f"{data['lap']:<6} {temps[0]:<10.1f} {temps[1]:<10.1f} {temps[2]:<10.1f} {temps[3]:<10.1f} {avg_temp:<10.1f}")
    else:
        print(f"\nNo laps completed. Simulation ended at {sim_time:.2f}s")

    print("="*60 + "\n")

    # Close live debug plot if it was open
    live_plot.close()

    # Trim arrays to actual simulation length
    history = history[:log_index, :]
    control_history = control_history[:log_index, :]
    force_history = force_history[:log_index, :]
    tire_history = tire_history[:log_index, :]
    suspension_history = suspension_history[:log_index, :]
    tire_force_history = tire_force_history[:log_index, :]
    camber_history = camber_history[:log_index, :]
    jacking_force_history = jacking_force_history[:log_index, :]
    diff_history = diff_history[:log_index, :]
    temperature_history = temperature_history[:log_index, :]
    bulk_temperature_history = bulk_temperature_history[:log_index, :]
    r_dot_history = r_dot_history[:log_index]
    
    # Create timestamped results directory
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(__file__).resolve().parent.parent / 'results' / 'simulation' / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)

    if len(history) > 10:
        plot_simulation_results(
            history, control_history, force_history, tire_history,
            suspension_history, tire_force_history, camber_history,
            jacking_force_history, diff_history, path, params,
            temperature_history=temperature_history,
            r_dot_history=r_dot_history,
            show_track_path=(mode == 'track'),
            output_dir=results_dir,
        )

    print(f"Results saved to: {results_dir}")

    # Create comprehensive sim_full dictionary with all simulation data
    sim_full = {
        'history': history,
        'control_history': control_history,
        'force_history': force_history,
        'tire_history': tire_history,
        'suspension_history': suspension_history,
        'tire_force_history': tire_force_history,
        'camber_history': camber_history,
        'jacking_force_history': jacking_force_history,
        'diff_history': diff_history,
        'temperature_history': temperature_history,
        'bulk_temperature_history': bulk_temperature_history,
        'r_dot_history': r_dot_history,
        'path': path,
        'params': params,
        'lap_times': lap_times,
        'lap_tire_data': lap_tire_data,
        'num_laps': num_laps,
        'sim_time_final': sim_time,
        'dt_log': params['dt_log']
    }

    return sim_full, path, history, control_history, force_history, params

if __name__ == '__main__':
    import argparse
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description='Run vehicle simulation (track or excitation mode).')
    parser.add_argument('--config', type=str, default=None,
                        help=f'Path to config YAML (default from DEFAULT_CONFIG in script: {DEFAULT_CONFIG})')
    parser.add_argument('--excitation', action='store_true',
                        help='Use simulation_config_excitation.yaml (overrides DEFAULT_CONFIG)')
    parser.add_argument('--live-debug', action='store_true',
                        help='Enable live debug plot (car position, velocity, Pacejka curves, traction ellipses)')
    args = parser.parse_args()
    if args.excitation:
        config_path = script_dir / "configs" / "simulation_config_excitation.yaml"
    else:
        config_path = args.config  # None → main() uses DEFAULT_CONFIG
    sim_full, path, history, control_history, force_history, params = main(
        config_path=config_path,
        live_debug_plot=args.live_debug if args.live_debug else None
    )
