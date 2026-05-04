#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Acceleration Event — TC Strategy Comparison

Simulates a Formula Student car performing a straight-line acceleration event
(default 75m from standstill) with different traction control strategies to
demonstrate the effect of knowing accurate tyre parameters.

TC Strategies:
  1. Fixed target slip ratio
  2. Pacejka-based optimal slip (D and stiffness params too HIGH)
  3. Pacejka-based optimal slip (D and stiffness params too LOW)

The plant (vehicle dynamics) always uses the TRUE tyre parameters.
Each TC strategy uses its own parameter set to determine the target slip ratio.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
import copy
import sys
import time as time_module
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

# PATH SETUP
script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir.parent))
sys.path.insert(0, str(script_dir))
master_est_path = script_dir.parent / 'shared'
sys.path.insert(0, str(master_est_path))

from helpers.track_tools import generate_parametric_track
from helpers.controls import PIDController
from vehicle.vehicle_dynamics import calculate_derivatives_for_solver, get_vehicle_forces
from vehicle import state_manager as sm
from run_simulation import get_tyre_temperatures, calculate_initial_preload

# Import TTC Pacejka by file path to avoid namespace clash with estimation/core
import importlib.util as _ilu
_ttc_path = (script_dir.parent / "estimator" / "core" /
             "tyre_models" / "ttc_pacejka.py")
_spec = _ilu.spec_from_file_location("ttc_pacejka", _ttc_path)
ttc_model = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(ttc_model)


# USER CONFIGURATION — edit these to change the experiment

EVENT_DISTANCE_M = 75.0
INITIAL_VX_MPS = 0.5
CONTROL_HZ = 200
LOG_HZ = 200
MAX_TIME_S = 15.0

# Slip control mode
USE_PID = False          # True  → PID modulates throttle to track target κ
                        # False → directly command rear wheel speed for target κ

# TC Strategy 1: Fixed slip ratio
FIXED_TARGET_SR = 0.07

# TC Strategy 2: Pacejka with perturbations HIGH
# Keys can be any of the scalar tyre parameters below (multiplicative factor; 1.0 = no change).
#
# Longitudinal (ttc_config_paper.py):  D_ref_x, C_ref_x, E_x, C_x, k_load_x, k_load_stiff_x,
# T_opt_x, T_disp_x, T_stiff0_x, T_stiff1_x, T_stiff2_x, p_load0_x, p_load1_x, p_load2_x,
# p_stiff0_x, p_stiff1_x, p_stiff2_x, bias_x, Fz0 (if present).
# Lateral:  D_ref, C_ref, E, C, k_load, k_load_stiff, T_opt, T_disp, T_stiff0, T_stiff1, T_stiff2,
# p_load0, p_load1, p_load2, p_stiff0, p_stiff1, p_stiff2, bias, Fz0.
# Other:  default_pressure_kPa.
#
PERTURB_HIGH = {
    'D_ref_x': 1,
    'C_ref_x': 1.30,
    'E_x': 1,
    'k_load_x': 1,
    'k_load_stiff_x': 0.9,
    'T_opt_x': 1,
}

# TC Strategy 3: Pacejka with perturbations LOW
PERTURB_LOW = {
    'D_ref_x': 1,
    'C_ref_x': 0.7,
    'E_x': 1,
    'k_load_x': 1,
    'k_load_stiff_x': 1.1,
    'T_opt_x': 1,
}

# PID gains for slip-ratio control → throttle (only used when USE_PID=True)
TC_PID_KP = 15.0
TC_PID_KI = 8.0
TC_PID_KD = 0.3

# Legend labels in all plots; summary.txt contains a table mapping these to full parameters
SET_LABELS = ['A', 'B', 'C', 'D']


# TRACTION CONTROLLER CLASSES

class TractionController:
    """Base TC.  Subclasses provide get_target_slip_ratio().
    Two control modes (selected by USE_PID flag):
      PID mode  – PID modulates throttle to track the target κ.
      Direct    – rear wheel speeds are overwritten after each integration
                  step so that the target κ is achieved exactly."""

    def __init__(self, R_eff, name="TC"):
        self.R_eff = R_eff
        self.name = name
        if USE_PID:
            self.pid = PIDController(
                Kp=TC_PID_KP, Ki=TC_PID_KI, Kd=TC_PID_KD,
                min_output=0.0, max_output=1.0,
            )
        self.target_sr_history: list = []

    def get_target_slip_ratio(self, Fz_rear, T_rear):
        raise NotImplementedError

    def compute_control(self, state_dict, Fz_rear, T_rear, dt):
        """Return (throttle, target_kappa).
        In PID mode throttle comes from the PID.
        In direct mode throttle is 1.0 (full); the sim loop handles
        overwriting wheel speeds."""
        kappa_target = self.get_target_slip_ratio(Fz_rear, T_rear)
        self.target_sr_history.append(kappa_target)

        if USE_PID:
            vx = max(float(np.ravel(state_dict['vx_mps'])[0]), 0.1)
            omega_rear = np.ravel(state_dict['wheel_w_radps'])[2:4]
            kappa_actual = float(np.mean((omega_rear * self.R_eff) / vx - 1.0))
            throttle = self.pid.update(kappa_target - kappa_actual, dt)
        else:
            throttle = 1.0

        return throttle, kappa_target


class FixedSlipTC(TractionController):
    """TC with a constant target slip ratio."""

    def __init__(self, target_sr, R_eff, name=None):
        super().__init__(R_eff, name=name or f"Fixed SR (κ={target_sr:.2f})")
        self.target_sr = target_sr

    def get_target_slip_ratio(self, Fz_rear, T_rear):
        return self.target_sr


class PacejkaSlipTC(TractionController):
    """TC that finds the slip ratio maximising Fx from a (possibly wrong)
    Pacejka model at the current rear Fz and tyre temperature."""

    def __init__(self, tyre_params, R_eff, name="Pacejka TC"):
        super().__init__(R_eff, name=name)
        self.tyre_params = tyre_params
        self._cache: dict = {}

    def get_target_slip_ratio(self, Fz_rear, T_rear):
        avg_Fz = float(np.mean(Fz_rear))
        avg_T = float(np.mean(T_rear))
        cache_key = (round(avg_Fz / 10) * 10, round(avg_T))
        if cache_key in self._cache:
            return self._cache[cache_key]

        kappa_peak = _find_peak_slip_ratio(self.tyre_params, avg_Fz, avg_T)
        self._cache[cache_key] = kappa_peak
        return kappa_peak


# HELPERS

def _find_peak_slip_ratio(tyre_params, Fz, T):
    """Numerically find the slip ratio that maximises |Fx|."""
    P = tyre_params.get('default_pressure_kPa', 120.0)

    def neg_Fx(sr):
        fx = ttc_model.longitudinal_force_model(
            sr, Fz, T, P, [], fixed_params=tyre_params)
        return -abs(float(np.ravel(fx)[0]))

    res = minimize_scalar(neg_Fx, bounds=(0.01, 0.5), method='bounded')
    return float(res.x)


def _make_perturbed_params(base, perturbations):
    """Return a deep copy of *base* with selected keys scaled."""
    p = copy.deepcopy(base)
    for key, factor in perturbations.items():
        if key in p:
            p[key] = p[key] * factor
    return p


def _clamp_slip_to_motor(target_sr, vx, true_tyre_params, Fz_rear, T_rear, pv):
    """Reduce target slip ratio so the drivetrain can sustain the tyre reaction.

    At high speed the motor is power-limited: T_avail = P_max / ω.
    If the tyre Fx at *target_sr* produces a reaction torque (Fx·R) that
    exceeds the available drive torque, bisect to find the slip ratio on the
    ascending side of the Pacejka curve where tyre reaction ≈ motor torque.
    """
    R_eff = pv['R_eff']
    avg_Fz = float(np.mean(Fz_rear))
    avg_T = float(np.mean(T_rear))
    P_kPa = true_tyre_params.get('default_pressure_kPa', 120.0)

    def _motor_torque(sr):
        omega_wheel = (1.0 + sr) * vx / R_eff
        motor_rpm = omega_wheel * pv['gear_ratio'] * 60.0 / (2.0 * np.pi)
        T_motor = float(np.interp(motor_rpm, pv['motor_torque_curve'][0],
                                   pv['motor_torque_curve'][1]))
        T_wheel = T_motor * pv['gear_ratio']
        max_pw = pv.get('max_power_W')
        if max_pw is not None and omega_wheel > 1e-6 and T_wheel * omega_wheel > max_pw:
            T_wheel = max_pw / omega_wheel
        return T_wheel

    def _tyre_reaction(sr):
        Fx = abs(float(ttc_model.longitudinal_force_model(
            sr, avg_Fz, avg_T, P_kPa, [], fixed_params=true_tyre_params)))
        return 2.0 * Fx * R_eff

    if _tyre_reaction(target_sr) <= _motor_torque(target_sr):
        return target_sr

    lo, hi = 0.0, target_sr
    for _ in range(30):
        mid = (lo + hi) * 0.5
        if _tyre_reaction(mid) < _motor_torque(mid):
            lo = mid
        else:
            hi = mid
    return (lo + hi) * 0.5


# DATA CONTAINER

@dataclass
class RunResult:
    name: str
    time: np.ndarray
    distance: np.ndarray
    vx: np.ndarray
    slip_ratio_actual: np.ndarray
    slip_ratio_target: np.ndarray
    slip_ratio_true_optimal: np.ndarray
    fx_rear_total: np.ndarray
    fz_rear_avg: np.ndarray
    throttle: np.ndarray
    temperature_rear: np.ndarray
    power_motor_W: np.ndarray
    pitch_rad: np.ndarray
    total_Fx: np.ndarray          # Sum of all tyre Fx (vehicle frame) minus drag; ax = total_Fx/m
    finish_time: float
    final_speed_mps: float
    tc_tyre_params: Optional[dict] = None
    legend_label: Optional[str] = None  # Short label with parameter values for plots


# VEHICLE + PATH SETUP  (mirrors main_simulation FS config)

def _build_params_and_path():
    """Construct vehicle params, a straight path, and load the true tyre config."""

    path = generate_parametric_track(
        [{'type': 'straight', 'length': 150}], step=0.5)
    path['target_speed'] = np.full(len(path['s']), 200.0)

    params: dict = {}
    params['vehicle'] = {
        'mass': 320,
        'wheelbase': 1.6,
        'track_width_f': 1.25, 'track_width_r': 1.25,
        'COG_z': 0.2, 'COG_ratio_x': 0.5,
        'Ix': 60, 'Iy': 100, 'Iz': 200,
        'm_us_f': 9, 'm_us_r': 12,
        'K_s_f': 20000, 'K_s_r': 30000,
        'C_s_f': 1500, 'C_s_r': 2500,
        'K_arb_f': 10000, 'K_arb_r': 0,
        'K_heave_f': 0, 'K_heave_r': 0,
        'C_heave_f': 0, 'C_heave_r': 0,
        'motion_ratio_f': 1, 'motion_ratio_r': 1,
        'motion_ratio_rate_f': 0, 'motion_ratio_rate_r': 0,
        'shock_stroke_max_m': 0.035,
        'shock_stroke_rebound_max_m': 0.025,
        'K_bump_stop': 1000000,
        'ride_height_f': 0.05, 'ride_height_r': 0.05,
        'K_torsion_Nm_rad': 800000,
        'RCH_f_m': 0.0, 'RCH_r_m': 0.0,
        'PCH_f_m': 0, 'PCH_r_m': 0,
        'caster_f_deg': 0.0, 'caster_r_deg': 0.0,
        'kpi_f_deg': 0.0, 'kpi_r_deg': 0.0,
        'scrub_radius_f': 0.0,
        'static_toe_f_deg': 0, 'static_toe_r_deg': 0,
        'bump_steer_f_deg_per_mm': 0, 'bump_steer_r_deg_per_mm': 0,
        'static_camber_f_deg': 0.0, 'static_camber_r_deg': 0.0,
        'camber_roll_gain_f': 0, 'camber_roll_gain_r': 0,
        'K_tire': 200000, 'R_eff': 0.22, 'I_wheel': 1.0,
        'brake_bias': 0.63, 'max_brake_torque': 2000,
        'motor_torque_curve': ([0, 3500, 4000, 5000, 6000, 7000, 8000],
                               [340, 340, 305, 230, 180, 150, 130]),
        'gear_ratio': 3.5,
        'max_power_W': 80000,   # Max drive power; torque scaled so power ≤ this
        'diff_lock_torque_accel': 400,
        'diff_lock_torque_coast': 100,
        'diff_preload': 50,
        'ClA': 0, 'CdA': 0.0, 'COP_z': 0.2, 'COP_x_ratio': 0.5,
        'ackerman_percentage': 0,
        'yaw_damping_coeff': 0.0,
        'max_speed_mps': 120,
    }

    pv = params['vehicle']
    pv['lf'] = pv['wheelbase'] * (1 - pv['COG_ratio_x'])
    pv['lr'] = pv['wheelbase'] * pv['COG_ratio_x']
    pv['m_s'] = pv['mass'] - 2 * (pv['m_us_f'] + pv['m_us_r'])
    pv['suspension_static_length_f_m'] = pv['R_eff'] - pv['ride_height_f']
    pv['suspension_static_length_r_m'] = pv['R_eff'] - pv['ride_height_r']

    params['environment'] = {'g': 9.81, 'rho_air': 1.225}

    # True tyre config (TTC paper parameters)
    import importlib.util
    cfg_path = (script_dir.parent / "shared" /
                "tyre_configs" / "ttc_config_paper.py")
    spec = importlib.util.spec_from_file_location("tyre_config", cfg_path)
    tyre_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tyre_mod)
    tyre_base = tyre_mod.get_config()
    tyre_base.setdefault('traction_ellipse', 'current')

    params['tyre_front'] = copy.deepcopy(tyre_base)
    params['tyre_rear'] = copy.deepcopy(tyre_base)

    # Lumped tyre-temperature model (pre-warmed, matching main sim)
    params['tyre_temperature'] = {
        'mode': 'lumped',
        'lumped_T_initial': 60.0,
        'lumped_T_air': 25.0,
        'lumped_T_ground': 30.0,
        'lumped_h_alpha': 0.03493,
        'lumped_h_kappa': 0.0151,
        'lumped_k_bt': 0.423864,
        'lumped_k_ta': 0.037442,
        'lumped_k_tg': 0.001,
        'lumped_c_tread': 1.0,
        'lumped_c_bulk': 9.278181,
    }

    params['driver'] = {}

    # Preloads
    preload_f, preload_r = calculate_initial_preload(params)
    pv['preload_f'] = preload_f
    pv['preload_r'] = preload_r

    # Static tyre squash (same procedure as main_simulation)
    g = params['environment']['g']
    Fz_static_f = (pv['mass'] * g * pv['lr'] / pv['wheelbase']) / 2.0
    Fz_static_r = (pv['mass'] * g * pv['lf'] / pv['wheelbase']) / 2.0
    pv['static_tire_squash_f'] = Fz_static_f / pv['K_tire']
    pv['static_tire_squash_r'] = Fz_static_r / pv['K_tire']
    pv['suspension_static_length_f_m'] -= pv['static_tire_squash_f']
    pv['suspension_static_length_r_m'] -= pv['static_tire_squash_r']

    params['path'] = path
    params['dt_log'] = 1.0 / LOG_HZ

    return params, path, tyre_base


def _create_initial_state(params, path):
    """Build the state vector for a near-standstill launch."""
    pv = params['vehicle']
    z0 = path['z_interp'](path['s'][0])
    theta0 = np.arctan(
        (pv['ride_height_f'] - pv['ride_height_r']) / pv['wheelbase'])
    lf, lr = pv['lf'], pv['lr']
    avg_rh = (pv['ride_height_f'] * lr + pv['ride_height_r'] * lf) / pv['wheelbase']
    Z0 = z0 + avg_rh + pv['COG_z']
    phi0 = 0.0
    tf, tr = pv['track_width_f'], pv['track_width_r']

    chassis_Z = np.array([
        pv['ride_height_f'] + lf * np.sin(theta0) - (tf / 2) * np.sin(phi0),
        pv['ride_height_f'] + lf * np.sin(theta0) + (tf / 2) * np.sin(phi0),
        pv['ride_height_r'] - lr * np.sin(theta0) - (tr / 2) * np.sin(phi0),
        pv['ride_height_r'] - lr * np.sin(theta0) + (tr / 2) * np.sin(phi0),
    ])

    # Wheel Z uses the *un-corrected* static length then subtracts squash
    # (mirrors main_simulation lines 499-514)
    uncorr_f = pv['suspension_static_length_f_m'] + pv['static_tire_squash_f']
    uncorr_r = pv['suspension_static_length_r_m'] + pv['static_tire_squash_r']
    wheel_Z = np.array([
        chassis_Z[0] + uncorr_f,
        chassis_Z[1] + uncorr_f,
        chassis_Z[2] + uncorr_r,
        chassis_Z[3] + uncorr_r,
    ])
    squash = np.array([
        pv['static_tire_squash_f'], pv['static_tire_squash_f'],
        pv['static_tire_squash_r'], pv['static_tire_squash_r'],
    ])
    wheel_Z -= squash

    ic = {
        'X_m': path['x'][0], 'Y_m': path['y'][0],
        'psi_rad': path['heading'][0],
        'vx_mps': INITIAL_VX_MPS,
        'wheel_w_radps': np.full(4, INITIAL_VX_MPS / pv['R_eff']),
        'Z_m': Z0,
        'phi_f_rad': phi0, 'phi_r_rad': phi0,
        'theta_rad': theta0,
        'wheel_Z_m': wheel_Z,
        'wheel_Z_dot_mps': np.zeros(4),
    }
    return sm.get_initial_state_vector(ic)


# SINGLE ACCELERATION RUN

def run_single_event(params_template, path, tc, true_tyre_params,
                     event_distance=EVENT_DISTANCE_M):
    """Simulate one acceleration event with the given TC and return a RunResult."""
    params = copy.deepcopy(params_template)
    state_vector = _create_initial_state(params, path)
    sd0 = sm.unpack_state_vector(state_vector)
    start_X = float(np.ravel(sd0['X_m'])[0])
    start_Y = float(np.ravel(sd0['Y_m'])[0])

    dt_control = 1.0 / CONTROL_HZ
    dt_log = 1.0 / LOG_HZ
    R_eff = params['vehicle']['R_eff']

    num_steps = int(MAX_TIME_S / dt_log) + 2
    log = {k: np.zeros(num_steps) for k in [
        'time', 'distance', 'vx', 'sr_actual', 'sr_target',
        'sr_true_opt', 'fx_rear', 'fz_rear', 'throttle', 'temp_rear',
        'power_motor', 'pitch_rad', 'total_Fx',
    ]}

    sim_time = 0.0
    idx = 0
    next_ctrl = 0.0
    controls = {'throttle_brake_cmd': 0.0, 'steer_cmd_rad': 0.0,
                'target_v_mps': 0.0}
    finish_time = MAX_TIME_S
    cur_target_sr = 0.0
    cur_true_opt = 0.0
    last_fx_rear_total = 0.0

    # Initialise Fz/T estimates for the first TC call
    pv = params['vehicle']
    Fz_static_rear = (pv['mass'] * 9.81 * pv['lf']
                      / pv['wheelbase'] / 2.0)
    last_fz_rear = np.array([Fz_static_rear, Fz_static_rear])
    T_init = params.get('tyre_temperature', {}).get('lumped_T_initial', 60.0)
    last_T_rear = np.array([T_init, T_init])

    # Pre-compute the state-vector slice for rear wheel speeds (indices 2,3)
    _ws = sm.STATE_VECTOR_LAYOUT['wheel_w_radps']
    rear_w_slice = slice(_ws['slice'].start + 2, _ws['slice'].start + 4)

    while sim_time < MAX_TIME_S and idx < num_steps:
        state_dict = sm.unpack_state_vector(state_vector)

        # Control update (CONTROL_HZ)
        if sim_time >= next_ctrl:
            throttle, cur_target_sr = tc.compute_control(
                state_dict, last_fz_rear, last_T_rear, dt_control)
            controls = {
                'throttle_brake_cmd': throttle,
                'steer_cmd_rad': 0.0,
                'target_v_mps': 0.0,
            }
            cur_true_opt = _find_peak_slip_ratio(
                true_tyre_params,
                float(np.mean(last_fz_rear)),
                float(np.mean(last_T_rear)),
            )
            next_ctrl += dt_control

        # In direct mode, set throttle to sustain the tyre reaction
        # Without this, the motor delivers max torque → excess spins up the
        # wheels during integration → slip drifts away from target → the Fx
        # that actually accelerates the car differs from the logged value.
        if not USE_PID and last_fx_rear_total > 0:
            avg_w = float(np.mean(np.ravel(state_dict['wheel_w_radps'])[2:4]))
            if avg_w > 1e-6:
                m_rpm = avg_w * pv['gear_ratio'] * 60.0 / (2.0 * np.pi)
                T_max_m = float(np.interp(m_rpm, pv['motor_torque_curve'][0],
                                          pv['motor_torque_curve'][1]))
                T_max_w = T_max_m * pv['gear_ratio']
                max_pw = pv.get('max_power_W')
                if max_pw and T_max_w * avg_w > max_pw:
                    T_max_w = max_pw / avg_w
                T_reaction = last_fx_rear_total * R_eff
                controls['throttle_brake_cmd'] = min(
                    max(T_reaction / max(T_max_w, 1.0), 0.0), 1.0)

        # Integrate one log step (LOG_HZ)
        result = solve_ivp(
            fun=calculate_derivatives_for_solver,
            t_span=[sim_time, sim_time + dt_log],
            y0=state_vector, method='LSODA',
            args=(params, controls, path),
            dense_output=True, max_step=0.001,
        )
        if result.status != 0:
            print(f"  Solver failed at t={sim_time:.3f}s: {result.message}")
            break

        sim_time = result.t[-1]
        state_vector = result.y[:, -1]

        # Direct slip command: overwrite rear wheel speeds
        # Clamp target slip so tyre reaction never exceeds drivetrain torque.
        if not USE_PID:
            vx_now = max(float(state_vector[sm.STATE_VECTOR_LAYOUT['vx_mps']['slice']][0]), 0.1)
            clamped_sr = _clamp_slip_to_motor(
                cur_target_sr, vx_now, true_tyre_params,
                last_fz_rear, last_T_rear, pv)
            omega_target = (1.0 + clamped_sr) * vx_now / R_eff
            state_vector[rear_w_slice] = omega_target

        # Post-step: update thermal model and read forces
        sd = sm.unpack_state_vector(state_vector)
        T_FL, T_FR, T_RL, T_RR, *_ = get_tyre_temperatures(
            sim_time, sd, dt_log, controls, params)

        frc = get_vehicle_forces(
            sd, controls, params, path,
            sim_time=sim_time, T_FL=T_FL, T_FR=T_FR, T_RL=T_RL, T_RR=T_RR)
        fz_tires = frc[1]    # (4,)
        fx_wheels = frc[6]   # (4,)
        forces_moments = frc[0]   # {'Fx': Sum_Fx_after_drag}; ax = Fx/m

        last_fz_rear = np.array([float(fz_tires[2]), float(fz_tires[3])])
        last_T_rear = np.array([T_RL, T_RR])

        vx = float(np.ravel(sd['vx_mps'])[0])
        X = float(np.ravel(sd['X_m'])[0])
        Y = float(np.ravel(sd['Y_m'])[0])
        omega_rear = np.ravel(sd['wheel_w_radps'])[2:4]
        vx_safe = max(abs(vx), 0.1)
        sr_actual = float(np.mean((omega_rear * R_eff) / vx_safe - 1.0))

        dist = np.sqrt((X - start_X)**2 + (Y - start_Y)**2)
        log['time'][idx] = sim_time
        log['distance'][idx] = dist
        log['vx'][idx] = vx
        log['sr_actual'][idx] = sr_actual
        log['sr_target'][idx] = cur_target_sr
        log['sr_true_opt'][idx] = cur_true_opt
        last_fx_rear_total = float(fx_wheels[2]) + float(fx_wheels[3])
        log['fx_rear'][idx] = last_fx_rear_total
        log['fz_rear'][idx] = float(np.mean(last_fz_rear))
        log['total_Fx'][idx] = float(forces_moments['Fx'])
        log['throttle'][idx] = controls['throttle_brake_cmd']
        log['temp_rear'][idx] = float(np.mean(last_T_rear))

        avg_rear_w = float(np.mean(omega_rear))
        fx_rear_total = float(fx_wheels[2]) + float(fx_wheels[3])
        if avg_rear_w > 1e-6 and fx_rear_total > 0:
            log['power_motor'][idx] = fx_rear_total * R_eff * avg_rear_w
        else:
            log['power_motor'][idx] = 0.0

        log['pitch_rad'][idx] = float(np.ravel(sd['theta_rad'])[0])

        idx += 1

        if dist >= event_distance:
            finish_time = sim_time
            break

    # Trim arrays
    for k in log:
        log[k] = log[k][:idx]

    return RunResult(
        name=tc.name,
        time=log['time'],
        distance=log['distance'],
        vx=log['vx'],
        slip_ratio_actual=log['sr_actual'],
        slip_ratio_target=log['sr_target'],
        slip_ratio_true_optimal=log['sr_true_opt'],
        fx_rear_total=log['fx_rear'],
        fz_rear_avg=log['fz_rear'],
        throttle=log['throttle'],
        temperature_rear=log['temp_rear'],
        power_motor_W=log['power_motor'],
        pitch_rad=log['pitch_rad'],
        total_Fx=log['total_Fx'],
        finish_time=finish_time,
        final_speed_mps=float(log['vx'][-1]) if idx > 0 else 0.0,
        tc_tyre_params=getattr(tc, 'tyre_params', None),
    )


# PLOTTING

COLORS = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']


def _plot_main_comparison(results: List[RunResult], save_dir: Path):
    """Top row only: speed vs time, speed vs distance, rear Fx vs time."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    def _label(r: RunResult):
        return r.legend_label if r.legend_label else r.name

    # 1. Speed vs time
    ax = axes[0]
    for i, r in enumerate(results):
        ax.plot(r.time, r.vx * 3.6, color=COLORS[i],
                label=f'{_label(r)} ({r.finish_time:.2f}s)')
        ax.axvline(r.finish_time, color=COLORS[i], ls='--', alpha=0.4)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Speed [km/h]')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. Speed vs distance
    ax = axes[1]
    for i, r in enumerate(results):
        ax.plot(r.distance, r.vx * 3.6, color=COLORS[i], label=_label(r))
    ax.axvline(EVENT_DISTANCE_M, color='k', ls=':', alpha=0.4,
               label=f'{EVENT_DISTANCE_M:.0f} m')
    ax.set_xlabel('Distance [m]')
    ax.set_ylabel('Speed [km/h]')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. Rear Fx vs time
    ax = axes[2]
    for i, r in enumerate(results):
        ax.plot(r.time, r.fx_rear_total, color=COLORS[i], label=_label(r))
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Rear Fx Total [N]')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_dir / 'acceleration_comparison.png',
                dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_dir / 'acceleration_comparison.png'}")
    return fig


def _plot_dynamics_overview(results: List[RunResult], save_dir: Path):
    """Five-panel figure: speed, motor power, total Fx (ax = Fx/m), pitch, rear tyre temp."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    def _label(r: RunResult):
        return r.legend_label if r.legend_label else r.name

    # 1. Vehicle speed
    ax = axes[0, 0]
    for i, r in enumerate(results):
        ax.plot(r.time, r.vx * 3.6, color=COLORS[i], label=_label(r))
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Speed [km/h]')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. Motor power (capped at max_power_W)
    ax = axes[0, 1]
    for i, r in enumerate(results):
        ax.plot(r.time, r.power_motor_W / 1000, color=COLORS[i], label=_label(r))
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Motor power [kW]')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # 3. Total Fx (ax = total_Fx/m)
    ax = axes[0, 2]
    for i, r in enumerate(results):
        ax.plot(r.time, r.total_Fx, color=COLORS[i], label=_label(r))
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Total Fx [N]')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', ls=':', alpha=0.5)

    # 4. Vehicle pitch
    ax = axes[1, 0]
    for i, r in enumerate(results):
        ax.plot(r.time, np.rad2deg(r.pitch_rad), color=COLORS[i], label=_label(r))
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Pitch [deg]')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 5. Rear tyre temperature
    ax = axes[1, 1]
    for i, r in enumerate(results):
        ax.plot(r.time, r.temperature_rear, color=COLORS[i], label=_label(r))
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Rear tyre temp [°C]')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 6. ax = total_Fx / m (sanity: should match dv/dt)
    ax = axes[1, 2]
    mass_kg = 320.0
    for i, r in enumerate(results):
        ax.plot(r.time, r.total_Fx / mass_kg, color=COLORS[i], label=_label(r))
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('ax (total_Fx/m) [m/s²]')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', ls=':', alpha=0.5)

    plt.tight_layout()
    fig.savefig(save_dir / 'dynamics_overview.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_dir / 'dynamics_overview.png'}")
    return fig


def _plot_slip_tracking(results: List[RunResult], save_dir: Path):
    """Per-TC-mode slip ratio tracking subplots."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4.5), sharey=True)
    if n == 1:
        axes = [axes]
    fig.suptitle('Slip Ratio Tracking Detail', fontsize=13, fontweight='bold')

    for i, (r, ax) in enumerate(zip(results, axes)):
        ax.plot(r.time, r.slip_ratio_actual * 100,
                color=COLORS[i], linewidth=1.5, label='Actual κ')
        ax.plot(r.time, r.slip_ratio_target * 100,
                color=COLORS[i], ls='--', alpha=0.7, label='Target κ')
        ax.plot(r.time, r.slip_ratio_true_optimal * 100,
                'k:', alpha=0.6, linewidth=1.2, label='True optimal κ')
        ax.fill_between(
            r.time,
            r.slip_ratio_target * 100,
            r.slip_ratio_actual * 100,
            color=COLORS[i], alpha=0.12,
        )
        ax.set_xlabel('Time [s]')
        if i == 0:
            ax.set_ylabel('Slip Ratio [%]')
        ax.set_title(r.legend_label if r.legend_label else r.name, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        max_target = max(float(np.max(r.slip_ratio_target)), 0.05) * 100
        ax.set_ylim(-2, max_target * 2.5)

    plt.tight_layout()
    fig.savefig(save_dir / 'slip_tracking_detail.png',
                dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_dir / 'slip_tracking_detail.png'}")
    return fig


def _plot_pacejka_curves(results: List[RunResult],
                         true_tyre_params: dict, save_dir: Path):
    """Pacejka Fx(κ) curves — true vs perturbed — with operating points."""
    repr_Fz = 900.0
    repr_T = 60.0
    P = true_tyre_params.get('default_pressure_kPa', 120.0)
    kappa = np.linspace(0, 0.4, 300)

    fig, ax = plt.subplots(figsize=(10, 6))

    # True curve
    Fx_true = np.array([float(ttc_model.longitudinal_force_model(
        k, repr_Fz, repr_T, P, [], fixed_params=true_tyre_params))
        for k in kappa])
    ax.plot(kappa * 100, Fx_true, 'k-', linewidth=2.5, label='True Pacejka')

    # True peak marker
    sr_peak = _find_peak_slip_ratio(true_tyre_params, repr_Fz, repr_T)
    Fx_peak = float(ttc_model.longitudinal_force_model(
        sr_peak, repr_Fz, repr_T, P, [], fixed_params=true_tyre_params))
    ax.plot(sr_peak * 100, Fx_peak, 'k*', markersize=15, zorder=6,
            label=f'True peak κ={sr_peak*100:.1f}%')

    # Perturbed curves + target markers at representative conditions
    for i, r in enumerate(results):
        if r.tc_tyre_params is not None:
            Fx_p = np.array([float(ttc_model.longitudinal_force_model(
                k, repr_Fz, repr_T, P, [], fixed_params=r.tc_tyre_params))
                for k in kappa])
            ax.plot(kappa * 100, Fx_p, color=COLORS[i], ls='--',
                    linewidth=1.5, label=f'{r.legend_label or r.name} curve')
            sr_target = _find_peak_slip_ratio(r.tc_tyre_params, repr_Fz, repr_T)
        else:
            sr_target = float(np.median(r.slip_ratio_target)) if len(r.slip_ratio_target) else None

        if sr_target is not None:
            Fx_at = float(ttc_model.longitudinal_force_model(
                sr_target, repr_Fz, repr_T, P, [], fixed_params=true_tyre_params))
            ax.axvline(sr_target * 100, color=COLORS[i], ls=':', alpha=0.5)
            ax.plot(sr_target * 100, Fx_at, 'o', color=COLORS[i],
                    markersize=10, zorder=5,
                    label=f'{r.legend_label or r.name} target κ={sr_target*100:.1f}%')

    ax.set_xlabel('Slip Ratio [%]', fontsize=11)
    ax.set_ylabel('Longitudinal Force Fx [N]', fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_dir / 'pacejka_curves.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_dir / 'pacejka_curves.png'}")
    return fig


# CONFIG + RESULTS OUTPUT

def _get_config_text():
    """Build a human-readable string of the full experiment configuration."""
    mode_str = "PID → throttle" if USE_PID else "Direct wheel-speed command"
    lines = [
        "=" * 72,
        "ACCELERATION EVENT CONFIGURATION",
        "=" * 72,
        "",
        "Event",
        f"  Distance:          {EVENT_DISTANCE_M} m",
        f"  Initial speed:     {INITIAL_VX_MPS} m/s",
        f"  Max sim time:      {MAX_TIME_S} s",
        f"  Control rate:      {CONTROL_HZ} Hz",
        f"  Log rate:          {LOG_HZ} Hz",
        "",
        f"Slip control mode:   {mode_str}",
    ]
    if USE_PID:
        lines += [
            "TC PID gains",
            f"  Kp = {TC_PID_KP},  Ki = {TC_PID_KI},  Kd = {TC_PID_KD}",
        ]
    lines += [
        "",
        "-" * 72,
        "PARAMETER SETS (for report table — plots use legend A, B, C, D)",
        "-" * 72,
        "",
        "Set A (Fixed slip ratio):",
        f"  Target slip ratio κ = {FIXED_TARGET_SR}",
        "  No Pacejka parameters; slip target is constant.",
        "",
        "Set B (Plant parameters):",
        "  Pacejka TC with unperturbed plant tyre parameters.",
        "  Base config: shared/tyre_configs/ttc_config_paper.py",
        "  Perturbation multipliers: none (all ×1).",
        "",
        "Set C (Perturbed — multipliers applied to base):",
    ]
    for k, v in sorted(PERTURB_HIGH.items()):
        lines.append(f"  {k}: ×{v}")
    lines += [
        "",
        "Set D (Perturbed — multipliers applied to base):",
    ]
    for k, v in sorted(PERTURB_LOW.items()):
        lines.append(f"  {k}: ×{v}")
    lines += ["", "=" * 72]
    return "\n".join(lines)


def _build_results_text(results: List[RunResult], max_power_W: Optional[float] = None):
    """Build a human-readable results string (printed and saved to file)."""
    lines = [
        "",
        "=" * 72,
        "ACCELERATION EVENT RESULTS",
        f"Event: {EVENT_DISTANCE_M:.1f} m from {INITIAL_VX_MPS:.1f} m/s standstill",
        f"Run:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 72,
    ]

    for r in results:
        avg_fx = float(np.mean(r.fx_rear_total))
        sr_target_med = float(np.median(r.slip_ratio_target))
        sr_actual_med = float(np.median(r.slip_ratio_actual))
        lines += [
            "",
            f"  {r.name}",
            f"    Time to {EVENT_DISTANCE_M:.0f}m:   {r.finish_time:.3f} s",
            f"    Final speed:     {r.final_speed_mps * 3.6:.1f} km/h",
            f"    Avg rear Fx:     {avg_fx:.0f} N",
            f"    Avg total Fx:    {float(np.mean(r.total_Fx)):.0f} N  (ax = total_Fx/m)",
            f"    Median target κ: {sr_target_med * 100:.2f}%",
            f"    Median actual κ: {sr_actual_med * 100:.2f}%",
        ]
        if max_power_W is not None and len(r.power_motor_W) > 0:
            at_limit = np.sum(r.power_motor_W >= 0.99 * max_power_W)
            pct = 100.0 * at_limit / len(r.power_motor_W)
            lines.append(f"    Fraction of run at power limit ({max_power_W/1000:.0f} kW): {pct:.1f}%")

    ranked = sorted(results, key=lambda r: r.finish_time)
    best = ranked[0].finish_time
    lines += ["", "-" * 72, "Ranking:"]
    for i, r in enumerate(ranked, 1):
        delta = r.finish_time - best
        lines.append(f"  {i}. {r.name}: {r.finish_time:.3f}s  (+{delta:.3f}s)")

    if max_power_W is not None:
        lines += [
            "",
            "Note: Similar acceleration times despite different rear Fx often occur because the",
            f"max power limit ({max_power_W/1000:.0f} kW) caps drive torque at higher speeds. The effective",
            "rear force the drivetrain can deliver is Fx_eff ≤ P_max/v, so above ~25–30 m/s all",
            "strategies are limited to similar force. The plotted rear Fx is the tyre-model output",
            "at commanded slip; when power-limited, the drivetrain cannot fully support that slip.",
        ]
    lines += [
        "",
        "Sanity: Longitudinal acceleration ax = total_Fx / m, where total_Fx = Sum(all tyre Fx in vehicle x) - drag.",
        "Rear Fx is only part of total_Fx; front Fx (free-rolling) is typically small. If total_Fx differs between",
        "runs, speeds must differ; if total_Fx is similar but rear Fx differs, check front Fx or drag.",
    ]
    lines += ["=" * 72, ""]
    return "\n".join(lines)


def _save_summary(config_text: str, results_text: str, save_dir: Path):
    """Write config + results to a single summary text file."""
    summary = config_text + "\n\n" + results_text
    out = save_dir / "summary.txt"
    out.write_text(summary)
    print(f"  Saved: {out}")


# MAIN

def main():
    np.seterr(all='warn')

    print("Building vehicle parameters and track …")
    params, path, true_tyre_params = _build_params_and_path()
    R_eff = params['vehicle']['R_eff']

    # Define TC strategies (results use Set A/B/C/D; plots use legend A–D)
    tc_list = [
        FixedSlipTC(FIXED_TARGET_SR, R_eff, name="Set A"),
        PacejkaSlipTC(
            copy.deepcopy(true_tyre_params),
            R_eff, name="Set B"),
        PacejkaSlipTC(
            _make_perturbed_params(true_tyre_params, PERTURB_HIGH),
            R_eff, name="Set C"),
        PacejkaSlipTC(
            _make_perturbed_params(true_tyre_params, PERTURB_LOW),
            R_eff, name="Set D"),
    ]

    # Run each strategy
    results: List[RunResult] = []
    for idx, tc in enumerate(tc_list):
        print(f"\nRunning: {tc.name} …")
        t0 = time_module.time()
        r = run_single_event(params, path, tc, true_tyre_params, EVENT_DISTANCE_M)
        r.legend_label = SET_LABELS[idx] if idx < len(SET_LABELS) else None
        wall = time_module.time() - t0
        print(f"  Done ({wall:.1f}s wall) — event {r.finish_time:.3f}s, "
              f"final {r.final_speed_mps * 3.6:.1f} km/h")
        results.append(r)

    # Build output text
    config_text = _get_config_text()
    results_text = _build_results_text(
        results, max_power_W=params['vehicle'].get('max_power_W'))
    print(config_text)
    print(results_text)

    # Timestamped output folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    repo_root = script_dir.parent
    save_dir = repo_root / 'results' / 'simulation' / f'acceleration_event_{timestamp}'
    save_dir.mkdir(parents=True, exist_ok=True)

    print("Generating plots …")
    _plot_main_comparison(results, save_dir)
    _plot_dynamics_overview(results, save_dir)
    _plot_slip_tracking(results, save_dir)
    _plot_pacejka_curves(results, true_tyre_params, save_dir)
    _save_summary(config_text, results_text, save_dir)
    print(f"Done. All outputs saved to: {save_dir}")

    return results


if __name__ == '__main__':
    results = main()
