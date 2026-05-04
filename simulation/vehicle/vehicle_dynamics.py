#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 18:43:28 2025

@author: lewisblake
"""

# vehicle_dynamics.py
import numpy as np
import sys
from pathlib import Path
# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
# Add path to local core module (in estimation folder)
project_root = Path(__file__).resolve().parent.parent.parent
estimation_path = project_root / 'estimator'
sys.path.insert(0, str(estimation_path))

from vehicle.state_manager import unpack_state_vector, STATE_VECTOR_LAYOUT, STATE_VECTOR_SIZE
from helpers.track_tools import get_track_elevation_at_points
# Import tyre functions from local core module
from core.tyre_model import (  #type: ignore
    D_peak, Calpha_brush, Ckappa_brush, 
    magic_formula_Fy, magic_formula_Fx, 
    apply_traction_ellipse, thermal_mult
)

# CONVENTION DEFINITION:
# +X: Forward (driver's front)
# +Y: Left (driver's left)
# +Z: Up
# +Roll (phi): Right side up (positive rotation around X-axis)
# +Pitch (theta): Nose down (positive rotation around Y-axis)
# +Yaw (r): Turn left (positive rotation around Z-axis)
# This is a standard right-hand coordinate system.

def get_vehicle_forces(state_dict, controls, params, path, sim_time=0.0, T_FL=None, T_FR=None, T_RL=None, T_RR=None):
    """
    Calculates all key forces and moments acting on the vehicle.
    Returns the final summed forces, vertical tire loads, and tire slip info.
    
    Args:
        state_dict: Vehicle state dictionary
        controls: Control inputs dictionary
        params: Vehicle and tyre parameters
        path: Track path dictionary
        sim_time: Current simulation time [s] (for temperature generation)
        T_FL, T_FR, T_RL, T_RR: Optional tyre temperatures [°C]. If all None, generated internally.
    
    Returns:
        forces_moments: Dictionary with Fx, Fy, Fz, Mz
        Fz_tires: Vertical loads [N] for 4 wheels
        My_aero: Aero moment [Nm]
        slip_angles: Slip angles [rad] for 4 wheels
        slip_ratios: Slip ratios [-] for 4 wheels
        susp_details: Suspension details dictionary
        Fx_wheels: Longitudinal forces [N] for 4 wheels
        Fy_wheels: Lateral forces [N] for 4 wheels
        camber_angles_rad: Camber angles [rad] for 4 wheels
        Fjack_total: Total jacking forces [N] for 4 wheels
        diff_details: Differential details array
        r_dot: Yaw acceleration [rad/s²]
        T_FL, T_FR, T_RL, T_RR: Tyre temperatures [°C]
    """
    
    steer_cmd = controls.get('steer_cmd_rad', 0.0)
    d_FL, d_FR = _calculate_ackerman_steer(steer_cmd, params)
    
    # Unpack suspension details to get heave forces for logging
    Fz_tires, susp_forces_at_wheel, _, _, susp_details, susp_defl_at_wheel = _calculate_vertical_and_suspension_forces(state_dict, params, path, d_FL, d_FR)
    
    # Pass susp_defl_at_wheel and get jacking forces (optionally pass tyre temps)
    Fx_wheels, Fy_wheels, slip_angles, slip_ratios, Fjack_total, camber_angles_rad, T_FL, T_FR, T_RL, T_RR = _calculate_tire_forces(
        state_dict, controls, Fz_tires, params, susp_defl_at_wheel, sim_time=sim_time, T_FL=T_FL, T_FR=T_FR, T_RL=T_RL, T_RR=T_RR)
    
    forces_moments, My_aero = _resolve_and_sum_forces(Fx_wheels, Fy_wheels, Fz_tires, controls, state_dict, params)

    # Get differential details for logging
    drive_torque_RL, drive_torque_RR, torque_difference, lock_factor = _get_drive_torques(state_dict, controls, params)
    diff_details = np.array([drive_torque_RL, drive_torque_RR, torque_difference, lock_factor])
    
    # Calculate r_dot (yaw acceleration) for IMU measurement
    # Use the suspension forces we already calculated
    chassis_derivs = _calculate_chassis_derivatives(state_dict, Fx_wheels, Fy_wheels, susp_forces_at_wheel, params, controls)
    r_dot = chassis_derivs.get('r_dot_radps', 0.0)
    
    # Return jacking forces, diff details, and r_dot for logging
    return forces_moments, Fz_tires, My_aero, slip_angles, slip_ratios, susp_details, Fx_wheels, Fy_wheels, camber_angles_rad, Fjack_total, diff_details, r_dot, T_FL, T_FR, T_RL, T_RR


def calculate_derivatives_for_solver(t, y, params, controls, path):
    """ Main function called by the numerical integrator. """
    if not np.all(np.isfinite(y)):
        print(f"!!! CRITICAL @ t={t:.6f}: Non-finite values in INPUT state vector 'y'!")
        return np.zeros_like(y)

    state = unpack_state_vector(y)
    
    steer_cmd = controls.get('steer_cmd_rad', 0.0)
    d_FL, d_FR = _calculate_ackerman_steer(steer_cmd, params)

    Fz_tires, susp_forces_at_wheel, _, _, _, susp_defl_at_wheel = _calculate_vertical_and_suspension_forces(state, params, path, d_FL, d_FR)

    Fx_wheels, Fy_wheels, _, _, Fjack_total, _, _, _, _, _ = _calculate_tire_forces(state, controls, Fz_tires, params, susp_defl_at_wheel, sim_time=t)

    chassis_derivs = _calculate_chassis_derivatives(state, Fx_wheels, Fy_wheels, susp_forces_at_wheel, params, controls)

    drive_torque_RL, drive_torque_RR, _, _ = _get_drive_torques(state, controls, params) # Unpack only what's needed
    brake_torques = _get_brake_torques(controls, params)
    wheel_rot_derivs = _calculate_wheel_rotational_derivatives(Fx_wheels, drive_torque_RL, drive_torque_RR, brake_torques, params)

    suspension_derivs = _calculate_suspension_derivatives(state, Fz_tires, Fjack_total, susp_forces_at_wheel, params)
    
    derivatives = {**chassis_derivs, **wheel_rot_derivs, **suspension_derivs}
    
    dydt = np.zeros(STATE_VECTOR_SIZE)
    for name, value in derivatives.items():
        if name in STATE_VECTOR_LAYOUT:
            dydt[STATE_VECTOR_LAYOUT[name]['slice']] = np.ravel(value)
            
    if not np.all(np.isfinite(dydt)):
        print(f"!!! CRITICAL @ t={t:.6f}: Non-finite values in OUTPUT 'dydt'!")
        for name, value in derivatives.items():
            if not np.all(np.isfinite(value)):
                print(f"    -> Culprit: '{name}' calculation resulted in: {value}")
        dydt[~np.isfinite(dydt)] = 0.0

    return dydt

# Private helper functions for physics calculations

def _get_drive_torques(state, controls, params):
    """ 
    Calculates drive torque per rear wheel, including a more stable LSD model.
    Also returns the torque transferred by the diff and the lock factor.
    """
    p = params['vehicle']
    throttle_cmd = controls.get('throttle_brake_cmd', 0)
    
    # Calculate total available input torque from motor
    avg_rear_w = np.nanmean([state['wheel_w_radps'][2], state['wheel_w_radps'][3]])
    if not np.isfinite(avg_rear_w): avg_rear_w = 0.0
    motor_rpm = avg_rear_w * p['gear_ratio'] * (60 / (2 * np.pi))
    total_input_torque = np.interp(motor_rpm, p['motor_torque_curve'][0], p['motor_torque_curve'][1])
    total_input_torque *= throttle_cmd * p['gear_ratio'] if throttle_cmd > 0 else 0

    # Cap by max power (power = torque * omega at the wheels)
    max_power_W = p.get('max_power_W')
    if max_power_W is not None and avg_rear_w > 1e-6:
        power_W = total_input_torque * avg_rear_w
        if power_W > max_power_W:
            total_input_torque = max_power_W / avg_rear_w
    w_RL = state['wheel_w_radps'][2]
    w_RR = state['wheel_w_radps'][3]
    speed_difference = w_RL - w_RR # Positive if left wheel is faster

    # Determine max locking torque based on accel or coast
    if throttle_cmd > 0.05: # Accelerating
        max_lock_T = p.get('diff_preload', 0) + p.get('diff_lock_torque_accel', 0)
    else: # Coasting or braking
        max_lock_T = p.get('diff_preload', 0) + p.get('diff_lock_torque_coast', 0)

    # The torque that the diff *tries* to transfer. It's capped by max_lock_T.
    # It cannot transfer more than is available (total_input_torque / 2).
    transfer_torque = min(max_lock_T, total_input_torque / 2.0)
    
    # Use tanh for a smooth, stable transition instead of a hard if/else switch
    # The gain (e.g., 50) controls how quickly the lock engages with speed difference.
    # A higher gain makes it more aggressive.
    gain = 1.0 
    lock_factor = np.tanh(speed_difference * gain)

    # The actual torque difference applied across the axle
    torque_difference = transfer_torque * lock_factor

    # Distribute the torque
    base_torque_per_wheel = total_input_torque / 2.0
    drive_torque_RL = base_torque_per_wheel - torque_difference
    drive_torque_RR = base_torque_per_wheel + torque_difference
    
    return drive_torque_RL, drive_torque_RR, torque_difference, lock_factor


def _get_brake_torques(controls, params):
    brake_cmd = controls.get('throttle_brake_cmd', 0)
    if brake_cmd >= 0: return np.zeros(4)
    p = params['vehicle']
    total_brake_torque = abs(brake_cmd) * p['max_brake_torque']
    bias = p['brake_bias']
    return np.array([total_brake_torque*bias/2.0, total_brake_torque*bias/2.0, total_brake_torque*(1-bias)/2.0, total_brake_torque*(1-bias)/2.0])

def _generate_temperature_wave(t, T_nominal=60.0, T_amplitude=0.0, T_freq=0.1, seed=42):
    """Generate temperature wave for each tyre with random variation.
    
    Args:
        t: Current time [s]
        T_nominal: Nominal temperature [°C]
        T_amplitude: Temperature amplitude [°C]
        T_freq: Frequency of temperature wave [Hz]
        seed: Random seed for variation between tyres
    
    Returns:
        T_FL, T_FR, T_RL, T_RR: Temperatures for each tyre [°C]
    """
    rng = np.random.RandomState(seed)
    # Base sinusoidal wave
    T_base = T_nominal  + T_amplitude * np.sin(2 * np.pi * T_freq * t)
    # Add random offsets for each tyre (different seeds)
    offsets = rng.uniform(-0, 0, 4)  # ±0°C variation
    # Add some noise
    noise = rng.normal(0, 0, 4)  # 0°C std noise
    T_all = T_base + offsets + noise
    # Clip to reasonable range
    T_all = np.clip(T_all, 20, 120)
    return T_all[0], T_all[1], T_all[2], T_all[3]

def _calculate_tire_forces(state, controls, Fz_tires, params, susp_defl_at_wheel, sim_time=0.0, T_FL=None, T_FR=None, T_RL=None, T_RR=None):
    """Calculate tyre forces using Combined Model structure with temperature effects.
    
    Args:
        state: Vehicle state dictionary
        controls: Control inputs dictionary
        Fz_tires: Vertical loads for 4 wheels [N]
        params: Vehicle and tyre parameters
        susp_defl_at_wheel: Suspension deflection at each wheel [m]
        sim_time: Current simulation time [s] (for temperature generation)
        T_FL, T_FR, T_RL, T_RR: Optional tyre temperatures [°C]. If None, generated from wave.
    
    Returns:
        Fx_wheels, Fy_wheels: Longitudinal and lateral forces [N] for 4 wheels
        slip_angles: Slip angles [rad] for 4 wheels
        slip_ratios: Slip ratios [-] for 4 wheels
        Fjack_total: Total jacking forces [N] for 4 wheels
        camber_angles_rad: Camber angles [rad] for 4 wheels
        T_FL, T_FR, T_RL, T_RR: Tyre temperatures [°C]
    """
    p_v = params['vehicle']
    vx, vy, r = np.ravel(state['vx_mps'])[0], np.ravel(state['vy_mps'])[0], np.ravel(state['r_radps'])[0]
    phi_f, phi_r = np.ravel(state['phi_f_rad'])[0], np.ravel(state['phi_r_rad'])[0]
    steer_cmd = controls.get('steer_cmd_rad', 0.0)
    d_FL, d_FR = _calculate_ackerman_steer(steer_cmd, params)
    
    vx_for_slip_calc = max(abs(vx), 1.0)

    # Generate temperatures if not provided
    if T_FL is None or T_FR is None or T_RL is None or T_RR is None:
        T_FL, T_FR, T_RL, T_RR = _generate_temperature_wave(sim_time, 
                                                             T_nominal=params.get('tyre_front', {}).get('T_opt', 60.0),
                                                             T_amplitude=20.0, T_freq=0.1)

    # Bump Steer Calculation
    # Positive susp_defl_at_wheel is jounce (bump)
    susp_defl_mm = susp_defl_at_wheel * 1000

    # Get bump steer rates (deg per mm). Convention: Positive = toe IN on bump.
    bump_steer_rate_f = p_v.get('bump_steer_f_deg_per_mm', 0.0)
    bump_steer_rate_r = p_v.get('bump_steer_r_deg_per_mm', 0.0)

    # Calculate toe change from bump steer for each wheel (in radians)
    bump_steer_f_rad = np.deg2rad(susp_defl_mm[0:2] * bump_steer_rate_f)
    bump_steer_r_rad = np.deg2rad(susp_defl_mm[2:4] * bump_steer_rate_r)

    
    # Slip angle calculation including static toe and bump steer
    toe_rad_f_perside = np.deg2rad(p_v['static_toe_f_deg'])
    toe_rad_r_perside = np.deg2rad(p_v['static_toe_r_deg'])
    
    alpha_FL = np.arctan2(vy + p_v['lf'] * r, vx_for_slip_calc) - d_FL + toe_rad_f_perside + bump_steer_f_rad[0]
    alpha_FR = np.arctan2(vy + p_v['lf'] * r, vx_for_slip_calc) - d_FR - toe_rad_f_perside - bump_steer_f_rad[1]
    alpha_RL = np.arctan2(vy - p_v['lr'] * r, vx_for_slip_calc) + toe_rad_r_perside + bump_steer_r_rad[0]
    alpha_RR = np.arctan2(vy - p_v['lr'] * r, vx_for_slip_calc) - toe_rad_r_perside - bump_steer_r_rad[1]
    
    slip_angles = np.array([alpha_FL, alpha_FR, alpha_RL, alpha_RR])

    # Calculate Dynamic Camber for each wheel
    # 1. Static Camber
    static_camber_f_rad = np.deg2rad(p_v['static_camber_f_deg'])
    static_camber_r_rad = np.deg2rad(p_v['static_camber_r_deg'])
    
    # 2. Camber from Roll
    # Convention: Positive roll (phi) is right side up. This induces negative camber on the right wheel and positive on the left.
    # Camber gain is deg/deg, so it's a direct ratio.
    roll_camber_FL =  phi_f * p_v['camber_roll_gain_f']
    roll_camber_FR = -phi_f * p_v['camber_roll_gain_f']
    roll_camber_RL =  phi_r * p_v['camber_roll_gain_r']
    roll_camber_RR = -phi_r * p_v['camber_roll_gain_r']

    # 3. Camber from Steer (due to Caster)
    # Caster induces an asymmetric camber change (negative on outside wheel, positive on inside).
    # Approx: camber_change = -asin(sin(steer) * sin(caster))
    caster_f_rad = np.deg2rad(p_v['caster_f_deg'])
    steer_camber_caster_FL = -np.arcsin(np.clip(np.sin(d_FL) * np.sin(caster_f_rad), -1.0, 1.0))
    steer_camber_caster_FR = -np.arcsin(np.clip(np.sin(d_FR) * np.sin(caster_f_rad), -1.0, 1.0))
    
    # 4. Camber from Steer (due to KPI)
    # KPI induces a symmetric positive camber change with steering.
    # Approx: camber_change = asin(sin(abs(steer)) * sin(kpi))
    kpi_f_rad = np.deg2rad(p_v['kpi_f_deg'])
    steer_camber_kpi_FL = np.arcsin(np.clip(np.sin(np.abs(d_FL)) * np.sin(kpi_f_rad), -1.0, 1.0))
    steer_camber_kpi_FR = np.arcsin(np.clip(np.sin(np.abs(d_FR)) * np.sin(kpi_f_rad), -1.0, 1.0))
    
    # 5. Total Camber
    camber_angles_rad = np.array([
        static_camber_f_rad + roll_camber_FL + steer_camber_caster_FL + steer_camber_kpi_FL,  # FL
        static_camber_f_rad + roll_camber_FR + steer_camber_caster_FR + steer_camber_kpi_FR,  # FR
        static_camber_r_rad + roll_camber_RL,                                                 # RL
        static_camber_r_rad + roll_camber_RR                                                  # RR
    ])

    
    R_eff = p_v['R_eff']
    vx_for_slip_ratio = np.sign(vx) * max(abs(vx), 0.1)
    slip_ratios = (state['wheel_w_radps'] * R_eff) / vx_for_slip_ratio - 1
    

    # Get tyre parameters (Combined Model structure)
    tyre_f = params.get('tyre_front', {})
    tyre_r = params.get('tyre_rear', {})
    
    # Extract Model Type
    # Default to 'Standard' to maintain backward compatibility if not specified
    model_type_f = tyre_f.get('tyre_model_type', 'Standard')
    model_type_r = tyre_r.get('tyre_model_type', 'Standard')
    
    # Default Pressure
    default_pressure = tyre_f.get('default_pressure_kPa', 120.0)
    
    Fx_wheels, Fy_wheels = np.zeros(4), np.zeros(4)
    
    # Calculate forces for each wheel using Unified Interface
    from vehicle import tyre_interface
    
    wheels = [
        (0, 'FL', T_FL, alpha_FL, slip_ratios[0], Fz_tires[0], tyre_f, model_type_f),
        (1, 'FR', T_FR, alpha_FR, slip_ratios[1], Fz_tires[1], tyre_f, model_type_f),
        (2, 'RL', T_RL, alpha_RL, slip_ratios[2], Fz_tires[2], tyre_r, model_type_r),
        (3, 'RR', T_RR, alpha_RR, slip_ratios[3], Fz_tires[3], tyre_r, model_type_r),
    ]
    
    for idx, wheel_name, T, alpha, kappa, Fz, tyre_params, model_type in wheels:
        # Prevent Fz=0 from reaching the tyre model (causes D=0 and division-by-zero in TTC).
        # Use a small minimum so D stays positive; force will still be ~0 when wheel is lifted.
        Fz_safe = max(float(np.ravel(Fz)[0]), 1.0)

        # Calculate forces using the unified interface
        Fx, Fy = tyre_interface.calculate_tyre_forces(
            model_type, alpha, kappa, Fz_safe, T, tyre_params, 
            default_pressure_kPa=default_pressure
        )
        
        Fx_wheels[idx] = Fx
        
        # Negate Fy to match vehicle coordinate convention (Pacejka opposes slip)
        # Note: tyre_interface now returns RAW Force (positive for positive alpha).
        # We negate it here to align with vehicle dynamics convention (Force opposes slip).
        Fy_wheels[idx] = -Fy


    # Note: Front wheels will naturally have small negative Fx during acceleration
    # due to wheel inertia. When the vehicle accelerates, non-driven wheels lag behind
    # because they must be accelerated by tyre forces alone (no drive torque).
    #     # Physics: For a non-driven wheel to accelerate at rate ax, it needs:
    # omega_dot = ax / R
    # Torque = I_wheel * omega_dot = I_wheel * ax / R
    # This torque comes from Fx: Torque = -Fx * R
    # Therefore: Fx = -I_wheel * ax / R²
    #
    # For typical values (I_wheel=1.2 kg·m², R=0.22 m, ax=3 m/s²):
    # Fx ≈ -1.2 * 3 / 0.22² ≈ -74 N per wheel
    #
    # This is physically correct and should not be forced to zero.
    # The magnitude should be proportional to vehicle acceleration and wheel inertia.
    
    # Jacking force calculations
    RC_f, RC_r = p_v['RCH_f_m'], p_v['RCH_r_m']
    TW_f, TW_r = p_v['track_width_f'], p_v['track_width_r']
    Fjack_roll_FL = -Fy_wheels[0] * (2 * RC_f) / TW_f
    Fjack_roll_FR =  Fy_wheels[1] * (2 * RC_f) / TW_f
    Fjack_roll_RL = -Fy_wheels[2] * (2 * RC_r) / TW_r
    Fjack_roll_RR =  Fy_wheels[3] * (2 * RC_r) / TW_r
    Fjack_roll = np.array([Fjack_roll_FL, Fjack_roll_FR, Fjack_roll_RL, Fjack_roll_RR])
    
    PCH_f, PCH_r = p_v['PCH_f_m'], p_v['PCH_r_m']
    L = p_v['wheelbase']
    Fjack_pitch_FL = -Fx_wheels[0] * (2 * PCH_f) / L
    Fjack_pitch_FR = -Fx_wheels[1] * (2 * PCH_f) / L
    Fjack_pitch_RL =  Fx_wheels[2] * (2 * PCH_r) / L
    Fjack_pitch_RR =  Fx_wheels[3] * (2 * PCH_r) / L
    Fjack_pitch = np.array([Fjack_pitch_FL, Fjack_pitch_FR, Fjack_pitch_RL, Fjack_pitch_RR])

    Fjack_total = Fjack_roll + Fjack_pitch

    return Fx_wheels, Fy_wheels, slip_angles, slip_ratios, Fjack_total, camber_angles_rad, T_FL, T_FR, T_RL, T_RR

def _pacejka_magic_formula(x, B, C, D, E):
    """The core Pacejka '94 magic formula equation."""
    # The equation is D*sin(C*arctan(B*x - E*(B*x - arctan(B*x))))
    Bx = B * x
    composite_slip = Bx - E * (Bx - np.arctan(Bx))
    return D * np.sin(C * np.arctan(composite_slip))

def _combined_pacejka(Fz, alpha, kappa, gamma, tyre_params):
    """
    [UNUSED when using unified tyre_interface] Pacejka with shifted friction ellipse.
    The main simulation uses Vehicle.tyre_interface.calculate_tyre_forces(), which
    for 'Standard' uses the same centered traction ellipse as the estimator
    (estimation/core/tyre_model.apply_traction_ellipse). This function is kept for
    reference or alternative pipelines that use raw LAT/LONG params.

    Calculates tire forces using the Pacejka '94 model for pure slip,
    and combines them using a friction ellipse. This model is sensitive to
    vertical load (Fz) and camber angle (gamma).
    """
    # Pre-processing and safety checks
    epsilon = 1e-9
    Fz = np.maximum(np.ravel(Fz), 1.0)  # Ensure Fz is always positive
    alpha = np.clip(np.nan_to_num(alpha), -np.pi/2, np.pi/2)
    kappa = np.clip(np.nan_to_num(kappa), -1.0, 1.0)
    gamma = np.clip(np.nan_to_num(gamma), -np.pi/4, np.pi/4) # Clip camber

    # Convert units for formula consistency (N->kN, rad->deg)
    Fz_kN = Fz / 1000.0
    gamma_deg = np.rad2deg(gamma)
    alpha_deg = np.rad2deg(alpha)

    # PURE LATERAL FORCE (Fy0) CALCULATION
    p_lat = tyre_params['LAT']

    # Calculate lateral coefficients
    C_lat = p_lat['pCy1']
    D_lat = Fz_kN * (p_lat['pDy1'] + p_lat['pDy2'] * Fz_kN) * (1 - p_lat['pDy3'] * gamma_deg**2)
    BCD_lat = p_lat['pKy1'] * np.sin(np.atan( Fz_kN / p_lat['pKy2']) * 2)* (1 - p_lat.get('pKy3', 0) * np.abs(gamma_deg)) 

    B_lat = BCD_lat / (C_lat * D_lat + epsilon)

    
    Sh_lat = p_lat['pHy1'] + p_lat['pHy2'] * Fz_kN + p_lat['pHy3'] * gamma_deg
    Sv_lat = p_lat['pVy1'] + p_lat['pVy2'] * Fz_kN + (p_lat['pVy3'] + p_lat['pVy4'] * Fz_kN) * gamma_deg
    
    alpha_shifted = alpha_deg + Sh_lat
    E_lat = (p_lat['pEy1'] + p_lat['pEy2'] * Fz_kN) * (1 - (p_lat['pEy4'] * gamma_deg + p_lat['pEy5']))
    
    Fy0 = _pacejka_magic_formula(alpha_shifted, B_lat, C_lat, D_lat, E_lat) + Sv_lat

    # PURE LONGITUDINAL FORCE (Fx0) CALCULATION
    p_lon = tyre_params['LONG']
    
    C_lon = p_lon['pCx1']
    D_lon = Fz_kN * (p_lon['pDx1'] + p_lon['pDx2'] * Fz_kN)
    BCD_lon = (p_lon['pKx1'] *  Fz_kN + p_lon['pKx2'] * Fz_kN**2) * np.exp(-p_lon['pKx3'] * Fz_kN)
    B_lon = BCD_lon / (C_lon * D_lon + epsilon)

    
    Sh_lon = p_lon['pHx1'] + p_lon['pHx2'] * Fz_kN
    Sv_lon = p_lon['pVx1'] + p_lon['pVx2'] * Fz_kN

    kappa_shifted = kappa * 100 + Sh_lon
    E_lon = (p_lon['pEx1'] + p_lon['pEx2'] * Fz_kN + p_lon['pEx3'] * Fz_kN**2) * (1 - p_lon['pEx4'] * np.sign(kappa_shifted))
    
    Fx0 = _pacejka_magic_formula(kappa_shifted, B_lon, C_lon, D_lon, E_lon) + Sv_lon

    # COMBINED SLIP (Friction Ellipse Method)
    # The ellipse is centered at (Sv_lon, Sv_lat) with radii D_lon and D_lat
    ellipse_val = ((Fx0 - Sv_lon) / (D_lon + epsilon))**2 + ((Fy0 - Sv_lat) / (D_lat + epsilon))**2

    Fx, Fy = Fx0, Fy0
    
    # Find where forces are outside the ellipse and scale them down
    mask = ellipse_val > 1.0
    if np.any(mask):
        scaling_factor = np.sqrt(1.0 / ellipse_val[mask])
        Fx[mask] = Sv_lon[mask] + (Fx0[mask] - Sv_lon[mask]) * scaling_factor
        Fy[mask] = Sv_lat[mask] + (Fy0[mask] - Sv_lat[mask]) * scaling_factor
            
    # Per vehicle coordinate system convention, Fy is positive for a positive alpha.
    # Pacejka calculates force opposing slip, so we negate Fy.
    return np.vstack([Fx, -Fy])

def _calculate_wheel_rotational_derivatives(Fx, drive_RL, drive_RR, brake, params):
    p = params['vehicle']
    R, I = p['R_eff'], p['I_wheel']
    net_torque = np.array([-brake[0]-Fx[0]*R, -brake[1]-Fx[1]*R, drive_RL-brake[2]-Fx[2]*R, drive_RR-brake[3]-Fx[3]*R])
    return {'wheel_w_radps': net_torque / I}

def _calculate_vertical_and_suspension_forces(state, params, path, d_FL, d_FR):
    p_v = params['vehicle']
    
    Z, theta = np.ravel(state['Z_m'])[0], np.ravel(state['theta_rad'])[0]
    phi_f, phi_r = np.ravel(state['phi_f_rad'])[0], np.ravel(state['phi_r_rad'])[0]
    vz, q, r = np.ravel(state['vz_mps'])[0], np.ravel(state['q_radps'])[0], np.ravel(state['r_radps'])[0]
    p_f, p_r = np.ravel(state['p_f_radps'])[0], np.ravel(state['p_r_radps'])[0]
    wheel_Z, wheel_Z_dot = state['wheel_Z_m'], state['wheel_Z_dot_mps']
    h_cg = p_v['COG_z'] 
    
    wheel_coords_world = _get_wheel_coords_world(state, params)
    z_track = get_track_elevation_at_points(wheel_coords_world, path)
    
   # Calculate kinematic vertical displacement from steering geometry
    caster_f_rad = np.deg2rad(p_v['caster_f_deg'])
    kpi_f_rad = np.deg2rad(p_v['kpi_f_deg'])
    
    dZ_kpi_FL = p_v['scrub_radius_f'] * np.tan(kpi_f_rad) * (1 - np.cos(d_FL))
    dZ_kpi_FR = p_v['scrub_radius_f'] * np.tan(kpi_f_rad) * (1 - np.cos(d_FR))
    
    dZ_caster_FL = -(-1) * p_v['R_eff'] * np.tan(caster_f_rad) * np.sin(d_FL)
    dZ_caster_FR = -(+1) * p_v['R_eff'] * np.tan(caster_f_rad) * np.sin(d_FR)
    
    kinematic_lift = np.array([
        dZ_kpi_FL + dZ_caster_FL,
        dZ_kpi_FR + dZ_caster_FR,
        0.0,
        0.0
    ])
    
    effective_z_ground = z_track + kinematic_lift
    
    tire_compression = p_v['R_eff'] - (wheel_Z - effective_z_ground)
    
    Fz_tires = p_v['K_tire'] * tire_compression
    Fz_tires = np.maximum(0, Fz_tires)

    chassis_corner_Z = np.array([
        Z - h_cg - p_v['lf'] * np.sin(theta) - (p_v['track_width_f']/2) * np.sin(phi_f), # FL (uses phi_f)
        Z - h_cg - p_v['lf'] * np.sin(theta) + (p_v['track_width_f']/2) * np.sin(phi_f), # FR (uses phi_f)
        Z - h_cg + p_v['lr'] * np.sin(theta) - (p_v['track_width_r']/2) * np.sin(phi_r), # RL (uses phi_r)
        Z - h_cg + p_v['lr'] * np.sin(theta) + (p_v['track_width_r']/2) * np.sin(phi_r), # RR (uses phi_r)
    ])
    
    p_avg = (p_f + p_r) / 2.0 
    phi_avg = (phi_f + phi_r) / 2.0 
    _, theta_dot, _ = _get_euler_rates(p_avg, q, r, phi_avg, theta)

    chassis_corner_Z_dot = np.array([
        vz - p_v['lf'] * np.cos(theta) * theta_dot - (p_v['track_width_f']/2) * np.cos(phi_f) * p_f, # FL (uses p_f)
        vz - p_v['lf'] * np.cos(theta) * theta_dot + (p_v['track_width_f']/2) * np.cos(phi_f) * p_f, # FR (uses p_f)
        vz + p_v['lr'] * np.cos(theta) * theta_dot - (p_v['track_width_r']/2) * np.cos(phi_r) * p_r, # RL (uses p_r)
        vz + p_v['lr'] * np.cos(theta) * theta_dot + (p_v['track_width_r']/2) * np.cos(phi_r) * p_r, # RR (uses p_r)
    ])

    current_suspension_length = wheel_Z - chassis_corner_Z
    static_lengths = np.array([p_v['suspension_static_length_f_m'], p_v['suspension_static_length_f_m'],
                               p_v['suspension_static_length_r_m'], p_v['suspension_static_length_r_m']])
    susp_defl_at_wheel = (current_suspension_length - static_lengths)

    susp_vel_at_wheel = chassis_corner_Z_dot - np.ravel(wheel_Z_dot)

    base_mr = np.array([p_v['motion_ratio_f'], p_v['motion_ratio_f'], p_v['motion_ratio_r'], p_v['motion_ratio_r']])
    mr_rate = np.array([p_v['motion_ratio_rate_f'], p_v['motion_ratio_rate_f'], p_v['motion_ratio_rate_r'], p_v['motion_ratio_rate_r']])
    
    approx_shock_travel = susp_defl_at_wheel / base_mr
    current_mr = base_mr * (1 + mr_rate * approx_shock_travel)

    shock_travel = susp_defl_at_wheel / current_mr
    shock_velocity = susp_vel_at_wheel / current_mr
    
    K_s = np.array([p_v['K_s_f'], p_v['K_s_f'], p_v['K_s_r'], p_v['K_s_r']])
    C_s = np.array([p_v['C_s_f'], p_v['C_s_f'], p_v['C_s_r'], p_v['C_s_r']])
    F_preload = np.array([p_v['preload_f'], p_v['preload_f'], p_v['preload_r'], p_v['preload_r']])

    F_spring_on_shock = K_s * shock_travel + F_preload
    F_damper_on_shock = -C_s * shock_velocity
    
    F_bump_stop = np.zeros(4)
    over_travel = shock_travel - p_v['shock_stroke_max_m']
    F_bump_stop[over_travel > 0] = p_v['K_bump_stop'] * over_travel[over_travel > 0]
    
    F_rebound_stop = np.zeros(4)
    rebound_over_travel = -shock_travel - p_v['shock_stroke_rebound_max_m']
    F_rebound_stop[rebound_over_travel > 0] = -p_v['K_bump_stop'] * rebound_over_travel[rebound_over_travel > 0]

    F_arb_on_shock = np.zeros(4)
    arb_twist_f = shock_travel[1] - shock_travel[0] # FR - FL
    arb_moment_f = p_v['K_arb_f'] * arb_twist_f
    F_arb_on_shock[0] = -arb_moment_f / (p_v['track_width_f'] / current_mr[0])
    F_arb_on_shock[1] = arb_moment_f / (p_v['track_width_f'] / current_mr[1])
    
    arb_twist_r = shock_travel[3] - shock_travel[2] # RR - RL
    arb_moment_r = p_v['K_arb_r'] * arb_twist_r
    F_arb_on_shock[2] = -arb_moment_r / (p_v['track_width_r'] / current_mr[2])
    F_arb_on_shock[3] = arb_moment_r / (p_v['track_width_r'] / current_mr[3])

    # NEW: Heave Spring Calculation
    heave_disp_f = (shock_travel[0] + shock_travel[1]) / 2.0
    heave_vel_f = (shock_velocity[0] + shock_velocity[1]) / 2.0
    F_heave_f = p_v.get('K_heave_f', 0) * heave_disp_f - p_v.get('C_heave_f', 0) * heave_vel_f
    
    heave_disp_r = (shock_travel[2] + shock_travel[3]) / 2.0
    heave_vel_r = (shock_velocity[2] + shock_velocity[3]) / 2.0
    F_heave_r = p_v.get('K_heave_r', 0) * heave_disp_r - p_v.get('C_heave_r', 0) * heave_vel_r
    
    F_heave_on_shock = np.array([F_heave_f / 2.0, F_heave_f / 2.0, F_heave_r / 2.0, F_heave_r / 2.0])

    total_force_on_shock = F_spring_on_shock + F_damper_on_shock + F_bump_stop + F_rebound_stop + F_arb_on_shock + F_heave_on_shock
    total_susp_force_at_wheel = total_force_on_shock / current_mr
    
    susp_details = {
        'spring': F_spring_on_shock,
        'damper': F_damper_on_shock,
        'arb': F_arb_on_shock,
        'shock_vel': shock_velocity,
        'heave': np.array([F_heave_f, F_heave_r]), # Store total axle heave force
        'bump': F_bump_stop + F_rebound_stop
    }
    
    return Fz_tires, total_susp_force_at_wheel, chassis_corner_Z, chassis_corner_Z_dot, susp_details, susp_defl_at_wheel


def _calculate_suspension_derivatives(state, Fz_tires, Fjack_total, susp_forces_at_wheel, params):
    p_v = params['vehicle']
    g = params['environment']['g']
    m_us = np.array([p_v['m_us_f'], p_v['m_us_f'], p_v['m_us_r'], p_v['m_us_r']])
    
    # Use the total combined jacking force in the net force calculation
    F_net_us = Fz_tires - Fjack_total - susp_forces_at_wheel - (m_us * g)
    
    wheel_Z_ddot = F_net_us / m_us
    
    return {
        'wheel_Z_m': state['wheel_Z_dot_mps'], 
        'wheel_Z_dot_mps': wheel_Z_ddot
    }

def _resolve_and_sum_forces(Fx_wheels, Fy_wheels, Fz_wheels, controls, state, params):
    p_v, p_e = params['vehicle'], params['environment']
    steer_cmd = controls.get('steer_cmd_rad', 0.0)
    d_FL,d_FR = _calculate_ackerman_steer(steer_cmd, params)
    
    Fx_FL,Fx_FR,Fx_RL,Fx_RR = Fx_wheels
    Fy_FL,Fy_FR,Fy_RL,Fy_RR = Fy_wheels
    
    Fx_FL_c = Fx_FL*np.cos(d_FL) - Fy_FL*np.sin(d_FL)
    Fy_FL_c = Fx_FL*np.sin(d_FL) + Fy_FL*np.cos(d_FL)
    
    Fx_FR_c = Fx_FR*np.cos(d_FR) - Fy_FR*np.sin(d_FR)
    Fy_FR_c = Fx_FR*np.sin(d_FR) + Fy_FR*np.cos(d_FR)

    Sum_Fx = Fx_FL_c + Fx_FR_c + Fx_RL + Fx_RR
    Sum_Fy = Fy_FL_c + Fy_FR_c + Fy_RL + Fy_RR
    
    vx_scalar = np.ravel(state['vx_mps'])[0]
    F_drag = p_v['CdA'] * 0.5 * p_e['rho_air'] * abs(vx_scalar) * vx_scalar
    Sum_Fx -= F_drag
    
    Sum_Mz = (Fy_FL_c + Fy_FR_c) * p_v['lf'] - (Fy_RL + Fy_RR) * p_v['lr'] \
           + (Fx_FR_c - Fx_FL_c) * p_v['track_width_f'] / 2 \
           + (Fx_RR - Fx_RL) * p_v['track_width_r'] / 2
           
    F_downforce = -p_v['ClA'] * 0.5 * p_e['rho_air'] * vx_scalar**2
    My_aero_pitch = F_downforce * (p_v['wheelbase'] * (p_v['COG_ratio_x'] - p_v['COP_x_ratio']))

    return {'Fx':Sum_Fx, 'Fy':Sum_Fy, 'Mz':Sum_Mz, 'F_downforce': F_downforce}, My_aero_pitch

def _calculate_chassis_derivatives(state, Fx_wheels, Fy_wheels, susp_forces_at_wheel, params, controls):
    p_v, p_e = params['vehicle'], params['environment']
    
    vx,vy,r = np.ravel(state['vx_mps'])[0], np.ravel(state['vy_mps'])[0], np.ravel(state['r_radps'])[0]
    p_f, p_r, q = np.ravel(state['p_f_radps'])[0], np.ravel(state['p_r_radps'])[0], np.ravel(state['q_radps'])[0]
    phi_f, phi_r, theta, psi = np.ravel(state['phi_f_rad'])[0], np.ravel(state['phi_r_rad'])[0], np.ravel(state['theta_rad'])[0], np.ravel(state['psi_rad'])[0]
    Z = np.ravel(state['Z_m'])[0]
    
    steer_cmd = controls.get('steer_cmd_rad', 0.0)
    d_FL, d_FR = _calculate_ackerman_steer(steer_cmd, params)
    
    Fx_FL_c = Fx_wheels[0]*np.cos(d_FL) - Fy_wheels[0]*np.sin(d_FL)
    Fy_FL_c = Fx_wheels[0]*np.sin(d_FL) + Fy_wheels[0]*np.cos(d_FL)
    Fx_FR_c = Fx_wheels[1]*np.cos(d_FR) - Fy_wheels[1]*np.sin(d_FR)
    Fy_FR_c = Fx_wheels[1]*np.sin(d_FR) + Fy_wheels[1]*np.cos(d_FR)

    Sum_Fx_tires = Fx_FL_c + Fx_FR_c + Fx_wheels[2] + Fx_wheels[3]
    Sum_Fy_tires_f = Fy_FL_c + Fy_FR_c
    Sum_Fy_tires_r = Fy_wheels[2] + Fy_wheels[3]
    Sum_Fy_tires = Sum_Fy_tires_f + Sum_Fy_tires_r

    F_drag = p_v['CdA'] * 0.5 * p_e['rho_air'] * abs(vx) * vx
    F_downforce = -p_v['ClA'] * 0.5 * p_e['rho_air'] * vx**2

    Sum_Fx_ext = Sum_Fx_tires - F_drag
    Sum_Fy_ext = Sum_Fy_tires
    
    ax = Sum_Fx_ext / p_v['mass']
    ay = Sum_Fy_ext / p_v['mass']
    
    vx_dot = ax + r * vy
    vy_dot = ay - r * vx

    Sum_Fz_sprung = np.sum(susp_forces_at_wheel) + F_downforce - p_v['m_s'] * p_e['g']
    vz_dot = Sum_Fz_sprung / p_v['m_s']
    
    
    # Roll Moments (Mx)
    Ix_f = p_v['Ix'] * p_v['lr'] / p_v['wheelbase']
    Ix_r = p_v['Ix'] * p_v['lf'] / p_v['wheelbase']

    # Front Roll Moments
    Mx_susp_f = (susp_forces_at_wheel[1] - susp_forces_at_wheel[0]) * p_v['track_width_f'] / 2.0
    Mx_tires_f = Sum_Fy_tires_f * (p_v['RCH_f_m'] - Z)
    Mx_torsion = p_v['K_torsion_Nm_rad'] * (phi_r - phi_f)
    Sum_Mx_f = Mx_susp_f + Mx_tires_f + Mx_torsion

    # Rear Roll Moments
    Mx_susp_r = (susp_forces_at_wheel[3] - susp_forces_at_wheel[2]) * p_v['track_width_r'] / 2.0
    Mx_tires_r = Sum_Fy_tires_r * (p_v['RCH_r_m'] - Z)
    Sum_Mx_r = Mx_susp_r + Mx_tires_r - Mx_torsion

    p_f_dot = Sum_Mx_f / (Ix_f + 1e-9)
    p_r_dot = Sum_Mx_r / (Ix_r + 1e-9)

    # Pitch and Yaw Moments (My, Mz)
    My_susp = (susp_forces_at_wheel[2] + susp_forces_at_wheel[3]) * p_v['lr'] - (susp_forces_at_wheel[0] + susp_forces_at_wheel[1]) * p_v['lf'] 
    My_tires = Sum_Fx_tires * ((p_v['PCH_f_m'] + p_v['PCH_r_m']) / 2 - Z)
    My_aero = F_downforce * (p_v['wheelbase'] * (p_v['COG_ratio_x'] - p_v['COP_x_ratio']))
    Sum_My = My_susp + My_tires + My_aero

    Sum_Mz = (Sum_Fy_tires_f) * p_v['lf'] - (Sum_Fy_tires_r) * p_v['lr'] \
           + (Fx_FR_c - Fx_FL_c) * p_v['track_width_f'] / 2 \
           + (Fx_wheels[3] - Fx_wheels[2]) * p_v['track_width_r'] / 2

    p_avg = (p_f * Ix_f + p_r * Ix_r) / (p_v['Ix'] + 1e-9)
    
    q_dot = (Sum_My - (p_v['Ix'] - p_v['Iz']) * p_avg * r) / p_v['Iy']
    r_dot = (Sum_Mz - (p_v['Iy'] - p_v['Ix']) * p_avg * q) / p_v['Iz']
    
    phi_avg = (phi_f * Ix_f + phi_r * Ix_r) / (p_v['Ix'] + 1e-9)
    _, theta_dot, psi_dot = _get_euler_rates(p_avg, q, r, phi_avg, theta)
    
    X_dot = vx * np.cos(psi) - vy * np.sin(psi)
    Y_dot = vx * np.sin(psi) + vy * np.cos(psi)
    
    return {
        'vx_mps': vx_dot, 'vy_mps': vy_dot, 'vz_mps': vz_dot,
        'p_f_radps': p_f_dot, 
        'p_r_radps': p_r_dot, 
        'q_radps': q_dot, 
        'r_radps': r_dot,  # This is actually r_dot (yaw acceleration), not r (yaw rate)
        'X_m': X_dot, 'Y_m': Y_dot, 'Z_m': state['vz_mps'],
        'phi_f_rad': state['p_f_radps'], 
        'phi_r_rad': state['p_r_radps'], 
        'theta_rad': theta_dot,
        'psi_rad': psi_dot,
        'r_dot_radps': r_dot  # Store r_dot explicitly
    }

def _get_euler_rates(p, q, r, phi, theta):
    """
    Calculates the time derivative of the Euler angles (phi, theta, psi)
    from the body-frame angular velocities (p, q, r).
    This is based on a ZYX rotation sequence.
    """
    sin_phi, cos_phi = np.sin(phi), np.cos(phi)
    sin_theta, cos_theta = np.sin(theta), np.cos(theta)
    tan_theta = sin_theta / (cos_theta + 1e-9)

    phi_dot = p + q * sin_phi * tan_theta - r * cos_phi * tan_theta
    theta_dot = q * cos_phi + r * sin_phi
    
    if abs(cos_theta) < 1e-6:
        psi_dot = 0 
    else:
        psi_dot = (-q * sin_phi + r * cos_phi) / cos_theta

    return phi_dot, theta_dot, psi_dot

def _calculate_ackerman_steer(delta_avg, params):
    delta_avg = np.nan_to_num(np.ravel(delta_avg)[0])
    p_v=params['vehicle']
    L,T_f,pct=p_v['wheelbase'],p_v['track_width_f'],p_v['ackerman_percentage']/100.0
    
    if abs(delta_avg)<1e-9: return 0.0,0.0
    
    R_turn = L / np.tan(abs(delta_avg)) if abs(np.tan(abs(delta_avg))) > 1e-9 else L / 1e-9
    if abs(R_turn) < T_f / 2: R_turn = np.sign(R_turn) * T_f / 2 if R_turn != 0 else T_f/2
        
    delta_outer = np.arctan(L / (R_turn + T_f / 2))
    delta_inner = np.arctan(L / (R_turn - T_f / 2))
    
    d_o_ack = (1-pct)*abs(delta_avg) + pct*delta_outer
    d_i_ack = (1-pct)*abs(delta_avg) + pct*delta_inner
    
    if delta_avg > 0:
        return d_i_ack, d_o_ack
    else:
        return -d_o_ack, -d_i_ack

def _get_wheel_coords_world(state, params):
    p_v = params['vehicle']
    X, Y, psi = np.ravel(state['X_m'])[0], np.ravel(state['Y_m'])[0], np.ravel(state['psi_rad'])[0]
    
    coords_body = np.array([
        [p_v['lf'], -p_v['track_width_f']/2], # FL
        [p_v['lf'], p_v['track_width_f']/2],  # FR
        [-p_v['lr'], -p_v['track_width_r']/2], # RL
        [-p_v['lr'], p_v['track_width_r']/2]  # RR
    ])
    
    R_psi = np.array([
        [np.cos(psi), -np.sin(psi)],
        [np.sin(psi), np.cos(psi)]
    ])
    
    coords_world = np.dot(coords_body, R_psi.T) + np.array([X, Y])
    return coords_world

def get_fx_at_slip_ratio(kappa, Fz, tyre_params, T=80.0):
    """Calculate longitudinal force Fx at a specific slip ratio for given tyre parameters.

    This helper function is used by the speed profile generator to determine what
    braking/acceleration force is available at the TC-limited slip ratio (not just peak).

    Args:
        kappa: Target slip ratio [-] (scalar or array)
        Fz: Vertical load [N] (scalar or array)
        tyre_params: Dictionary of tyre parameters (must contain D_ref_lon, k_load_lon, etc.)
        T: Tyre temperature [°C] (default 80°C)

    Returns:
        Fx: Longitudinal force at the given slip ratio [N]
    """
    # Extract Model Type
    model_type = tyre_params.get('tyre_model_type', 'Standard')
    
    # Use unified interface. Alpha=0 for pure longitudinal.
    # Note: T default is 80.0 in argument signature, pass it through.
    from vehicle import tyre_interface
    
    # Pressure
    default_pressure = tyre_params.get('default_pressure_kPa', 120.0)
    
    Fx, _ = tyre_interface.calculate_tyre_forces(
        model_type, 0.0, kappa, Fz, T, tyre_params, default_pressure_kPa=default_pressure
    )
    
    return Fx
