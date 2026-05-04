#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 27 10:22:39 2025

@author: lewisblake
"""

# gg_diagram_generator.py
import numpy as np
import sys
from pathlib import Path

from vehicle import tyre_interface





def calculate_gg_limits(params, speed_mps):
    """
    Calculates the maximum longitudinal and lateral G-forces the vehicle can
    achieve at a given speed, including longitudinal load transfer and brake bias.

    Returns a dict with:
      - 'lat_g'   : estimated combined lateral g-capability (total tires) (g units)
      - 'brake_g' : self-consistent maximum deceleration (positive number, in g)
      - 'accel_g' : self-consistent maximum acceleration (in g)
      - additional helpful diagnostics may be added if needed
    """
    p_v, p_e = params['vehicle'], params['environment']
    tyre_f, tyre_r = params['tyre_front'], params['tyre_rear']
    m, g = p_v['m_s'], p_e['g']
    L, lf, lr, tw = p_v['wheelbase'], p_v['lf'], p_v['lr'], 0.5*(p_v['track_width_f']+p_v['track_width_r'])
    COG_z = p_v.get('COG_z', 0.2) + (p_v['ride_height_f'] + p_v['ride_height_r'])/2
    COP_x = p_v['COP_x_ratio']

    # Aerodynamic downforce (negative coefficient means pushing down)
    rho = p_e.get('rho_air', 1.225)
    ClA = p_v.get('ClA', 0.0)
    F_downforce = ClA * 0.5 * rho * speed_mps**2

    # Static axle loads (total per axle)
    Fz_static_F_total = m * g * (lr / L)       # front axle total (N)
    Fz_static_R_total = m * g * (lf / L)       # rear axle total  (N)

    # Distribute aero to axles (same convention you had)
    Fz_aero_F = F_downforce * COP_x
    Fz_aero_R = F_downforce * (1-COP_x)

    # Helper: per-corner static (divide axle total by 2)
    def per_corner(F_total):
        return F_total / 2.0

    # Per-wheel brake-system force available (from max_brake_torque) [N]
    R_eff = p_v.get('R_eff', 0.22)
    max_brake_torque = p_v.get('max_brake_torque', 0.0)
    total_brake_force_system = max_brake_torque / max(R_eff, 1e-6)  # N (at wheels, summed both sides)
    bias = p_v.get('brake_bias', 0.5)  # fraction to front axle
    # per-wheel system-limited brake force (positive)
    Fbrake_per_wheel_system = np.array([
        (bias * total_brake_force_system) / 2.0,   # FL
        (bias * total_brake_force_system) / 2.0,   # FR
        ((1.0 - bias) * total_brake_force_system) / 2.0, # RL
        ((1.0 - bias) * total_brake_force_system) / 2.0  # RR
    ])

    # Utility: compute tyre peak Fx and Fy for a given corner vertical load and tyre params
    def tyre_peak_forces_from_Fz(Fz_corner, tyre_params):
        model_type = tyre_params.get('tyre_model_type', 'Standard')
        # Use optimal temperature T_opt from params if possible
        T = tyre_params.get('T', tyre_params.get('T_opt', 60.0))
        Fx_max, Fy_max = tyre_interface.calculate_peak_forces(model_type, Fz_corner, tyre_params, T=T)
        return Fx_max, Fy_max

    # 1) Compute lateral capability at the static (zero-long-accel) loads (approx)
    # Use the static+ aero axle totals (no long transfer)
    Fz_F_corner = per_corner(Fz_static_F_total + Fz_aero_F)
    Fz_R_corner = per_corner(Fz_static_R_total + Fz_aero_R)

    Fx_max_f, Fy_max_f = tyre_peak_forces_from_Fz(Fz_F_corner, tyre_f)
    Fx_max_r, Fy_max_r = tyre_peak_forces_from_Fz(Fz_R_corner, tyre_r)

    # Now total lateral peak force (sum of per-corner lateral peaks)
    Fy_total = Fy_max_f * 2.0 + Fy_max_r * 2.0
    lat_g = Fy_total / (m * g)  # combined lateral capability (g units)

    # 2) Solve for self-consistent brake deceleration (positive g) with longitudinal load transfer
    # We'll iterate on 'a' (decel, positive scalar) until total available brake force = m * a.
    # Start with an initial conservative guess: system-limited sum or tyre-limited static estimate
    tyre_limited_static_brake = min(
        (Fx_max_f * 2.0 + Fx_max_r * 2.0),
        total_brake_force_system
    )
    a_guess = tyre_limited_static_brake / (m + 1e-9)  # m/s^2
    

    

    # iterative solver
    a = a_guess
    for _ in range(10):
        # longitudinal load transfer (ΔF total shifted from front -> rear for positive a)
        deltaF = m * a * COG_z / max(L, 1e-6)  # N
        deltaF_lat = m * lat_g * p_e['g'] * COG_z / max(tw, 1e-6)  # N
        # apply sign: braking (positive a) shifts load to front (because decel), note: we treat 'a' as decel>0
        # For decel, front gains, rear loses:
        Fz_front_total = Fz_static_F_total + Fz_aero_F + deltaF   # note original code signed aero oppositely; keep consistent
        Fz_rear_total  = Fz_static_R_total + Fz_aero_R - deltaF

        # Ensure positivity
        Fz_front_total = max(Fz_front_total, 1.0)
        Fz_rear_total = max(Fz_rear_total, 1.0)

        # per corner loads
        Fz_FL = per_corner(Fz_front_total)
        Fz_FR = per_corner(Fz_front_total)
        Fz_RL = per_corner(Fz_rear_total)
        Fz_RR = per_corner(Fz_rear_total)

        # tyre-limited Fx at each corner for pure longitudinal (peak)
        Fx_FL, _ = tyre_peak_forces_from_Fz(Fz_FL, tyre_f)
        Fx_FR, _ = tyre_peak_forces_from_Fz(Fz_FR, tyre_f)
        Fx_RL, _ = tyre_peak_forces_from_Fz(Fz_RL, tyre_r)
        Fx_RR, _ = tyre_peak_forces_from_Fz(Fz_RR, tyre_r)
        


        tyre_Fx_array = np.array([Fx_FL, Fx_FR, Fx_RL, Fx_RR])  # per-corner tyre capacity (N)

        # system-limited brake per wheel (computed above)
        # actual brake force usable per wheel is min(tyre_capability, system_capability)
        usable_brake_per_wheel = np.minimum(tyre_Fx_array, Fbrake_per_wheel_system)
        
        usable_F = 2*min(usable_brake_per_wheel[0], usable_brake_per_wheel[1])
        usable_R = 2*min(usable_brake_per_wheel[2], usable_brake_per_wheel[3])

        total_usable_brake_force = min(usable_F/(bias + 1e-9), usable_R/(1-bias + 1e-9))

        a_new = total_usable_brake_force / (m + 1e-9)  # m/s^2

        # convergence check
        if abs(a_new - a) < 1e-4:
            a = a_new
            break
        a = 0.5 * (a + a_new)  # relax a bit for stability

    brake_g = a / g if g != 0 else 0.0

    # 3) Solve for self-consistent acceleration g (tyre-limited) using same load-transfer logic
    # For acceleration positive a_accel, load transfers from front -> rear (rear gains load).
    a_acc = 0.5  # initial guess m/s^2
    for _ in range(10):
        deltaF = m * a_acc * COG_z / max(L, 1e-6)
        # acceleration shifts load to rear: front loses, rear gains
        Fz_front_total = Fz_static_F_total + Fz_aero_F - deltaF
        Fz_rear_total  = Fz_static_R_total + Fz_aero_R + deltaF

        Fz_front_total = max(Fz_front_total, 1.0)
        Fz_rear_total = max(Fz_rear_total, 1.0)

        Fz_RL = per_corner(Fz_rear_total)
        Fz_RR = per_corner(Fz_rear_total)

        Fx_RL, _ = tyre_peak_forces_from_Fz(Fz_RL, tyre_r)
        Fx_RR, _ = tyre_peak_forces_from_Fz(Fz_RR, tyre_r)

        tyre_Fx_array = np.array([Fx_RL, Fx_RR])

        # For acceleration, there's no brake-bias cap — if you have a motor force limit, insert it here per wheel
        # For now, we assume tyre-limited only:
        total_drive_tyrelimited = np.sum(tyre_Fx_array)

        a_new = total_drive_tyrelimited / max(m, 1e-9)

        if abs(a_new - a_acc) < 1e-4:
            a_acc = a_new
            break
        a_acc = 0.5 * (a_acc + a_new)

    accel_g = a_acc / g if g != 0 else 0.0

    # Pack results - include both g values (for compatibility) and force maxima (for elliptical constraint)
    return {
        'lat_g': float(lat_g),
        'brake_g': float(brake_g),   # positive (g)
        'accel_g': float(accel_g),    # positive (g)
        'Fx_max_brake': float(brake_g * m * g),  # Max braking force [N]
        'Fx_max_accel': float(accel_g * m * g),  # Max acceleration force [N]
        'Fy_max': float(lat_g * m * g)  # Max lateral force [N]
    }





