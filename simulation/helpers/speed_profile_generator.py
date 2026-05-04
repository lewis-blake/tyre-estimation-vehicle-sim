#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 18:42:49 2025

@author: lewisblake
"""

# speed_profile_generator.py
import numpy as np
from scipy.signal import find_peaks
import sys
from pathlib import Path
import time
# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from analysis.gg_diagram_generator import calculate_gg_limits
from vehicle.vehicle_dynamics import get_fx_at_slip_ratio

def calculate_tc_limited_forces(params, ax_g, lat_g_req, gg_limits):
    """Calculate TC-limited braking and acceleration forces accounting for load transfer.

    Args:
        params: Vehicle parameters dictionary
        ax_g: Longitudinal acceleration [g] (positive = accel, negative = brake)
        lat_g_req: Required lateral acceleration [g] (currently unused but kept for future enhancement)
        gg_limits: G-G limits from calculate_gg_limits (currently unused but kept for future enhancement)

    Returns:
        Fx_max_tc_limited: Maximum longitudinal force available at TC threshold [N]
    """
    p_v = params['vehicle']
    p_env = params['environment']
    p_driver = params.get('driver', {})
    tyre_f = params['tyre_front']
    tyre_r = params['tyre_rear']

    m = p_v['m_s']
    g = p_env['g']
    L = p_v['wheelbase']
    lf = p_v['lf']
    lr = p_v['lr']
    COG_z = p_v.get('COG_z', 0.2) + (p_v['ride_height_f'] + p_v['ride_height_r']) / 2

    # Get TC thresholds
    tc_brake_sr_threshold = abs(p_driver.get('tc_brake_sr_threshold', 0.3))
    tc_throttle_sr_threshold = abs(p_driver.get('tc_throttle_sr_threshold', 0.08))
    brake_bias = p_v.get('brake_bias', 0.63)

    # Calculate longitudinal load transfer
    deltaF_lon = m * abs(ax_g) * g * COG_z / max(L, 1e-6)

    # Static loads per corner
    Fz_static_F = m * g * (lr / L) / 2  # Per front corner
    Fz_static_R = m * g * (lf / L) / 2  # Per rear corner

    if ax_g < 0:  # Braking - load transfers to front
        Fz_F = Fz_static_F + deltaF_lon / 2
        Fz_R = max(Fz_static_R - deltaF_lon / 2, 100.0)  # Prevent negative

        # Calculate Fx available at TC brake threshold for each axle
        Fx_F_per_wheel = abs(get_fx_at_slip_ratio(-tc_brake_sr_threshold, Fz_F, tyre_f))
        Fx_R_per_wheel = abs(get_fx_at_slip_ratio(-tc_brake_sr_threshold, Fz_R, tyre_r))

        # Account for brake bias - total braking limited by whichever axle hits limit first
        # Total available from front: 2 * Fx_F_per_wheel, needs to be <= brake_bias * total
        # Total available from rear: 2 * Fx_R_per_wheel, needs to be <= (1-brake_bias) * total
        Fx_max_tc_limited = min(
            2 * Fx_F_per_wheel / max(brake_bias, 1e-6),
            2 * Fx_R_per_wheel / max(1 - brake_bias, 1e-6)
        )

    else:  # Acceleration - load transfers to rear
        Fz_R = Fz_static_R + deltaF_lon / 2

        # Only rear wheels driven (RWD)
        Fx_R_per_wheel = abs(get_fx_at_slip_ratio(tc_throttle_sr_threshold, Fz_R, tyre_r))
        Fx_max_tc_limited = 2 * Fx_R_per_wheel

    return Fx_max_tc_limited

def generate_dynamic_speed_profile(path, params):
    """
    Generates a physically-based target speed profile using an elliptical friction
    constraint to account for combined longitudinal and lateral forces.

    Key features:
    - Elliptical constraint: (Fx/Fx_max)^2 + (Fy/Fy_max)^2 <= 1
    - TC-aware: Respects driver TC slip ratio thresholds (not just peak tyre capability)
    - Load transfer: Accounts for longitudinal load transfer in force calculations
    - Brake bias: Considers fixed brake bias when calculating max braking force
    """
    g = params['environment']['g']
    num_points = len(path['s'])
    max_speed = params['vehicle']['max_speed_mps']

    # Get stability margin from driver params (default: no margin, use full TC capability)
    p_driver = params.get('driver', {})
    stability_margin = p_driver.get('speed_profile_stability_margin', 1.00)  # 1.0 = use full capability
    
    curv_abs = np.abs(path['curv'])
    apexes, _ = find_peaks(curv_abs, prominence=0.005, distance=10) 
    print(f"Found {len(apexes)} apexes at indices: {apexes}")

    if len(apexes) == 0:
        print("Warning: No apexes found. Speed profile will be flat.")
        path['target_speed'] = np.full(num_points, max_speed)
        return path

    # Cache gg_limits for similar speeds to avoid redundant calculations
    # This significantly speeds up the iterative process
    gg_cache = {}
    cache_tolerance = 0.5  # Cache speeds within 0.5 m/s
    
    def get_cached_gg_limits(speed):
        # Round to nearest cache_tolerance
        speed_key = round(speed / cache_tolerance) * cache_tolerance
        if speed_key not in gg_cache:
            gg_cache[speed_key] = calculate_gg_limits(params, speed_key)
        return gg_cache[speed_key]

    v_corner_limited = np.full(num_points, float(max_speed))
    print("Solving for max cornering speeds iteratively...")
    
    start_time = time.time()
    for iter_num in range(10):
        if iter_num % 2 == 0:  # Print every 2 iterations
            elapsed = time.time() - start_time
            print(f"  Iteration {iter_num+1}/10... (elapsed: {elapsed:.1f}s)", end='\r')
        
        max_change = 0.0
        for i in range(num_points):
            radius = 1.0 / (curv_abs[i] + 1e-6)
            v_guess = v_corner_limited[i] 
            gg_limits = get_cached_gg_limits(v_guess)

            # Apply stability margin to lateral G
            lat_g_available = gg_limits['lat_g'] * stability_margin
            
            sqrt_val = lat_g_available * g * radius
            if sqrt_val > 0:
                v_max_at_point = np.sqrt(sqrt_val)
                max_change = max(max_change, abs(v_max_at_point - v_corner_limited[i]))
                v_corner_limited[i] = v_max_at_point
        
        # Early exit if converged
        if max_change < 0.01:  # Less than 0.01 m/s change
            print(f"  Converged after {iter_num+1} iterations (max change: {max_change:.4f} m/s)")
            break
    
    elapsed = time.time() - start_time
    print(f"  Cornering speed calculation complete ({elapsed:.1f}s)")

    # Apply lap boundary margin to corner-limited speeds BEFORE accel/braking passes
    # This ensures smooth transitions since the passes will account for the slower boundary speeds
    p_driver = params.get('driver', {})
    lap_boundary_margin = p_driver.get('lap_boundary_margin', 0.98)
    lap_boundary_zone_fraction = p_driver.get('lap_boundary_zone_fraction', 0.02)
    boundary_zone = int(lap_boundary_zone_fraction * num_points)

    if boundary_zone > 0:
        print(f"Applying lap boundary margin (zone: {boundary_zone} points, margin: {lap_boundary_margin:.2f})...")
        for i in range(boundary_zone):
            # Smooth cosine taper from full margin at boundary to no margin at boundary_zone
            taper = 0.5 * (1.0 + np.cos(np.pi * i / boundary_zone))  # 1.0 at i=0, 0.0 at i=boundary_zone
            margin_factor = 1.0 - (1.0 - lap_boundary_margin) * taper
            v_corner_limited[-(i+1)] *= margin_factor

    print("Generating independent acceleration and braking limit profiles...")

    v_brake_limited = v_corner_limited.copy()
    start_time = time.time()
    total_iterations = num_points * 2
    
    for iter_num in range(10):
        if iter_num % 2 == 0:  # Print every 2 iterations
            elapsed = time.time() - start_time
            print(f"  Braking profile iteration {iter_num+1}/10... (elapsed: {elapsed:.1f}s)", end='\r')
        
        max_change = 0.0
        for i in range(total_iterations):
            idx = (num_points - 1) - (i % num_points)
            idx_next = (idx + 1) % num_points
            
            ds = path['ds'][idx_next]
            if ds < 1e-6: continue

            v_next = v_brake_limited[idx_next]
            gg = get_cached_gg_limits(v_next)
            lat_g_req = abs(v_next**2 * path['curv'][idx_next] / g)

            # Apply stability margin to lateral G
            lat_g_max = gg['lat_g'] * stability_margin

            # Ensure we don't exceed the stability-limited lateral G
            if lat_g_req > lat_g_max:
                # Reduce speed to stay within stability margin
                v_max_stable = np.sqrt(lat_g_max * g / (abs(path['curv'][idx_next]) + 1e-6))
                v_next = min(v_next, v_max_stable)
                lat_g_req = lat_g_max

            # Use elliptical friction constraint instead of circular
            # (Fx/Fx_max)^2 + (Fy/Fy_max)^2 <= 1
            # Solving for Fx: Fx = Fx_max * sqrt(1 - (Fy/Fy_max)^2)
            Fx_max_brake = gg['Fx_max_brake']
            Fy_max = gg['Fy_max']
            Fy_req = lat_g_req * params['vehicle']['m_s'] * g

            # Calculate what the TC threshold allows (for pure longitudinal slip)
            # Use iterative approach: estimate decel, calculate TC limit, update
            ax_g_estimate = gg['brake_g']  # Initial estimate from g-g diagram
            Fx_tc_pure = calculate_tc_limited_forces(params, -ax_g_estimate, 0.0, gg)  # Pure longitudinal

            # Now apply traction ellipse to TC-limited force
            # The TC limit is what we can do in pure slip; ellipse reduces it for combined slip
            Fx_brake = Fx_tc_pure * np.sqrt(max(0, 1 - (Fy_req / max(Fy_max, 1e-6))**2))

            # Also ensure we don't exceed the maximum possible (from g-g diagram)
            Fx_avail_ellipse = Fx_max_brake * np.sqrt(max(0, 1 - (Fy_req / max(Fy_max, 1e-6))**2))
            Fx_brake = min(Fx_brake, Fx_avail_ellipse)

            decel = -Fx_brake / params['vehicle']['m_s']
            
            v_max_sq = v_next**2 - 2 * decel * ds
            if v_max_sq > 0:
                v_new = np.sqrt(v_max_sq)
                max_change = max(max_change, abs(v_new - v_brake_limited[idx]))
                v_brake_limited[idx] = min(v_brake_limited[idx], v_new)
        
        # Early exit if converged
        if max_change < 0.01:
            print(f"  Braking profile converged after {iter_num+1} iterations (max change: {max_change:.4f} m/s)")
            break
    
    elapsed = time.time() - start_time
    print(f"  Braking profile complete ({elapsed:.1f}s)")

    v_accel_limited = v_corner_limited.copy()
    start_time = time.time()
    
    for iter_num in range(10):
        if iter_num % 2 == 0:  # Print every 2 iterations
            elapsed = time.time() - start_time
            print(f"  Acceleration profile iteration {iter_num+1}/10... (elapsed: {elapsed:.1f}s)", end='\r')
        
        max_change = 0.0
        for i in range(total_iterations):
            idx = i % num_points
            idx_prev = (idx - 1 + num_points) % num_points
            
            ds = path['ds'][idx]
            if ds < 1e-6: continue

            v_prev = v_accel_limited[idx_prev]
            gg = get_cached_gg_limits(v_prev)
            lat_g_req = abs(v_prev**2 * path['curv'][idx] / g)

            # Apply stability margin to lateral G
            lat_g_max = gg['lat_g'] * stability_margin

            # Ensure we don't exceed the stability-limited lateral G
            if lat_g_req > lat_g_max:
                # Reduce speed to stay within stability margin
                v_max_stable = np.sqrt(lat_g_max * g / (abs(path['curv'][idx]) + 1e-6))
                v_prev = min(v_prev, v_max_stable)
                lat_g_req = lat_g_max

            # Use elliptical friction constraint instead of circular
            # (Fx/Fx_max)^2 + (Fy/Fy_max)^2 <= 1
            # Solving for Fx: Fx = Fx_max * sqrt(1 - (Fy/Fy_max)^2)
            Fx_max_accel = gg['Fx_max_accel']
            Fy_max = gg['Fy_max']
            Fy_req = lat_g_req * params['vehicle']['m_s'] * g

            # Calculate what the TC threshold allows (for pure longitudinal slip)
            ax_g_estimate = gg['accel_g']  # Initial estimate from g-g diagram
            Fx_tc_pure = calculate_tc_limited_forces(params, ax_g_estimate, 0.0, gg)  # Pure longitudinal

            # Now apply traction ellipse to TC-limited force
            # The TC limit is what we can do in pure slip; ellipse reduces it for combined slip
            Fx_accel = Fx_tc_pure * np.sqrt(max(0, 1 - (Fy_req / max(Fy_max, 1e-6))**2))

            # Also ensure we don't exceed the maximum possible (from g-g diagram)
            Fx_avail_ellipse = Fx_max_accel * np.sqrt(max(0, 1 - (Fy_req / max(Fy_max, 1e-6))**2))
            Fx_accel = min(Fx_accel, Fx_avail_ellipse)

            accel = Fx_accel / params['vehicle']['m_s']

            v_max_sq = v_prev**2 + 2 * accel * ds
            if v_max_sq > 0:
                v_new = np.sqrt(v_max_sq)
                max_change = max(max_change, abs(v_new - v_accel_limited[idx]))
                v_accel_limited[idx] = min(v_accel_limited[idx], v_new)
        
        # Early exit if converged
        if max_change < 0.01:
            print(f"  Acceleration profile converged after {iter_num+1} iterations (max change: {max_change:.4f} m/s)")
            break
    
    elapsed = time.time() - start_time
    print(f"  Acceleration profile complete ({elapsed:.1f}s)")

    v_profile = np.minimum(v_brake_limited, v_accel_limited)

    # Safety margin and smoothing to avoid exceeding theoretical friction limit
    p_driver = params.get('driver', {})
    safety_factor = p_driver.get('target_speed_safety_factor', 0.97)  # 0.97 = 3% margin by default
    smoothing_tau = p_driver.get('target_speed_smoothing_tau', 0.15)  # 0..1 (higher => smoother)

    v_profile_safe = v_profile * safety_factor

    # Circular exponential moving average smoothing for smooth lap transitions
    v_smoothed = v_profile_safe.copy()
    alpha = smoothing_tau
    n = len(v_smoothed)

    # Forward pass (wraps around to handle lap boundary)
    for i in range(n):
        prev_idx = (i - 1) % n
        v_smoothed[i] = alpha * v_smoothed[prev_idx] + (1 - alpha) * v_smoothed[i]

    # Backward pass (wraps around to reduce phase lag)
    for i in range(n-1, -1, -1):
        next_idx = (i + 1) % n
        v_smoothed[i] = alpha * v_smoothed[next_idx] + (1 - alpha) * v_smoothed[i]

    # Lap boundary margin already applied before accel/braking passes for smooth transitions

    print("Dynamic speed profile generation complete.")
    path['target_speed'] = np.clip(v_smoothed, 1.0, max_speed)


    return path
