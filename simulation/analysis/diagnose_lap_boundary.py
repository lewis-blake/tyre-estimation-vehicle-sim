#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnostic script to analyze the lap boundary and closing segments.
This helps identify why there's oversteer at the end of each lap.
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from helpers.track_tools import generate_parametric_track
from helpers.speed_profile_generator import generate_dynamic_speed_profile
import copy

def main():
    # Replicate the track from main_simulation.py
    track_segments = [
        {'type': 'straight', 'length': 300},
        {'type': 'arc', 'radius': 15, 'angle': -120},
        {'type': 'straight', 'length': 300},
        {'type': 'arc', 'radius': 35, 'angle': -90},
        {'type': 'straight', 'length': 100},
    ]

    print("="*60)
    print("LAP BOUNDARY DIAGNOSTIC")
    print("="*60)

    # Calculate expected track length from explicit segments
    explicit_length = 0
    for seg in track_segments:
        if seg['type'] == 'straight':
            explicit_length += seg['length']
        elif seg['type'] == 'arc':
            explicit_length += abs(seg['radius'] * np.deg2rad(seg['angle']))

    print(f"\nExplicit segments total length: {explicit_length:.1f} m")

    path = generate_parametric_track(track_segments, step=0.1)
    total_length = path['s'][-1]
    closing_length = total_length - explicit_length

    print(f"Total track length (with closing): {total_length:.1f} m")
    print(f"Closing segments length: {closing_length:.1f} m ({closing_length/total_length*100:.1f}% of lap)")

    # Find where closing segments start (approximately)
    closing_start_idx = np.argmin(np.abs(path['s'] - explicit_length))

    print(f"\nClosing segments start at index {closing_start_idx} / {len(path['x'])}")
    print(f"  Position: ({path['x'][closing_start_idx]:.1f}, {path['y'][closing_start_idx]:.1f})")

    # Analyze curvature in closing segments
    closing_curv = path['curv'][closing_start_idx:]
    print(f"\nClosing segments curvature:")
    print(f"  Max absolute curvature: {np.max(np.abs(closing_curv)):.6f} (1/m)")
    print(f"  Min radius in closing: {1/np.max(np.abs(closing_curv)):.1f} m")

    # Compare to explicit corners
    print(f"\nExplicit corner radii for comparison:")
    for i, seg in enumerate(track_segments):
        if seg['type'] == 'arc':
            print(f"  Corner {i+1}: radius = {seg['radius']} m")

    # Set up parameters (simplified version)
    params = {
        'vehicle': {
            'mass': 320,
            'wheelbase': 1.6,
            'track_width_f': 1.25,
            'track_width_r': 1.25,
            'COG_z': 0.2,
            'COG_ratio_x': 0.5,
            'lf': 0.8,
            'lr': 0.8,
            'ride_height_f': 0.05,
            'ride_height_r': 0.05,
            'm_s': 280,
            'max_speed_mps': 120,
            'K_tire': 200000,
            'R_eff': 0.22,
            'brake_bias': 0.63
        },
        'environment': {'g': 9.81, 'rho_air': 1.225},
        'driver': {
            'tc_throttle_sr_threshold': 0.10,
            'tc_brake_sr_threshold': 0.10,
            'target_speed_safety_factor': 0.99,
            'target_speed_smoothing_tau': 0.05,
            'speed_profile_stability_margin': 1.00,
            'lap_boundary_margin': 0.92,
            'lap_boundary_zone_fraction': 0.10
        }
    }

    # Add tyre parameters
    tyre_params_base = {
        'D_ref': 1.5, 'k_load': -0.2, 'Cref': 30.0, 'k_load_stiff': -0.6,
        'C': 1.3, 'E': -1.0, 'D_ref_lon': 1.5, 'k_load_lon': -0.2,
        'Cref_lon': 30.0, 'k_load_stiff_lon': -0.6, 'C_lon': 1.65,
        'E_lon': -1.0, 'T_opt': 60.0, 'sigma_left': 15.0,
        'sigma_right': 15.0, 'T_opt_BCD': 60.0, 'sigma_left_BCD': 80.0,
        'sigma_right_BCD': 100.0, 'T_factor_min_BCD': 0.9, 'Fz0': 1000.0,
        'deg': {
            'D_ref': {'k_time': 0.0, 'k_Fz': 0.0, 'k_alpha': 0.0, 'k_T': 0.0, 'min_val': 0.3},
            'Cref': {'k_time': 0.0, 'k_Fz': 0.0, 'k_alpha': 0.0, 'k_T': 0.0, 'min_val': 20.0},
            'D_ref_lon': {'k_time': 0.0, 'k_Fz': 0.0, 'k_kappa': 0.0, 'k_T': 0.0, 'min_val': 0.3},
            'Cref_lon': {'k_time': 0.0, 'k_Fz': 0.0, 'k_kappa': 0.0, 'k_T': 0.0, 'min_val': 20.0}
        }
    }
    params['tyre_front'] = copy.deepcopy(tyre_params_base)
    params['tyre_rear'] = copy.deepcopy(tyre_params_base)

    print("\nGenerating speed profile...")
    path = generate_dynamic_speed_profile(path, params)

    # Analyze speed profile at lap boundary
    boundary_zone_size = int(params['driver']['lap_boundary_zone_fraction'] * len(path['s']))

    print(f"\n{'='*60}")
    print(f"SPEED PROFILE ANALYSIS AT LAP BOUNDARY")
    print(f"{'='*60}")
    print(f"Boundary zone size: {boundary_zone_size} points ({params['driver']['lap_boundary_zone_fraction']*100:.0f}% of lap)")
    print(f"Boundary zone length: {boundary_zone_size * 0.1:.1f} m (approx)")

    # Speed at different locations
    print(f"\nTarget speeds around lap boundary:")
    print(f"  At start of lap (idx 0): {path['target_speed'][0]*3.6:.1f} km/h")
    print(f"  At end of lap (idx -1): {path['target_speed'][-1]*3.6:.1f} km/h")
    print(f"  Before boundary zone (idx -{boundary_zone_size}): {path['target_speed'][-boundary_zone_size]*3.6:.1f} km/h")
    print(f"  Mid closing segments (approx): {path['target_speed'][closing_start_idx + (len(path['s'])-closing_start_idx)//2]*3.6:.1f} km/h")

    # Check for discontinuities
    speed_jump_at_boundary = abs(path['target_speed'][0] - path['target_speed'][-1])
    print(f"\nSpeed discontinuity at lap boundary: {speed_jump_at_boundary*3.6:.2f} km/h")
    if speed_jump_at_boundary > 0.5:
        print("  WARNING: Large speed discontinuity detected!")

    # Plot the results
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))

    # Plot 1: Track layout with closing segments highlighted
    ax = axes[0]
    ax.plot(path['x'], path['y'], 'b-', linewidth=1, label='Track')
    ax.plot(path['x'][closing_start_idx:], path['y'][closing_start_idx:], 'r-', linewidth=2, label='Closing segments')
    ax.plot(path['x'][0], path['y'][0], 'go', markersize=10, label='Start/Finish')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Track Layout (Closing Segments in Red)')
    ax.legend()
    ax.grid(True)
    ax.axis('equal')

    # Plot 2: Curvature along track
    ax = axes[1]
    ax.plot(path['s'], path['curv'], 'b-', linewidth=1)
    ax.axvline(path['s'][closing_start_idx], color='r', linestyle='--', label='Closing segments start')
    ax.axvspan(path['s'][-boundary_zone_size], path['s'][-1], alpha=0.3, color='orange', label='Boundary zone')
    ax.set_xlabel('Distance along track (m)')
    ax.set_ylabel('Curvature (1/m)')
    ax.set_title('Track Curvature')
    ax.legend()
    ax.grid(True)

    # Plot 3: Speed profile
    ax = axes[2]
    ax.plot(path['s'], path['target_speed']*3.6, 'b-', linewidth=1)
    ax.axvline(path['s'][closing_start_idx], color='r', linestyle='--', label='Closing segments start')
    ax.axvspan(path['s'][-boundary_zone_size], path['s'][-1], alpha=0.3, color='orange', label='Boundary zone')
    ax.set_xlabel('Distance along track (m)')
    ax.set_ylabel('Target Speed (km/h)')
    ax.set_title('Speed Profile')
    ax.legend()
    ax.grid(True)

    # Plot 4: Zoom on lap boundary region
    ax = axes[3]
    # Show last 20% and first 5% of lap
    n = len(path['s'])
    boundary_indices = list(range(int(0.8*n), n)) + list(range(0, int(0.05*n)))
    boundary_distances = np.concatenate([path['s'][int(0.8*n):], path['s'][:int(0.05*n)] + path['s'][-1]])
    boundary_speeds = np.concatenate([path['target_speed'][int(0.8*n):], path['target_speed'][:int(0.05*n)]])

    ax.plot(boundary_distances, boundary_speeds*3.6, 'b-', linewidth=2)
    ax.axvline(path['s'][-1], color='g', linestyle='-', linewidth=2, label='Lap boundary')
    ax.axvspan(path['s'][-boundary_zone_size], path['s'][-1], alpha=0.3, color='orange', label='Boundary zone')
    ax.set_xlabel('Distance along track (m)')
    ax.set_ylabel('Target Speed (km/h)')
    ax.set_title('Speed Profile Zoom: Lap Boundary Region')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    diag_dir = Path(__file__).resolve().parent.parent.parent / 'results' / 'simulation'
    diag_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(diag_dir / 'lap_boundary_diagnostic.png'), dpi=150)
    print(f"\nDiagnostic plot saved to '{diag_dir / 'lap_boundary_diagnostic.png'}")

    print("\n" + "="*60)
    print("DIAGNOSTIC COMPLETE")
    print("="*60)

if __name__ == '__main__':
    main()
