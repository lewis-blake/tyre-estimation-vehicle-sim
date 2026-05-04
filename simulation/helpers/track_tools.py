#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 18:42:29 2025

@author: lewisblake
"""

# track_tools.py
import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d

def _generate_straight(seg, pos, heading, step, x_pts, y_pts):
    """Helper to generate points for a straight segment."""
    length = seg['length']
    direction = np.array([np.cos(heading), np.sin(heading)])
    num_pts = max(2, int(np.floor(length / step)))
    for _ in range(num_pts):
        pos += step * direction
        x_pts.append(pos[0])
        y_pts.append(pos[1])
    return pos, heading

def _generate_arc(seg, pos, heading, step, x_pts, y_pts):
    """Helper to generate points for an arc segment."""
    radius = seg['radius']
    angle_rad = np.deg2rad(seg['angle'])
    turn_sign = np.sign(angle_rad)

    direction = np.array([np.cos(heading), np.sin(heading)])
    normal = turn_sign * np.array([-direction[1], direction[0]])
    
    center = pos + radius * normal
    arc_len = abs(radius * angle_rad)
    num_arc_pts = max(2, int(np.floor(arc_len / step)))
    
    thetas = np.linspace(0, angle_rad, num_arc_pts)
    start_angle_to_center = np.arctan2(pos[1] - center[1], pos[0] - center[0])

    arc_pt = pos
    for i in range(1, num_arc_pts):
        current_angle = start_angle_to_center + thetas[i]
        arc_pt = center + radius * np.array([np.cos(current_angle), np.sin(current_angle)])
        x_pts.append(arc_pt[0])
        y_pts.append(arc_pt[1])
    
    pos = np.copy(arc_pt)
    heading += angle_rad
    return pos, heading

def _calculate_closing_path_segments(p1, h1, p0, h0, R):
    """
    Calculates the segments for a continuous-curvature path.
    """
    # RSR
    n1_R = np.array([np.sin(h1), -np.cos(h1)])
    n0_R = np.array([np.sin(h0), -np.cos(h0)])
    c1_R = p1 + R * n1_R
    c0_R = p0 + R * n0_R
    v_R = c0_R - c1_R
    dist_R = np.linalg.norm(v_R)
    
    # LSL
    n1_L = np.array([-np.sin(h1), np.cos(h1)])
    n0_L = np.array([-np.sin(h0), np.cos(h0)])
    c1_L = p1 + R * n1_L
    c0_L = p0 + R * n0_L
    v_L = c0_L - c1_L
    dist_L = np.linalg.norm(v_L)

    # RSR path
    v_norm_R = v_R / dist_R
    v_perp_R = np.array([-v_norm_R[1], v_norm_R[0]])
    t1_R = c1_R + R * v_perp_R
    t0_R = c0_R + R * v_perp_R
    
    h_straight_R = np.arctan2(t0_R[1] - t1_R[1], t0_R[0] - t1_R[0])
    angle1_R = (h_straight_R - h1 + np.pi) % (2 * np.pi) - np.pi
    if angle1_R > 0: angle1_R -= 2*np.pi
    angle0_R = (h0 - h_straight_R + np.pi) % (2 * np.pi) - np.pi
    if angle0_R > 0: angle0_R -= 2*np.pi
    len_R = abs(angle1_R*R) + np.linalg.norm(t0_R - t1_R) + abs(angle0_R*R)

    # LSL path
    v_norm_L = v_L / dist_L
    v_perp_L = np.array([v_norm_L[1], -v_norm_L[0]])
    t1_L = c1_L + R * v_perp_L
    t0_L = c0_L + R * v_perp_L
    
    h_straight_L = np.arctan2(t0_L[1] - t1_L[1], t0_L[0] - t1_L[0])
    angle1_L = (h_straight_L - h1 + np.pi) % (2 * np.pi) - np.pi
    if angle1_L < 0: angle1_L += 2*np.pi
    angle0_L = (h0 - h_straight_L + np.pi) % (2 * np.pi) - np.pi
    if angle0_L < 0: angle0_L += 2*np.pi
    len_L = abs(angle1_L*R) + np.linalg.norm(t0_L - t1_L) + abs(angle0_L*R)
    
    if len_R <= len_L:
        return [
            {'type': 'arc', 'radius': R, 'angle': np.rad2deg(angle1_R)},
            {'type': 'straight', 'length': np.linalg.norm(t0_R - t1_R)},
            {'type': 'arc', 'radius': R, 'angle': np.rad2deg(angle0_R)},
        ]
    else:
        return [
            {'type': 'arc', 'radius': R, 'angle': np.rad2deg(angle1_L)},
            {'type': 'straight', 'length': np.linalg.norm(t0_L - t1_L)},
            {'type': 'arc', 'radius': R, 'angle': np.rad2deg(angle0_L)},
        ]

def generate_parametric_track(segments, step):
    """
    Generates a track centerline from a list of segments.
    """
    x_pts, y_pts = [0.0], [0.0]
    pos = np.array([0.0, 0.0])
    heading = np.deg2rad(90)

    for seg in segments:
        if seg['type'] == 'straight':
            pos, heading = _generate_straight(seg, pos, heading, step, x_pts, y_pts)
        elif seg['type'] == 'arc':
            pos, heading = _generate_arc(seg, pos, heading, step, x_pts, y_pts)

    start_pos = np.array([0.0, 0.0])
    start_heading = np.deg2rad(90)
    closing_radius = 25.0  # Uniform radius across all corners
    
    closing_segments = _calculate_closing_path_segments(
        pos, heading, start_pos, start_heading, closing_radius
    )

    for seg in closing_segments:
        if seg['type'] == 'straight':
            pos, heading = _generate_straight(seg, pos, heading, step, x_pts, y_pts)
        elif seg['type'] == 'arc':
            pos, heading = _generate_arc(seg, pos, heading, step, x_pts, y_pts)

    x, y = np.array(x_pts), np.array(y_pts)

    # Force perfect closure by making last point equal to first
    # This ensures smooth wraparound for lap simulation
    x[-1] = x[0]
    y[-1] = y[0]

    # Calculate derivatives with periodic boundary conditions for smooth lap transition
    # Pad the arrays to handle wraparound smoothly
    n_pad = 10  # Number of points to pad at each end
    x_padded = np.concatenate([x[-n_pad-1:-1], x, x[1:n_pad+1]])
    y_padded = np.concatenate([y[-n_pad-1:-1], y, y[1:n_pad+1]])

    dx_padded, dy_padded = np.gradient(x_padded), np.gradient(y_padded)
    d2x_padded, d2y_padded = np.gradient(dx_padded), np.gradient(dy_padded)

    # Extract the center portion (remove padding)
    dx = dx_padded[n_pad:-n_pad]
    dy = dy_padded[n_pad:-n_pad]
    d2x = d2x_padded[n_pad:-n_pad]
    d2y = d2y_padded[n_pad:-n_pad]

    ds = np.sqrt(dx**2 + dy**2)
    s = np.cumsum(ds)
    heading_raw = np.arctan2(dy, dx)
    heading_unwrapped = np.unwrap(heading_raw)

    # Heading adjustment for circular tracks removed to preserve physical winding
    # heading_unwrapped is already continuous and correct
    pass

    curvature = (dx * d2y - dy * d2x) / ((dx**2 + dy**2)**1.5 + 1e-9)

    # Apply circular smoothing to entire curvature array for continuity
    # Use multiple passes of moving average with wraparound
    # Increase smoothing passes significantly to ensure derivative continuity
    curvature_smoothed = curvature.copy()
    n_smooth_passes = 20  # Much more aggressive smoothing for derivative continuity
    for _ in range(n_smooth_passes):
        curvature_temp = curvature_smoothed.copy()
        for i in range(len(curvature)):
            idx_prev = (i - 1) % len(curvature)
            idx_next = (i + 1) % len(curvature)
            curvature_temp[i] = 0.25 * curvature_smoothed[idx_prev] + 0.5 * curvature_smoothed[i] + 0.25 * curvature_smoothed[idx_next]
        curvature_smoothed = curvature_temp

    # Force exact continuity at lap boundary (first and last point must match)
    curvature_smoothed[-1] = curvature_smoothed[0]

    curvature = curvature_smoothed

    # Calculate curvature derivative with periodic boundaries
    # Pad the curvature array to handle wraparound correctly
    n_pad_curv = 5
    curv_padded = np.concatenate([curvature[-n_pad_curv:], curvature, curvature[:n_pad_curv]])
    dkappa_padded = np.gradient(curv_padded)
    dkappa = dkappa_padded[n_pad_curv:-n_pad_curv]

    # Apply circular smoothing to curvature derivative for additional continuity
    dkappa_smoothed = dkappa.copy()
    for _ in range(10):  # Additional smoothing passes for derivative
        dkappa_temp = dkappa_smoothed.copy()
        for i in range(len(dkappa)):
            idx_prev = (i - 1) % len(dkappa)
            idx_next = (i + 1) % len(dkappa)
            dkappa_temp[i] = 0.25 * dkappa_smoothed[idx_prev] + 0.5 * dkappa_smoothed[i] + 0.25 * dkappa_smoothed[idx_next]
        dkappa_smoothed = dkappa_temp

    # Force perfect derivative continuity at lap boundary
    # Average the derivatives at start and end
    avg_derivative = 0.5 * (dkappa_smoothed[0] + dkappa_smoothed[-1])
    dkappa_smoothed[0] = avg_derivative
    dkappa_smoothed[-1] = avg_derivative

    dkappa = dkappa_smoothed

    # Diagnostic: Check continuity at lap boundary
    print("\n=== Lap Boundary Continuity Check ===")
    print(f"Position gap: {np.sqrt((x[-1]-x[0])**2 + (y[-1]-y[0])**2):.6f} m")
    print(f"Heading jump: {abs(heading_unwrapped[-1] - heading_unwrapped[0]):.6f} rad ({np.rad2deg(abs(heading_unwrapped[-1] - heading_unwrapped[0])):.3f}°)")
    print(f"Curvature jump: {abs(curvature[-1] - curvature[0]):.6f}")
    print(f"Curvature derivative at boundary:")
    print(f"  dκ/ds[-3]: {dkappa[-3]:.6f}")
    print(f"  dκ/ds[-2]: {dkappa[-2]:.6f}")
    print(f"  dκ/ds[-1]: {dkappa[-1]:.6f}")
    print(f"  dκ/ds[0]:  {dkappa[0]:.6f}")
    print(f"  dκ/ds[1]:  {dkappa[1]:.6f}")
    print(f"  dκ/ds[2]:  {dkappa[2]:.6f}")
    print(f"Curvature derivative jump: {abs(dkappa[-1] - dkappa[0]):.6f}")
    print("=" * 40 + "\n")

    z_centerline, width_L, width_R, banking = _define_surface_profile(s)

    z_interp = interp1d(s, z_centerline, kind='linear', fill_value='extrapolate', bounds_error=False)  # type: ignore

    return {
        'x': x, 'y': y, 'z_centerline': z_centerline, 'ds': ds, 's': s,
        'curv': curvature, 'heading': heading_unwrapped, 'track_width_L': width_L,
        'track_width_R': width_R, 'banking_angle': banking, 'z_interp': z_interp
    }

def _define_surface_profile(s):
    """
    Defines the 3D characteristics of the track.
    """
    num_points = len(s)
    z_centerline = 0 * np.sin(s / 20) + 0 * np.cos(s/5)
    track_width_L = np.full(num_points, 7.5)
    track_width_R = np.full(num_points, 7.5)
    banking_angle = np.zeros(num_points)
    return z_centerline, track_width_L, track_width_R, banking_angle

def find_closest_point_on_path(x_car, y_car, path):
    """
    Finds the closest point on the centerline and the signed cross-track error.
    """
    if '_kdtree' not in path:
        path['_kdtree'] = cKDTree(np.vstack((path['x'], path['y'])).T)

    query_point = [np.ravel(x_car)[0], np.ravel(y_car)[0]]
    dist, idx = path['_kdtree'].query(query_point, k=1)
    
    idx_plus_1 = (idx + 1) % len(path['x'])
    
    path_dx = path['x'][idx_plus_1] - path['x'][idx]
    path_dy = path['y'][idx_plus_1] - path['y'][idx]
    car_dx = query_point[0] - path['x'][idx]
    car_dy = query_point[1] - path['y'][idx]
    
    cross_product_sign = np.sign(path_dx * car_dy - path_dy * car_dx)
    
    return dist * cross_product_sign, idx

def get_track_elevation_at_points(wheel_coords_world, path):
    """
    Calculates the track surface elevation (z) under each of the four tires.
    """
    z_track = np.zeros(4)
    for i in range(4):
        _, idx = find_closest_point_on_path(wheel_coords_world[i, 0], wheel_coords_world[i, 1], path)
        s_val = path['s'][idx]
        z_track[i] = path['z_interp'](s_val)
    return z_track
