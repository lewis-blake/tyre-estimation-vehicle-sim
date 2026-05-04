#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 18:44:06 2025

@author: lewisblake
"""

# analysis_tools.py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import sys
from pathlib import Path
# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from vehicle import state_manager as sm
from scipy.signal import find_peaks
from vehicle.vehicle_dynamics import _pacejka_magic_formula
from data_export.io_tools_export import save_full_export_json

from vehicle import tyre_interface

def calculate_and_print_suspension_analytics(params):
    """
    Calculates and prints key suspension parameters like ride frequencies,
    wheel rates, and roll stiffness based on the vehicle parameters.
    """
    _prev_backend = matplotlib.get_backend()
    try:
        matplotlib.use('Agg')  # save to file only during this function
        _run_suspension_analytics(params)
    finally:
        matplotlib.use(_prev_backend)


def _run_suspension_analytics(params):
    """Implementation of suspension analytics (runs with Agg backend)."""
    print("\n" + "="*50)
    print("= Suspension Analytics".center(48) + "=")
    print("="*50)
    
    p_v = params['vehicle']
    m_s, L, lf, lr = p_v['m_s'], p_v['wheelbase'], p_v['lf'], p_v['lr']
    T_f = p_v['track_width_f']
    T_r = p_v['track_width_r']

    # Wheel Rates
    # Wheel Rate is the effective spring rate at the wheel, accounting for motion ratio.
    # K_wheel = K_spring / MR^2
    K_w_f = p_v['K_s_f'] / (p_v['motion_ratio_f']**2)
    K_w_r = p_v['K_s_r'] / (p_v['motion_ratio_r']**2)
    print(f"Wheel Rate (Front): {K_w_f/1000:.1f} N/mm")
    print(f"Wheel Rate (Rear):  {K_w_r/1000:.1f} N/mm")
    
    # NEW: Heave Wheel Rates
    K_heave_w_f = p_v.get('K_heave_f', 0) / (p_v['motion_ratio_f']**2)
    K_heave_w_r = p_v.get('K_heave_r', 0) / (p_v['motion_ratio_r']**2)
    print(f"Heave Rate (Front): {K_heave_w_f/1000:.1f} N/mm (at wheel)")
    print(f"Heave Rate (Rear):  {K_heave_w_r/1000:.1f} N/mm (at wheel)")


    # Ride Frequencies
    # This is the undamped natural frequency of the suspension in heave/pitch.
    # f = (1 / 2*pi) * sqrt(K_effective / m_sprung_corner)
    m_s_corner_f = (m_s * lr / L) / 2.0
    m_s_corner_r = (m_s * lf / L) / 2.0
    
    # Effective heave spring rate per corner is K_w + K_heave_w
    freq_f = (1 / (2 * np.pi)) * np.sqrt((K_w_f + K_heave_w_f) / m_s_corner_f)
    freq_r = (1 / (2 * np.pi)) * np.sqrt((K_w_r + K_heave_w_r) / m_s_corner_r)
    print(f"Ride Frequency (Front): {freq_f:.2f} Hz  (Sprung Mass/Corner: {m_s_corner_f:.1f} kg)")
    print(f"Ride Frequency (Rear):  {freq_r:.2f} Hz  (Sprung Mass/Corner: {m_s_corner_r:.1f} kg)")

    # Roll Stiffness
    # This is the chassis' resistance to roll, in Nm per degree of roll.
    # It comes from the springs and the anti-roll bars. Heave springs do not contribute to roll.
    # Moment = Force * LeverArm. Stiffness = Moment / Angle.
    # K_roll_springs = 0.5 * K_wheel * TrackWidth^2 (in Nm/rad)
    K_roll_springs_f_nm_rad = 0.5 * K_w_f * T_f**2
    K_roll_springs_r_nm_rad = 0.5 * K_w_r * T_r**2
    
    # Assuming K_arb is the effective rate in Nm/rad of chassis roll.
    # This is a simplification; a more detailed model would use bar geometry.
    K_arb_f_nm_rad = p_v['K_arb_f'] 
    K_arb_r_nm_rad = p_v['K_arb_r'] 
    
    K_roll_total_f_nm_rad = K_roll_springs_f_nm_rad + K_arb_f_nm_rad
    K_roll_total_r_nm_rad = K_roll_springs_r_nm_rad + K_arb_r_nm_rad
    K_roll_total_nm_rad = K_roll_total_f_nm_rad + K_roll_total_r_nm_rad
    
    # Include torsional stiffness in analytics
    K_torsion_nm_rad = p_v.get('K_torsion_Nm_rad', 0)
    print(f"Chassis Torsional Stiffness: {K_torsion_nm_rad / 1000:.1f} kNm/rad")
    # End Modification

    # Convert from Nm/rad to Nm/deg for easier interpretation
    K_roll_total_nm_deg = K_roll_total_nm_rad * (np.pi / 180.0)
    
    if K_roll_total_nm_rad > 0:
        roll_dist_f_pct = (K_roll_total_f_nm_rad / K_roll_total_nm_rad) * 100
        roll_dist_r_pct = (K_roll_total_r_nm_rad / K_roll_total_nm_rad) * 100
    else:
        roll_dist_f_pct = 50.0 # Avoid division by zero
        roll_dist_r_pct = 50.0
        
    # Calculate LLTD (Lateral Load Transfer Distribution)
    # LLTD is the fraction of lateral load transfer going to each axle
    # LLTD_f = fraction going to front axle, LLTD_r = fraction going to rear axle
    # This is related to roll stiffness distribution
    # Higher roll stiffness on an axle means more load transfer goes to that axle
    if K_roll_total_nm_rad > 0:
        LLTD_f = K_roll_total_f_nm_rad / K_roll_total_nm_rad
        LLTD_r = K_roll_total_r_nm_rad / K_roll_total_nm_rad
    else:
        LLTD_f = 0.5
        LLTD_r = 0.5
        
    print(f"Total Suspension Roll Stiffness: {K_roll_total_nm_deg:.1f} Nm/deg")
    print(f"Suspension Roll Dist:  {roll_dist_f_pct:.1f}% Front, {roll_dist_r_pct:.1f}% Rear")
    print(f"LLTD (Lateral Load Transfer Distribution):")
    print(f"  LLTD_f (Front): {LLTD_f:.4f} ({LLTD_f*100:.2f}%)")
    print(f"  LLTD_r (Rear):  {LLTD_r:.4f} ({LLTD_r*100:.2f}%)")
    print("="*50 + "\n")
    
    # Store LLTD in params for reference
    params['vehicle']['LLTD_f'] = LLTD_f
    params['vehicle']['LLTD_r'] = LLTD_r


def plot_simulation_results(state_history, control_history, force_history, tire_history,
                            suspension_history, tire_force_history, camber_history, 
                            jacking_force_history, diff_history, path, params,
                            temperature_history=None, r_dot_history=None, show_track_path=True,
                            output_dir=None):
    """
    Generates a comprehensive dashboard of plots to analyze the simulation run.
    This version creates three separate figures for clarity.
    """
    _prev_backend = matplotlib.get_backend()
    try:
        matplotlib.use('Agg')
        _do_plot_simulation_results(state_history, control_history, force_history, tire_history,
            suspension_history, tire_force_history, camber_history, jacking_force_history,
            diff_history, path, params, temperature_history, r_dot_history, show_track_path)
    finally:
        matplotlib.use(_prev_backend)


def _do_plot_simulation_results(state_history, control_history, force_history, tire_history,
                                suspension_history, tire_force_history, camber_history,
                                jacking_force_history, diff_history, path, params,
                                temperature_history=None, r_dot_history=None, show_track_path=True):
    print("Generating analysis plots...")
    
    dt = params['dt_log']
    
    num_rows = state_history.shape[0]
    if num_rows < 2:
        print("Not enough data to plot.")
        return

    # 1. Create a clean DataFrame for easier data access
    df_state = pd.DataFrame(state_history)
    df_state.columns = [f'state_{i}' for i in range(df_state.shape[1])]
    
    named_cols = {}
    for name, info in sm.STATE_VECTOR_LAYOUT.items():
        s = info['slice']
        if info['size'] == 1:
            named_cols[f'state_{s.start}'] = name
        else:
            for i in range(info['size']):
                named_cols[f'state_{s.start + i}'] = f"{name}_{i}"
    df_state.rename(columns=named_cols, inplace=True)

    df_controls = pd.DataFrame(control_history, columns=['throttle_brake_cmd', 'steer_cmd_rad', 'target_v_mps'])
    df_forces = pd.DataFrame(force_history, columns=['Sum_Fx', 'Sum_Fy', 'Sum_Fz', 'Sum_Mz'])
    
    tire_cols = ['Fz_FL', 'Fz_FR', 'Fz_RL', 'Fz_RR', 'SA_FL_rad', 'SA_FR_rad', 'SA_RL_rad', 'SA_RR_rad', 'SR_FL', 'SR_FR', 'SR_RL', 'SR_RR']
    df_tires = pd.DataFrame(tire_history, columns=tire_cols)
    
    # Added heave force columns
    susp_cols = ['spring_F_FL','spring_F_FR','spring_F_RL','spring_F_RR', 
                 'damper_F_FL','damper_F_FR','damper_F_RL','damper_F_RR', 
                 'arb_F_FL','arb_F_FR','arb_F_RL','arb_F_RR',
                 'heave_F_f', 'heave_F_r',
                 'bump_F_FL', 'bump_F_FR', 'bump_F_RL', 'bump_F_RR']
    df_susp = pd.DataFrame(suspension_history, columns=susp_cols)
    
    tire_force_cols = ['Fx_FL','Fx_FR','Fx_RL','Fx_RR', 'Fy_FL','Fy_FR','Fy_RL','Fy_RR']
    df_tire_forces = pd.DataFrame(tire_force_history, columns=tire_force_cols)

    camber_cols = ['camber_FL_rad', 'camber_FR_rad', 'camber_RL_rad', 'camber_RR_rad']
    df_camber = pd.DataFrame(camber_history, columns=camber_cols)

    # Added jacking force DataFrame
    jacking_cols = ['jacking_F_FL', 'jacking_F_FR', 'jacking_F_RL', 'jacking_F_RR']
    df_jacking = pd.DataFrame(jacking_force_history, columns=jacking_cols)

    # Added differential torque DataFrame
    diff_cols = ['Tq_drive_RL', 'Tq_drive_RR', 'Tq_diff_transfer', 'lock_factor']
    df_diff = pd.DataFrame(diff_history, columns=diff_cols)

    # Added temperature DataFrame
    if temperature_history is not None:
        temp_cols = ['T_FL', 'T_FR', 'T_RL', 'T_RR']
        df_temp = pd.DataFrame(temperature_history, columns=temp_cols)
    else:
        df_temp = pd.DataFrame()
    
    # Added r_dot DataFrame
    if r_dot_history is not None:
        df_r_dot = pd.DataFrame({'r_dot_radps': r_dot_history})
    else:
        df_r_dot = pd.DataFrame()
    
    # Concatenate all DataFrames
    dfs_to_concat = [df_state, df_controls, df_forces, df_tires, df_susp, df_tire_forces, df_camber, df_jacking, df_diff]
    if not df_temp.empty:
        dfs_to_concat.append(df_temp)
    if not df_r_dot.empty:
        dfs_to_concat.append(df_r_dot)
    df = pd.concat(dfs_to_concat, axis=1)
    
    final_sim_time = (len(df) - 1) * dt
    df['time_s'] = np.linspace(0, final_sim_time, len(df))

    # 2. Calculate additional useful channels
    p_v = params['vehicle']
    g = params['environment']['g']
    mass, h_cg = p_v['mass'], p_v['COG_z']
    lf, lr, L = p_v['lf'], p_v['lr'], p_v['wheelbase']
    T_f, T_r = p_v['track_width_f'], p_v['track_width_r']
    mr_f, mr_r = p_v['motion_ratio_f'], p_v['motion_ratio_r']
    mrr_f, mrr_r = p_v['motion_ratio_rate_f'], p_v['motion_ratio_rate_r']
    
    df['ax_g'] = df['Sum_Fx'] / mass / g
    df['ay_g'] = df['Sum_Fy'] / mass / g
    df['ax_mps2'] = df['ax_g'] * g
    df['ay_mps2'] = df['ay_g'] * g
    
    df['speed_error'] = df['target_v_mps'] - df['vx_mps']
    df['sideslip_rad'] = np.arctan2(df['vy_mps'], df['vx_mps'] + 1e-6)
    
    # Chassis corner Z calculation now uses front and rear roll angles
    df['chassis_Z_FL'] = df['Z_m'] - p_v['COG_z'] - lf * np.sin(df['theta_rad']) - (T_f/2) * np.sin(df['phi_f_rad'])
    df['chassis_Z_FR'] = df['Z_m'] - p_v['COG_z'] - lf * np.sin(df['theta_rad']) + (T_f/2) * np.sin(df['phi_f_rad'])
    df['chassis_Z_RL'] = df['Z_m'] - p_v['COG_z'] + lr * np.sin(df['theta_rad']) - (T_r/2) * np.sin(df['phi_r_rad'])
    df['chassis_Z_RR'] = df['Z_m'] - p_v['COG_z'] + lr * np.sin(df['theta_rad']) + (T_r/2) * np.sin(df['phi_r_rad'])

    df['wheel_travel_FL_mm'] = (p_v['suspension_static_length_f_m'] - (df['wheel_Z_m_0'] - df['chassis_Z_FL'])) * 1000
    df['wheel_travel_FR_mm'] = (p_v['suspension_static_length_f_m'] - (df['wheel_Z_m_1'] - df['chassis_Z_FR'])) * 1000
    df['wheel_travel_RL_mm'] = (p_v['suspension_static_length_r_m'] - (df['wheel_Z_m_2'] - df['chassis_Z_RL'])) * 1000
    df['wheel_travel_RR_mm'] = (p_v['suspension_static_length_r_m'] - (df['wheel_Z_m_3'] - df['chassis_Z_RR'])) * 1000


    approx_shock_FL = df['wheel_travel_FL_mm'] / (1000 * mr_f)
    current_mr_FL = mr_f * (1 + mrr_f * approx_shock_FL)
    df['shock_travel_FL_mm'] = -df['wheel_travel_FL_mm'] / current_mr_FL

    approx_shock_FR = df['wheel_travel_FR_mm'] / (1000 * mr_f)
    current_mr_FR = mr_f * (1 + mrr_f * approx_shock_FR)
    df['shock_travel_FR_mm'] = -df['wheel_travel_FR_mm'] / current_mr_FR

    approx_shock_RL = df['wheel_travel_RL_mm'] / (1000 * mr_r)
    current_mr_RL = mr_r * (1 + mrr_r * approx_shock_RL)
    df['shock_travel_RL_mm'] = -df['wheel_travel_RL_mm'] / current_mr_RL
    
    approx_shock_RR = df['wheel_travel_RR_mm'] / (1000 * mr_r)
    current_mr_RR = mr_r * (1 + mrr_r * approx_shock_RR)
    df['shock_travel_RR_mm'] = -df['wheel_travel_RR_mm'] / current_mr_RR

    # Ride Height Calculation
    df['ride_height_f_mm'] = ((df['chassis_Z_FL'] + df['chassis_Z_FR']) / 2.0) * 1000
    df['ride_height_r_mm'] = ((df['chassis_Z_RL'] + df['chassis_Z_RR']) / 2.0) * 1000
    
    # Load Transfer Calculation
    # Theoretical load transfer
    
    Z = h_cg + (df['ride_height_f_mm'] + df['ride_height_f_mm']) / (2 * 1000)
    
    df['LT_lat_theory'] = (df['ay_mps2'] * mass * Z) / ((T_f + T_r) / 2.0)
    df['LT_long_theory'] = - ((df['ax_mps2'] * Z * mass)) / L
    
    # Sprung load transfer (from spring, damper, ARB, heave)
    total_susp_F_FL = (df['spring_F_FL'] + df['damper_F_FL'] + df['arb_F_FL'] + df['heave_F_f'] + df['bump_F_FL']) / mr_f
    total_susp_F_FR = (df['spring_F_FR'] + df['damper_F_FR'] + df['arb_F_FR'] + df['heave_F_f'] + df['bump_F_FR']) / mr_f
    total_susp_F_RL = (df['spring_F_RL'] + df['damper_F_RL'] + df['arb_F_RL'] + df['heave_F_r'] + df['bump_F_RL']) / mr_r
    total_susp_F_RR = (df['spring_F_RR'] + df['damper_F_RR'] + df['arb_F_RR'] + df['heave_F_r'] + df['bump_F_RR']) / mr_r
    
    df['LT_lat_sprung'] = 0.5 * ((total_susp_F_FR + total_susp_F_RR) - (total_susp_F_FL + total_susp_F_RL))
    df['LT_long_sprung'] = 0.5 * ((total_susp_F_FL + total_susp_F_FR) - (total_susp_F_RL + total_susp_F_RR) + mass * g * (lf-lr)/L) 
    
    # Instantaneous/Jacking load transfer
    df['LT_lat_jacking'] = 0.5 * ((df['jacking_F_FR'] + df['jacking_F_RR']) - (df['jacking_F_FL'] + df['jacking_F_RL']))
    df['LT_long_jacking'] = 0.5 * ((df['jacking_F_FL'] + df['jacking_F_FR']) - (df['jacking_F_RL'] + df['jacking_F_RR']))

    # 3. Generate the plot dashboard
    
    project_root = Path(__file__).parent.parent.parent
    if output_dir is None:
        output_dir = project_root / 'data' / 'plots'
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Figure 1: Performance & Chassis Dynamics
    fig1 = plt.figure(figsize=(16, 21))
    fig1.suptitle('Figure 1: Performance & Chassis Dynamics', fontsize=20, y=0.99)
    gs1 = fig1.add_gridspec(7, 2)

    ax1 = fig1.add_subplot(gs1[0, 0])
    sc = ax1.scatter(df['X_m'], df['Y_m'], c=df['vx_mps'] * 3.6, cmap='viridis', s=10)
    if show_track_path:
        ax1.plot(path['x'], path['y'], 'k--', alpha=0.5, linewidth=0.8)
    cbar = plt.colorbar(sc, ax=ax1); cbar.set_label('Speed (km/h)')
    ax1.set_title('Vehicle Trajectory & Speed' if not show_track_path else 'Track Path & Vehicle Speed')
    ax1.axis('equal'); ax1.grid(True)
    ax1.set_xlabel('X (m)'); ax1.set_ylabel('Y (m)')
    
    ax2 = fig1.add_subplot(gs1[0, 1])
    ax2.scatter(df['ay_g'], df['ax_g'], s=5, alpha=0.5, label='Achieved G')
    max_ax = df['ax_g'].max(); min_ax = df['ax_g'].min()
    max_ay = df['ay_g'].abs().max()
    ax2.axhline(max_ax, color='g', linestyle='--', label=f'Max Accel G ({max_ax:.2f})')
    ax2.axhline(min_ax, color='r', linestyle='--', label=f'Max Brake G ({min_ax:.2f})')
    ax2.axvline(max_ay, color='b', linestyle='--', label=f'Max Lat G ({max_ay:.2f})')
    ax2.axvline(-max_ay, color='b', linestyle='--')
    ax2.set_title('G-G Diagram (Achieved Envelope)'); ax2.grid(True)
    ax2.set_xlabel('Lateral Accel (g)'); ax2.set_ylabel('Longitudinal Accel (g)')
    ax2.set_xlim(-max_ay*1.1, max_ay*1.1); ax2.set_ylim(min_ax*1.1, max_ax*1.1)
    ax2.axis('equal'); ax2.legend(fontsize='small')

    ax3 = fig1.add_subplot(gs1[1, :])
    ax3.plot(df['time_s'], df['vx_mps'] * 3.6, label='Actual'); ax3.plot(df['time_s'], df['target_v_mps'] * 3.6, 'r--', label='Target')
    ax3.grid(True); ax3.set_title('Speed Profile'); ax3.legend(); ax3.set_xlabel('Time (s)'); ax3.set_ylabel('Speed (km/h)')

    ax4 = fig1.add_subplot(gs1[2, :])
    ax4.plot(df['time_s'], df['throttle_brake_cmd'], label='Throttle/Brake', c='g'); ax4_twin = ax4.twinx()
    ax4_twin.plot(df['time_s'], np.rad2deg(df['steer_cmd_rad']), label='Steer', c='purple')
    ax4.set_title('Driver Inputs'); ax4.grid(True); ax4.set_xlabel('Time (s)'); ax4.set_ylabel('Throttle/Brake Cmd', color='g')
    ax4_twin.set_ylabel('Steer Angle (deg)', color='purple'); ax4.legend(loc='upper left'); ax4_twin.legend(loc='upper right')

    ax5 = fig1.add_subplot(gs1[3, :])
    ax5.plot(df['time_s'], np.rad2deg(df['phi_f_rad']), label='Front Roll'); ax5.plot(df['time_s'], np.rad2deg(df['phi_r_rad']), label='Rear Roll', linestyle='--')
    ax5.plot(df['time_s'], np.rad2deg(df['theta_rad']), label='Pitch', alpha=0.7)
    ax5.grid(True); ax5.set_title('Chassis Attitude'); ax5.legend(); ax5.set_xlabel('Time (s)'); ax5.set_ylabel('Angle (deg)')
    
    ax6 = fig1.add_subplot(gs1[4, :])
    ax6.plot(df['time_s'], np.rad2deg(df['r_radps']), label='Yaw Rate'); ax6_twin = ax6.twinx()
    ax6_twin.plot(df['time_s'], np.rad2deg(df['sideslip_rad']), 'r--', label='Sideslip Angle')
    ax6.grid(True); ax6.set_title('Yaw Rate & Sideslip'); ax6.set_xlabel('Time (s)'); ax6.set_ylabel('Yaw Rate (deg/s)')
    ax6_twin.set_ylabel('Sideslip Angle (deg)', color='r'); ax6.legend(loc='upper left'); ax6_twin.legend(loc='upper right')
    
    ax7 = fig1.add_subplot(gs1[5, :])
    ax7.plot(df['time_s'], df['Z_m']); ax7.grid(True); ax7.set_title('Chassis Heave (CoG Z Position)')
    ax7.set_xlabel('Time (s)'); ax7.set_ylabel('Heave Z (m)')

    # NEW: Ride Height Plot
    ax8 = fig1.add_subplot(gs1[6, :])
    ax8.plot(df['time_s'], df['ride_height_f_mm'], label='Front Ride Height')
    ax8.plot(df['time_s'], df['ride_height_r_mm'], label='Rear Ride Height')
    ax8.grid(True); ax8.set_title('Axle Ride Height (Chassis to Ground)'); ax8.legend()
    ax8.set_xlabel('Time (s)'); ax8.set_ylabel('Height (mm)')

    fig1.tight_layout(rect=(0, 0, 1, 0.98))
    
    # Save Figure 1
    fig1.savefig(output_dir / '01_performance_chassis_dynamics.png', dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print(f"Saved: {output_dir / '01_performance_chassis_dynamics.png'}")

    # FIGURE 2: SUSPension ANALYSIS
    fig2 = plt.figure(figsize=(16, 20))
    fig2.suptitle('Figure 2: Suspension Analysis', fontsize=20, y=0.98)
    gs2 = fig2.add_gridspec(7, 2) 

    ax2_1 = fig2.add_subplot(gs2[0, :]); ax2_1.set_title('Suspension Forces: Spring')
    ax2_1.plot(df['time_s'], df['spring_F_FL'], label='FL'); ax2_1.plot(df['time_s'], df['spring_F_FR'], label='FR')
    ax2_1.plot(df['time_s'], df['spring_F_RL'], label='RL'); ax2_1.plot(df['time_s'], df['spring_F_RR'], label='RR')
    ax2_1.grid(True); ax2_1.set_ylabel('Force (N)'); ax2_1.legend()

    ax2_2 = fig2.add_subplot(gs2[1, :]); ax2_2.set_title('Suspension Forces: Damper')
    ax2_2.plot(df['time_s'], df['damper_F_FL'], label='FL'); ax2_2.plot(df['time_s'], df['damper_F_FR'], label='FR')
    ax2_2.plot(df['time_s'], df['damper_F_RL'], label='RL'); ax2_2.plot(df['time_s'], df['damper_F_RR'], label='RR')
    ax2_2.grid(True); ax2_2.set_ylabel('Force (N)'); ax2_2.legend()
    
    ax2_3 = fig2.add_subplot(gs2[2, :]); ax2_3.set_title('Suspension Forces: Anti-Roll Bar')
    ax2_3.plot(df['time_s'], df['arb_F_FL'], label='FL'); ax2_3.plot(df['time_s'], df['arb_F_FR'], label='FR')
    ax2_3.plot(df['time_s'], df['arb_F_RL'], label='RL'); ax2_3.plot(df['time_s'], df['arb_F_RR'], label='RR')
    ax2_3.grid(True); ax2_3.set_ylabel('Force (N)'); ax2_3.legend()

    # NEW: Heave Spring Force Plot
    ax2_4 = fig2.add_subplot(gs2[3, :]); ax2_4.set_title('Suspension Forces: Heave Spring (at Axle)')
    ax2_4.plot(df['time_s'], df['heave_F_f'], label='Front')
    ax2_4.plot(df['time_s'], df['heave_F_r'], label='Rear')
    ax2_4.grid(True); ax2_4.set_xlabel('Time (s)'); ax2_4.set_ylabel('Force (N)'); ax2_4.legend()

    ax2_5 = fig2.add_subplot(gs2[4, :]); ax2_5.set_title('Shock Velocity Histogram')
    all_shock_vels = pd.concat([df[f'wheel_Z_dot_mps_{i}'] for i in range(4)])
    ax2_5.hist(all_shock_vels, bins=50); ax2_5.grid(True); ax2_5.set_xlabel('Shock Velocity (m/s)'); ax2_5.set_ylabel('Counts')

    ax2_6 = fig2.add_subplot(gs2[5, :])
    ax2_6.plot(df['time_s'], df['shock_travel_FL_mm'], label='FL')
    ax2_6.plot(df['time_s'], df['shock_travel_FR_mm'], label='FR')
    ax2_6.plot(df['time_s'], df['shock_travel_RL_mm'], label='RL')
    ax2_6.plot(df['time_s'], df['shock_travel_RR_mm'], label='RR')
    
    max_stroke_mm = p_v['shock_stroke_max_m'] * 1000
    max_rebound_mm = -p_v.get('shock_stroke_rebound_max_m', p_v['shock_stroke_max_m']) * 1000
    ax2_6.axhline(max_stroke_mm, color='r', linestyle='--', linewidth=1.5, label='Bump Stop')
    ax2_6.axhline(max_rebound_mm, color='b', linestyle='--', linewidth=1.5, label='Rebound Stop')
    
    for col in ['shock_travel_FL_mm', 'shock_travel_FR_mm', 'shock_travel_RL_mm', 'shock_travel_RR_mm']:
        hit_bump = df[df[col] >= max_stroke_mm]
        ax2_6.scatter(hit_bump['time_s'], hit_bump[col], color='red', s=20, zorder=5)
        hit_rebound = df[df[col] <= max_rebound_mm]
        ax2_6.scatter(hit_rebound['time_s'], hit_rebound[col], color='blue', s=20, zorder=5)

    ax2_6.grid(True); ax2_6.set_title('Shock Travel (Jounce/Rebound)');
    ax2_6.set_xlabel('Time (s)'); ax2_6.set_ylabel('Shock Travel (mm)')
    ax2_6.legend()
    
    ax2_7 = fig2.add_subplot(gs2[6, :])
    ax2_7.plot(df['time_s'], df['wheel_travel_FL_mm'], label='FL')
    ax2_7.plot(df['time_s'], df['wheel_travel_FR_mm'], label='FR')
    ax2_7.plot(df['time_s'], df['wheel_travel_RL_mm'], label='RL')
    ax2_7.plot(df['time_s'], df['wheel_travel_RR_mm'], label='RR')
    ax2_7.grid(True); ax2_7.set_title('Wheel Travel (Jounce/Rebound)');
    ax2_7.set_xlabel('Time (s)'); ax8.set_ylabel('Travel (mm)')
    ax2_7.legend()

    fig2.tight_layout(rect=(0, 0, 1, 0.97))
    
    # Save Figure 2
    fig2.savefig(output_dir / '02_suspension_analysis.png', dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"Saved: {output_dir / '02_suspension_analysis.png'}")

    # Figure 3: Load Transfer & Tire Analysis
    fig3 = plt.figure(figsize=(16, 34)) # Increased figure height
    fig3.suptitle('Figure 3: Load Transfer & Tire Analysis', fontsize=20, y=0.98)
    gs3 = fig3.add_gridspec(9, 1) # Increased rows from 8 to 9

    # Load Transfer Plots
    ax3_1 = fig3.add_subplot(gs3[0, 0]); ax3_1.set_title('Lateral Load Transfer (Right minus Left)')
    ax3_1.plot(df['time_s'], df['LT_lat_theory'], 'k--', label='Theoretical')
    ax3_1.plot(df['time_s'], df['LT_lat_sprung'], label='Actual (Sprung)')
    ax3_1.plot(df['time_s'], df['LT_lat_jacking'], label='Actual (Jacking/Instant)')
    ax3_1.grid(True); ax3_1.set_ylabel('Load Transfer (N)'); ax3_1.legend()

    ax3_2 = fig3.add_subplot(gs3[1, 0]); ax3_2.set_title('Longitudinal Load Transfer (Front minus Rear)')
    ax3_2.plot(df['time_s'], df['LT_long_theory'], 'k--', label='Theoretical')
    ax3_2.plot(df['time_s'], df['LT_long_sprung'], label='Actual (Sprung)')
    ax3_2.plot(df['time_s'], df['LT_long_jacking'], label='Actual (Jacking/Instant)')
    ax3_2.grid(True); ax3_2.set_ylabel('Load Transfer (N)'); ax3_2.legend()
    
    ax3_3 = fig3.add_subplot(gs3[2, 0]); ax3_3.set_title('Vertical Tire Loads (Fz)')
    ax3_3.plot(df['time_s'], df['Fz_FL'], label='FL'); ax3_3.plot(df['time_s'], df['Fz_FR'], label='FR')
    ax3_3.plot(df['time_s'], df['Fz_RL'], label='RL'); ax3_3.plot(df['time_s'], df['Fz_RR'], label='RR')
    ax3_3.axhline(0, color='k', linestyle='--', linewidth=1)
    ax3_3.grid(True); ax3_3.set_ylabel('Force (N)'); ax3_3.legend()
    
    ax3_4 = fig3.add_subplot(gs3[3, 0]); ax3_4.set_title('Tire Forces: Longitudinal (Fx)')
    ax3_4.plot(df['time_s'], df['Fx_FL'], label='FL'); ax3_4.plot(df['time_s'], df['Fx_FR'], label='FR')
    ax3_4.plot(df['time_s'], df['Fx_RL'], label='RL'); ax3_4.plot(df['time_s'], df['Fx_RR'], label='RR')
    ax3_4.grid(True); ax3_4.set_ylabel('Force (N)'); ax3_4.legend()
    
    ax3_5 = fig3.add_subplot(gs3[4, 0]); ax3_5.set_title('Tire Forces: Lateral (Fy)')
    ax3_5.plot(df['time_s'], df['Fy_FL'], label='FL'); ax3_5.plot(df['time_s'], df['Fy_FR'], label='FR')
    ax3_5.plot(df['time_s'], df['Fy_RL'], label='RL'); ax3_5.plot(df['time_s'], df['Fy_RR'], label='RR')
    ax3_5.grid(True); ax3_5.set_ylabel('Force (N)'); ax3_5.legend()
    
    ax3_6 = fig3.add_subplot(gs3[5, 0]); ax3_6.set_title('Dynamic Wheel Camber Angle')
    ax3_6.plot(df['time_s'], np.rad2deg(df['camber_FL_rad']), label='FL'); ax3_6.plot(df['time_s'], np.rad2deg(df['camber_FR_rad']), label='FR')
    ax3_6.plot(df['time_s'], np.rad2deg(df['camber_RL_rad']), label='RL'); ax3_6.plot(df['time_s'], np.rad2deg(df['camber_RR_rad']), label='RR')
    ax3_6.axhline(0, color='k', linestyle='--', linewidth=1)
    ax3_6.grid(True); ax3_6.set_xlabel('Time (s)'); ax3_6.set_ylabel('Camber (deg)'); ax3_6.legend()
    
    # NEW: Differential Torque Plot
    ax3_7 = fig3.add_subplot(gs3[6, 0])
    ax3_7.set_title(f'Rear Axle Drive Torque & Differential Action (K={50.0})')
    ax3_7.plot(df['time_s'], df['Tq_drive_RL'], label='Torque RL (Nm)')
    ax3_7.plot(df['time_s'], df['Tq_drive_RR'], label='Torque RR (Nm)')
    ax3_7.grid(True)
    ax3_7.set_xlabel('Time (s)')
    ax3_7.set_ylabel('Drive Torque (Nm)')
    
    ax3_7_twin = ax3_7.twinx()
    ax3_7_twin.plot(df['time_s'], df['Tq_diff_transfer'], 'r--', alpha=0.7, label='Torque Transferred (Nm)')
    ax3_7_twin.set_ylabel('Locking Torque (Nm)', color='r')
    
    lines, labels = ax3_7.get_legend_handles_labels()
    lines2, labels2 = ax3_7_twin.get_legend_handles_labels()
    ax3_7.legend(lines + lines2, labels + labels2, loc='best', fontsize='small')

    
    # Slip Ratio Plot
    ax3_8 = fig3.add_subplot(gs3[7, 0]); 
    ax3_8.set_title('Tire Slip Ratios & Wheel Lock/Spin')
    
    # Plot slip ratios for each wheel
    ax3_8.plot(df['time_s'], df['SR_FL'], label='FL', linewidth=1)
    ax3_8.plot(df['time_s'], df['SR_FR'], label='FR', linewidth=1)
    ax3_8.plot(df['time_s'], df['SR_RL'], label='RL', linewidth=1)
    ax3_8.plot(df['time_s'], df['SR_RR'], label='RR', linewidth=1)
    
    # Define thresholds
    spin_threshold = 0.15
    lock_threshold = -0.15
    ax3_8.axhline(spin_threshold, color='r', linestyle='--', linewidth=1, label=f'Spin Threshold ({spin_threshold})')
    ax3_8.axhline(lock_threshold, color='b', linestyle='--', linewidth=1, label=f'Lock Threshold ({lock_threshold})')
    
    # Highlight areas of wheel spin (red) and lock (blue)
    ax3_8.fill_between(df['time_s'], spin_threshold, df['SR_FL'], where=(df['SR_FL'] > spin_threshold).values, color='red', alpha=0.3, interpolate=True)  # type: ignore
    ax3_8.fill_between(df['time_s'], lock_threshold, df['SR_FL'], where=(df['SR_FL'] < lock_threshold).values, color='blue', alpha=0.3, interpolate=True)  # type: ignore
    ax3_8.fill_between(df['time_s'], spin_threshold, df['SR_FR'], where=(df['SR_FR'] > spin_threshold).values, color='red', alpha=0.3, interpolate=True)  # type: ignore
    ax3_8.fill_between(df['time_s'], lock_threshold, df['SR_FR'], where=(df['SR_FR'] < lock_threshold).values, color='blue', alpha=0.3, interpolate=True)  # type: ignore
    ax3_8.fill_between(df['time_s'], spin_threshold, df['SR_RL'], where=(df['SR_RL'] > spin_threshold).values, color='red', alpha=0.3, interpolate=True)  # type: ignore
    ax3_8.fill_between(df['time_s'], lock_threshold, df['SR_RL'], where=(df['SR_RL'] < lock_threshold).values, color='blue', alpha=0.3, interpolate=True)  # type: ignore
    ax3_8.fill_between(df['time_s'], spin_threshold, df['SR_RR'], where=(df['SR_RR'] > spin_threshold).values, color='red', alpha=0.3, interpolate=True)  # type: ignore
    ax3_8.fill_between(df['time_s'], lock_threshold, df['SR_RR'], where=(df['SR_RR'] < lock_threshold).values, color='blue', alpha=0.3, interpolate=True)  # type: ignore

    ax3_8.grid(True)
    ax3_8.set_xlabel('Time (s)')
    ax3_8.set_ylabel('Slip Ratio')
    ax3_8.legend(fontsize='small')
    ax3_8.set_ylim(-1.1, 1.1) # Set reasonable limits for slip ratio


    # Slip Angle Plot
    ax3_9 = fig3.add_subplot(gs3[8, 0]); 
    ax3_9.set_title('Tire Slip Angles')
    
    # Plot slip angles for each wheel, converting from radians to degrees
    ax3_9.plot(df['time_s'], np.rad2deg(df['SA_FL_rad']), label='FL', linewidth=1)
    ax3_9.plot(df['time_s'], np.rad2deg(df['SA_FR_rad']), label='FR', linewidth=1)
    ax3_9.plot(df['time_s'], np.rad2deg(df['SA_RL_rad']), label='RL', linewidth=1)
    ax3_9.plot(df['time_s'], np.rad2deg(df['SA_RR_rad']), label='RR', linewidth=1)

    ax3_9.axhline(0, color='k', linestyle='--', linewidth=1)
    ax3_9.grid(True)
    ax3_9.set_xlabel('Time (s)')
    ax3_9.set_ylabel('Slip Angle (deg)')
    ax3_9.legend(fontsize='small')

    fig3.tight_layout(rect=(0, 0, 1, 0.97))
    
    # Save Figure 3
    fig3.savefig(output_dir / '03_load_transfer_tire_analysis.png', dpi=150, bbox_inches='tight')
    plt.close(fig3)
    print(f"Saved: {output_dir / '03_load_transfer_tire_analysis.png'}")

    # Figure 4: Corner Balance Analysis
    fig4, (ax4_slip, ax4_steer) = plt.subplots(2, 1, figsize=(16, 18))
    fig4.suptitle('Figure 4: Corner Balance Analysis', fontsize=20, y=0.98)

    # Shared Calculation For Both Plots
    # 1. Calculate Understeer Angle (for top plot)
    df['understeer_angle_deg'] = np.rad2deg(
        (df['SA_FL_rad'] + df['SA_FR_rad']) / 2 - 
        (df['SA_RL_rad'] + df['SA_RR_rad']) / 2
    )

    # 2. Calculate Steer Angle Difference (for bottom plot)
    turn_radius = df['vx_mps'] / (df['r_radps'] + 1e-9)
    df['neutral_steer_rad'] = np.arctan(p_v['wheelbase'] / turn_radius)
    df['steer_angle_difference_deg'] = np.rad2deg(df['neutral_steer_rad'] - df['steer_cmd_rad'])

    # 3. Find vehicle's path progress 's' for each time step
    try:
        path_kdtree = path['_kdtree']
        points = np.vstack((np.asarray(df['X_m'].values), np.asarray(df['Y_m'].values))).T  # type: ignore
        _, indices = path_kdtree.query(points, k=1)
        df['path_s'] = path['s'][indices]
    except Exception as e:
        print(f"Could not map vehicle position to path, skipping Figure 4. Error: {e}")
        return

    # 4. Detect corners from track curvature
    curv_abs = np.abs(path['curv'])
    apexes, _ = find_peaks(curv_abs, prominence=0.005, distance=50)

    if len(apexes) > 0:
        corner_data_slip = []
        corner_data_steer = []
        curv_threshold = 0.002 

        in_corner = curv_abs > curv_threshold
        corner_change = np.diff(in_corner.astype(int))
        corner_starts = np.where(corner_change == 1)[0]
        corner_ends = np.where(corner_change == -1)[0]
        
        if len(corner_starts) > 0 and len(corner_ends) > 0:
            if corner_ends[0] < corner_starts[0]: corner_starts = np.insert(corner_starts, 0, 0)
            if corner_starts[-1] > corner_ends[-1]: corner_ends = np.append(corner_ends, len(curv_abs) - 1)
            
            min_len = min(len(corner_starts), len(corner_ends))
            corner_starts, corner_ends = corner_starts[:min_len], corner_ends[:min_len]

            valid_corners = []
            for start, end in zip(corner_starts, corner_ends):
                if end - start > 20 and any((apex >= start) and (apex <= end) for apex in apexes):
                     valid_corners.append({'start_s': path['s'][start], 'end_s': path['s'][end]})
                     
            apex_overlay_data = []
            
            corner_list = []

            for i, corner in enumerate(valid_corners):
                start_s, end_s = corner['start_s'], corner['end_s']
                corner_df = df[(df['path_s'] >= start_s) & (df['path_s'] <= end_s)]
                if len(corner_df) < 6: continue
                
                split_indices = np.array_split(corner_df.index, 3)
                entry_indices, mid_indices, exit_indices = split_indices[0], split_indices[1], split_indices[2]

                # Data for Slip Angle plot
                avg_us_entry = df.loc[entry_indices, 'understeer_angle_deg'].mean()
                avg_us_mid = df.loc[mid_indices, 'understeer_angle_deg'].mean()
                avg_us_exit = df.loc[exit_indices, 'understeer_angle_deg'].mean()
                corner_data_slip.append({'corner': i + 1, 'phase': 'Entry', 'us_angle': avg_us_entry})
                corner_data_slip.append({'corner': i + 1, 'phase': 'Mid', 'us_angle': avg_us_mid})
                corner_data_slip.append({'corner': i + 1, 'phase': 'Exit', 'us_angle': avg_us_exit})

                # Data for Steer Angle plot
                avg_steer_diff_entry = df.loc[entry_indices, 'steer_angle_difference_deg'].mean()
                avg_steer_diff_mid = df.loc[mid_indices, 'steer_angle_difference_deg'].mean()
                avg_steer_diff_exit = df.loc[exit_indices, 'steer_angle_difference_deg'].mean()
                corner_data_steer.append({'corner': i + 1, 'phase': 'Entry', 'steer_diff': avg_steer_diff_entry})
                corner_data_steer.append({'corner': i + 1, 'phase': 'Mid', 'steer_diff': avg_steer_diff_mid})
                corner_data_steer.append({'corner': i + 1, 'phase': 'Exit', 'steer_diff': avg_steer_diff_exit})
                
                corner_df = df.loc[entry_indices]
                peak_sr_entry_RL = corner_df['SR_RL'].min()
                peak_sr_entry_RR = corner_df['SR_RR'].min()
                peak_sr_entry_FL = corner_df['SR_FL'].min()
                peak_sr_entry_FR = corner_df['SR_FR'].min()
                apex_overlay_data.append({
                    'label': f'T{i+1}',  # e.g. T1, T2 …
                    'peak_sr_entry_RL': peak_sr_entry_RL,
                    'peak_sr_entry_RR': peak_sr_entry_RR,
                    'peak_sr_entry_FL': peak_sr_entry_FL,
                    'peak_sr_entry_FR': peak_sr_entry_FR
                })
                
                corner_df = df.loc[mid_indices]
                peak_sa_deg_FL = np.rad2deg(corner_df['SA_FL_rad']).max()
                peak_sa_deg_FR = np.rad2deg(corner_df['SA_FR_rad']).max()
                peak_sa_deg_RL = np.rad2deg(corner_df['SA_RL_rad']).max()
                peak_sa_deg_RR = np.rad2deg(corner_df['SA_RR_rad']).max()
                apex_overlay_data.append({
                    'label': f'T{i+1}',  # e.g. T1, T2 …
                    'peak_sa_deg_FL': peak_sa_deg_FL,
                    'peak_sa_deg_FR': peak_sa_deg_FR,
                    'peak_sa_deg_RL': peak_sa_deg_RL,
                    'peak_sa_deg_RR': peak_sa_deg_RR
                })
                
                corner_df = df.loc[exit_indices]
                peak_sr_exit_RL = corner_df['SR_RL'].max()
                peak_sr_exit_RR = corner_df['SR_RR'].max()
                apex_overlay_data.append({
                    'label': f'T{i+1}',  # e.g. T1, T2 …
                    'peak_sr_exit_RL': peak_sr_exit_RL,
                    'peak_sr_exit_RR': peak_sr_exit_RR
                })
                
                corner_list.append({
                    'id': i,
                    'start_idx': entry_indices[0],
                    'end_idx': exit_indices[-1],
                    'entry_indices': entry_indices,
                    'mid_indices': mid_indices,   
                    'exit_indices': exit_indices            
                    })

            # Subplot 1: Slip Angle Method
            if corner_data_slip:
                plot_df_slip = pd.DataFrame(corner_data_slip)
                bar_labels = [f"C{row.corner}\n{row.phase}" for _, row in plot_df_slip.iterrows()]
                colors = ['red' if x < 0 else 'royalblue' for x in plot_df_slip['us_angle']]
                bars = ax4_slip.bar(bar_labels, plot_df_slip['us_angle'], color=colors)
                ax4_slip.axhline(0, color='k', lw=1); ax4_slip.grid(True, axis='y', ls='--', alpha=0.7)
                ax4_slip.set_ylabel('Avg. Understeer Angle (deg)'); ax4_slip.set_title('Balance from Slip Angle Difference (Vehicle State)')
                for bar in bars:
                    yval = bar.get_height()
                    if not np.isnan(yval): ax4_slip.text(bar.get_x() + bar.get_width()/2.0, yval + np.sign(yval)*0.05, f'{yval:.2f}', ha='center', va='bottom' if yval >= 0 else 'top')
                ax4_slip.text(0.98, 0.98, 'Understeer', transform=ax4_slip.transAxes, va='top', ha='right', color='royalblue', weight='bold')
                ax4_slip.text(0.98, 0.02, 'Oversteer', transform=ax4_slip.transAxes, va='bottom', ha='right', color='red', weight='bold')

            # Subplot 2: Steer Angle Method
            if corner_data_steer:
                plot_df_steer = pd.DataFrame(corner_data_steer)
                bar_labels = [f"C{row.corner}\n{row.phase}" for _, row in plot_df_steer.iterrows()]
                colors = ['red' if x < 0 else 'royalblue' for x in plot_df_steer['steer_diff']]
                bars = ax4_steer.bar(bar_labels, plot_df_steer['steer_diff'], color=colors)
                ax4_steer.axhline(0, color='k', lw=1); ax4_steer.grid(True, axis='y', ls='--', alpha=0.7)
                ax4_steer.set_ylabel('Avg. Steer Angle Difference (deg)'); ax4_steer.set_title('Balance from Steer Angle Difference (Driver Compensation)')
                for bar in bars:
                    yval = bar.get_height()
                    if not np.isnan(yval): ax4_steer.text(bar.get_x() + bar.get_width()/2.0, yval + np.sign(yval)*0.05, f'{yval:.2f}', ha='center', va='bottom' if yval >= 0 else 'top')
                ax4_steer.text(0.98, 0.98, 'Correcting for Understeer', transform=ax4_steer.transAxes, va='top', ha='right', color='royalblue', weight='bold')
                ax4_steer.text(0.98, 0.02, 'Correcting for Oversteer', transform=ax4_steer.transAxes, va='bottom', ha='right', color='red', weight='bold')
        else:
            ax4_slip.text(0.5, 0.5, 'Could not detect distinct corners.', ha='center', va='center')
            ax4_steer.text(0.5, 0.5, 'Could not detect distinct corners.', ha='center', va='center')
    else:
        ax4_slip.text(0.5, 0.5, 'No corners were detected on the track.', ha='center', va='center')
        ax4_steer.text(0.5, 0.5, 'No corners were detected on the track.', ha='center', va='center')

    fig4.tight_layout(rect=(0, 0, 1, 0.96))
    
    # Save Figure 4
    fig4.savefig(output_dir / '04_corner_balance_analysis.png', dpi=150, bbox_inches='tight')
    plt.close(fig4)
    print(f"Saved: {output_dir / '04_corner_balance_analysis.png'}")

    # Save sim_full.json to results directory
    save_full_export_json(str(output_dir / 'sim_full.json'), df, corner_list, params,
                          per_tyre_time_series=None,
                          simulation_meta={'dt': dt, 'run_id': 'run001'})

    # Also save to data/synthetic/ for the estimator pipeline
    canonical_data_dir = project_root / 'data' / 'synthetic'
    canonical_data_dir.mkdir(parents=True, exist_ok=True)
    save_full_export_json(str(canonical_data_dir / 'sim_full.json'), df, corner_list, params,
                          per_tyre_time_series=None,
                          simulation_meta={'dt': dt, 'run_id': 'run001'})

    

    
    def _get_param(d, key, default=0.0):
        """Safe param read."""
        return d.get(key, default)
    
    
    # Plotting wrapper that uses above Pacejka functions

    def plot_normalised_tire_curves_from_params(params, df=None, apex_overlay_data=None,
                                                tyre='FL', load_max=None, load_avg=None, load_min=None,
                                                kappa_range=None, alpha_range=None,
                                                n_points=300, camber_deg=0.0):
        """
        Build μx and μy curves using Pacejka params in params['tyre_front'] / ['tyre_rear'].
        - x-axis for lateral plot is slip angle in DEGREES.
        - df: optional DataFrame for heatmap / hist overlays (same format as your analysis code).
        - apex_data: optional list of apex dicts with 'label', 'peak_sa_deg', 'peak_sr'.
        """

        if tyre == 'FL' or tyre == 'FR':
            tyre_key = 'tyre_front'
        else:
            tyre_key = 'tyre_rear'
        
        tyre_params = params.get(tyre_key, {})


        # slip grids
        if kappa_range is None:
            kappa_range = np.linspace(-0.5, 0.5, n_points)
        if alpha_range is None:
            alpha_range = np.linspace(-0.175, 0.175, n_points)  # rad
        alpha_deg = np.rad2deg(alpha_range)

        # Extract Fz0 and temperature (tyre_params already in new format)
        Fz0 = tyre_params.get('Fz0', 1000.0)
        T = tyre_params.get('T_opt', 60.0)  # Use optimal temperature for plotting

        # create figure
        fig, (ax_lon, ax_lat) = plt.subplots(2, 1, figsize=(10, 11))
        fig.suptitle(f'{tyre.capitalize()} Tyre: μ vs Slip (Magic Formula 5.2)', fontsize=14)
        
        loads_N = [load_max,load_avg,load_min]

        # compute and plot curves for each load
        for Fz in loads_N:
            if Fz is None:
                continue
            
            # Use tyre_interface for calculations
            model_type = tyre_params.get('tyre_model_type', 'Standard')
            default_pressure = tyre_params.get('default_pressure_kPa', 120.0)

            # Calculate Longitudinal Force (alpha=0)
            Fx, _ = tyre_interface.calculate_tyre_forces(
                model_type, 0.0, kappa_range, Fz, T, tyre_params, default_pressure_kPa=default_pressure
            )

            # Calculate Lateral Force (kappa=0)
            # Alpha needs to be passed in radians to interface (it handles conversion if needed)
            _, Fy = tyre_interface.calculate_tyre_forces(
                model_type, alpha_range, 0.0, Fz, T, tyre_params, default_pressure_kPa=default_pressure
            )
            mu_x = Fx / (Fz + 1e-12)
            mu_y = Fy / (Fz + 1e-12)
            ax_lon.plot(kappa_range, mu_x, label=f'Fz={Fz:.0f} N', linewidth=1.4, zorder=10)
            ax_lat.plot(alpha_deg, mu_y, label=f'Fz={Fz:.0f} N', linewidth=1.4, zorder=10)

        # axis labels (lateral in degrees)
        ax_lon.set_title('Longitudinal μx = Fx / Fz')
        ax_lon.set_xlabel('Slip ratio κ')
        ax_lon.set_ylabel('μx')
        ax_lon.grid(True)

        ax_lat.set_title('Lateral μy = Fy / Fz')
        ax_lat.set_xlabel('Slip angle (deg)')
        ax_lat.set_ylabel('μy')
        ax_lat.grid(True)


        # Add apex markers (if present)
        if apex_overlay_data is not None and len(apex_overlay_data) > 0:
            # We'll produce two separate ordered lists for lateral peaks and longitudinal (entry/exit).
            lateral_apexes = []
            long_entry_apexes = []
            long_exit_apexes = []
            long_single_apexes = []
    
            # preserve the original list order (assumed to be corner order)
            for apex in apex_overlay_data:
                label = apex.get('label', None)
                # lateral
                if f'peak_sa_deg_{tyre}' in apex and apex[f'peak_sa_deg_{tyre}'] is not None:
                    lateral_apexes.append({'label': label, 'x': apex[f'peak_sa_deg_{tyre}']})
                # longitudinal entry/exit or single
                # check various possible keys (be robust)
                if f'peak_sr_entry_{tyre}' in apex and apex[f'peak_sr_entry_{tyre}'] is not None:
                    long_entry_apexes.append({'label': label, 'x': apex[f'peak_sr_entry_{tyre}']})
                if f'peak_sr_exit_{tyre}' in apex and apex[f'peak_sr_exit_{tyre}'] is not None:
                    long_exit_apexes.append({'label': label, 'x': apex[f'peak_sr_exit_{tyre}']})
                if f'peak_sr_{tyre}' in apex and apex[f'peak_sr_{tyre}'] is not None and \
                   (f'peak_sr_entry_{tyre}' not in apex and f'peak_sr_exit_{tyre}' not in apex):
                    long_single_apexes.append({'label': label, 'x': apex[f'peak_sr_{tyre}']})
    
            # compute y anchors: spacing them evenly in axis vertical span (bottom->top) in same order
            def compute_anchors(ax, n):
                y_min, y_max = ax.get_ylim()
                # leave small padding inside axis
                pad = 0.06 * (y_max - y_min)
                if n <= 0:
                    return []
                # evenly spaced excluding extremes
                anchors = np.linspace(y_min + pad, y_max - pad, n)
                return anchors
    
            # lateral anchors
            lat_anchors = compute_anchors(ax_lat, len(lateral_apexes))
            for i, ap in enumerate(lateral_apexes):
                y = lat_anchors[i]
                x = ap['x']
                lbl = ap.get('label', '')
                ax_lat.scatter(x, y, color='red', s=64, zorder=60, edgecolor='white', linewidth=0.8)
                ax_lat.text(x, y, f' {lbl}', color='red', fontsize=9, weight='bold', zorder=70, va='center', ha='left')
    
            # longitudinal anchors: combine entry then exit then single, each group in order to avoid overlap
            n_total_long = len(long_entry_apexes) + len(long_exit_apexes) + len(long_single_apexes)
            long_anchors = compute_anchors(ax_lon, n_total_long)
            idx = 0
            # entries (blue)
            for ap in long_entry_apexes:
                y = long_anchors[idx]; idx += 1
                x = ap['x']; lbl = ap.get('label', '')
                ax_lon.scatter(x, y, color='blue', s=64, zorder=60, edgecolor='white', linewidth=0.8)
                ax_lon.text(x, y, f' {lbl}-E', color='blue', fontsize=9, weight='bold', zorder=70, va='center', ha='left')
            # exits (green)
            for ap in long_exit_apexes:
                y = long_anchors[idx]; idx += 1
                x = ap['x']; lbl = ap.get('label', '')
                ax_lon.scatter(x, y, color='green', s=64, zorder=60, edgecolor='white', linewidth=0.8)
                ax_lon.text(x, y, f' {lbl}-X', color='green', fontsize=9, weight='bold', zorder=70, va='center', ha='left')
            # single (red)
            for ap in long_single_apexes:
                y = long_anchors[idx]; idx += 1
                x = ap['x']; lbl = ap.get('label', '')
                ax_lon.scatter(x, y, color='red', s=64, zorder=60, edgecolor='white', linewidth=0.8)
                ax_lon.text(x, y, f' {lbl}', color='red', fontsize=9, weight='bold', zorder=70, va='center', ha='left')
    
            # final layout
            ax_lon.set_ylim(ax_lon.get_ylim())  # ensure anchors are inside ranges
            ax_lat.set_ylim(ax_lat.get_ylim())
        
        fig.tight_layout()
        ax_lon.legend(fontsize='small', loc='upper left')
        ax_lat.legend(fontsize='small', loc='upper left')

        return fig
    
    load_max_fl = df['Fz_FL'].max()
    load_max_fr = df['Fz_FR'].max()
    load_max_rl = df['Fz_RL'].max()
    load_max_rr = df['Fz_RR'].max()
    
    load_avg_fl = df['Fz_FL'].mean()
    load_avg_fr = df['Fz_FR'].mean()
    load_avg_rl = df['Fz_RL'].mean()
    load_avg_rr = df['Fz_RR'].mean()
    
    load_min_fl = np.percentile((df['Fz_FL']),25)
    load_min_fr = np.percentile((df['Fz_FR']),25)
    load_min_rl = np.percentile((df['Fz_RL']),25)
    load_min_rr = np.percentile((df['Fz_RR']),25)
    

    
    # Plot Pacejka curves for each wheel with corner slip angles/ratios overlaid
    print("\nGenerating Pacejka curve plots with corner data...")
    for wheel in ['FL', 'FR', 'RL', 'RR']:
        if wheel == 'FL':
            load_max, load_avg, load_min = load_max_fl, load_avg_fl, load_min_fl
        elif wheel == 'FR':
            load_max, load_avg, load_min = load_max_fr, load_avg_fr, load_min_fr
        elif wheel == 'RL':
            load_max, load_avg, load_min = load_max_rl, load_avg_rl, load_min_rl
        else:  # RR
            load_max, load_avg, load_min = load_max_rr, load_avg_rr, load_min_rr
        
        fig_tyre = plot_normalised_tire_curves_from_params(
            params, df=df, apex_overlay_data=apex_overlay_data, 
            tyre=wheel, load_max=load_max, load_avg=load_avg, load_min=load_min
        )
        
        # Save the plot
        output_filename = f'05_pacejka_curves_{wheel}.png'
        fig_tyre.savefig(output_dir / output_filename, dpi=150, bbox_inches='tight')
        plt.close(fig_tyre)
        print(f"Saved: {output_dir / output_filename}")
    



