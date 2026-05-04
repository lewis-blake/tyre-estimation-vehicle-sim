#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Live debug plot for vehicle simulation.

Uses the same logic as MPC live plots: plt.ion(), show(block=False), draw(), flush_events(), pause().
"""

import numpy as np
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt


def _is_interactive_backend():
    return matplotlib.get_backend().lower() not in ('agg', 'svg', 'pdf', 'ps', 'cairo')

_script_dir = Path(__file__).resolve().parent.parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))
from vehicle import tyre_interface
from vehicle import state_manager as sm

# Number of points for Pacejka curve sweep
N_ALPHA = 80
N_KAPPA = 80
ALPHA_DEG_MAX = 15.0
KAPPA_MAX = 0.25
# Update every N log steps to avoid slowing sim too much (fallback)
LIVE_UPDATE_INTERVAL = 5
# Preferred: update interval in seconds (live plot refreshes at this rate)
LIVE_UPDATE_INTERVAL_SEC = 0.2


def _pacejka_lateral_curve(tyre_params, Fz, T, default_pressure=120.0):
    """Return (alpha_deg, Fy) for lateral Pacejka at given Fz, T."""
    model_type = tyre_params.get('tyre_model_type', 'Standard')
    alpha_deg = np.linspace(-ALPHA_DEG_MAX, ALPHA_DEG_MAX, N_ALPHA)
    Fy_list = []
    for ad in alpha_deg:
        alpha_rad = np.deg2rad(ad)
        _, Fy = tyre_interface.calculate_tyre_forces(
            model_type, alpha_rad, 0.0, Fz, T, tyre_params,
            default_pressure_kPa=default_pressure
        )
        Fy_list.append(float(np.ravel(Fy)[0]))
    return alpha_deg, np.array(Fy_list)


def _pacejka_longitudinal_curve(tyre_params, Fz, T, default_pressure=120.0):
    """Return (kappa, Fx) for longitudinal Pacejka at given Fz, T."""
    model_type = tyre_params.get('tyre_model_type', 'Standard')
    kappa = np.linspace(-KAPPA_MAX, KAPPA_MAX, N_KAPPA)
    Fx_list = []
    for k in kappa:
        Fx, _ = tyre_interface.calculate_tyre_forces(
            model_type, 0.0, k, Fz, T, tyre_params,
            default_pressure_kPa=default_pressure
        )
        Fx_list.append(float(np.ravel(Fx)[0]))
    return kappa, np.array(Fx_list)


def _traction_ellipse_boundary(tyre_params, Fz, T, default_pressure=120.0, n_theta=60):
    """Return (Fx, Fy) points on the traction ellipse at current Fz (pure slip limits)."""
    model_type = tyre_params.get('tyre_model_type', 'Standard')
    Fx_max, Fy_max = tyre_interface.calculate_peak_forces(
        model_type, Fz, tyre_params, T=T
    )
    Fx_max, Fy_max = max(float(Fx_max), 1e-10), max(float(Fy_max), 1e-10)
    theta = np.linspace(0, 2 * np.pi, n_theta)
    # Ellipse: (Fx/Fx_max)^2 + (Fy/Fy_max)^2 = 1
    Fx = Fx_max * np.cos(theta)
    Fy = Fy_max * np.sin(theta)
    return Fx, Fy, float(Fx_max), float(Fy_max)


class LiveDebugPlot:
    """Live-updating debug figure. Same logic as MPC: ion(), show(block=False), draw(), flush_events(), pause()."""

    def __init__(self, params, enabled=True, save_dir=None):
        self.enabled = enabled
        self.params = params
        self.dt_log = params.get('dt_log', 0.01)
        self.default_pressure = params.get('tyre_front', {}).get('default_pressure_kPa', 120.0)
        self.tyre_f = params.get('tyre_front', {})
        self.tyre_r = params.get('tyre_rear', {})
        if not enabled:
            self.fig = None
            return
        self._build_figure()

    def _build_figure(self):
        plt.ion()
        self.fig = plt.figure(figsize=(16, 12))
        self.fig.suptitle('Live simulation debug', fontsize=12)
        gs = self.fig.add_gridspec(4, 4, hspace=0.35, wspace=0.3)
        self.ax_pos = self.fig.add_subplot(gs[0, 0])
        self.ax_vel = self.fig.add_subplot(gs[0, 1])
        self.ax_time_inputs = self.fig.add_subplot(gs[0, 2:4])
        self.ax_time_inputs.axis('off')
        self.ax_lat = [self.fig.add_subplot(gs[1, i]) for i in range(4)]
        self.ax_lon = [self.fig.add_subplot(gs[2, i]) for i in range(4)]
        self.ax_ellipse = [self.fig.add_subplot(gs[3, i]) for i in range(4)]
        if _is_interactive_backend():
            plt.show(block=False)

    def update(
        self,
        sim_time,
        log_index,
        state_dict,
        history,
        tire_history,
        tire_force_history,
        temperature_history,
        bulk_temperature_history=None,
        total_sim_time=None,
        controls=None,
    ):
        if not self.enabled or self.fig is None or log_index < 1:
            return
        if not plt.fignum_exists(self.fig.number):
            return

        n = log_index + 1
        t_hist = np.arange(n) * self.dt_log
        vx_hist = history[:n, sm.STATE_VECTOR_LAYOUT['vx_mps']['slice'].start]
        vy_hist = history[:n, sm.STATE_VECTOR_LAYOUT['vy_mps']['slice'].start]
        X_hist = history[:n, sm.STATE_VECTOR_LAYOUT['X_m']['slice'].start]
        Y_hist = history[:n, sm.STATE_VECTOR_LAYOUT['Y_m']['slice'].start]

        X = float(np.ravel(state_dict['X_m'])[0])
        Y = float(np.ravel(state_dict['Y_m'])[0])

        # tire_history: Fz(4), slip_angles(4), slip_ratios(4) = 12 cols; row index log_index-1
        idx = min(log_index - 1, tire_history.shape[0] - 1)
        if idx < 0:
            return
        fz = tire_history[idx, :4]
        slip_angles = tire_history[idx, 4:8]   # rad
        slip_ratios = tire_history[idx, 8:12]
        fx = tire_force_history[idx, :4]
        fy = tire_force_history[idx, 4:8]
        temps = temperature_history[idx, :4] if temperature_history.shape[1] >= 4 else np.full(4, 60.0)
        if bulk_temperature_history is not None and bulk_temperature_history.shape[0] > idx:
            bulk_temps = bulk_temperature_history[idx, :4]
        else:
            bulk_temps = temps.copy()

        wheel_names = ['FL', 'FR', 'RL', 'RR']
        tyre_params_per_wheel = [self.tyre_f, self.tyre_f, self.tyre_r, self.tyre_r]

        # Position
        self.ax_pos.clear()
        self.ax_pos.plot(X_hist, Y_hist, 'b-', alpha=0.7, label='Path')
        self.ax_pos.plot(X, Y, 'ro', markersize=10, label='Current')
        self.ax_pos.set_xlabel('X (m)')
        self.ax_pos.set_ylabel('Y (m)')
        self.ax_pos.set_title('Car position')
        self.ax_pos.legend(loc='upper right', fontsize=8)
        self.ax_pos.axis('equal')
        self.ax_pos.grid(True, alpha=0.3)

        # Velocity history
        self.ax_vel.clear()
        self.ax_vel.plot(t_hist, vx_hist, 'b-', label='vx')
        self.ax_vel.plot(t_hist, vy_hist, 'r-', label='vy')
        speed = np.sqrt(vx_hist**2 + vy_hist**2)
        self.ax_vel.plot(t_hist, speed, 'k--', alpha=0.7, label='|v|')
        self.ax_vel.set_xlabel('Time (s)')
        self.ax_vel.set_ylabel('Velocity (m/s)')
        self.ax_vel.set_title('Velocity history')
        self.ax_vel.legend(loc='upper right', fontsize=8)
        self.ax_vel.grid(True, alpha=0.3)

        # Time and inputs (numbers)
        self.ax_time_inputs.clear()
        self.ax_time_inputs.axis('off')
        time_str = f"t = {sim_time:.2f} s"
        if total_sim_time is not None and total_sim_time > 0:
            time_str += f" / {total_sim_time:.2f} s"
        self.ax_time_inputs.text(0.05, 0.7, time_str, transform=self.ax_time_inputs.transAxes, fontsize=12, family='monospace')
        temp_str = "Tread temps °C: FL: {:.1f}  FR: {:.1f}  RL: {:.1f}  RR: {:.1f}".format(
            temps[0], temps[1], temps[2], temps[3])
        bulk_str = "Bulk  temps °C: FL: {:.1f}  FR: {:.1f}  RL: {:.1f}  RR: {:.1f}".format(
            bulk_temps[0], bulk_temps[1], bulk_temps[2], bulk_temps[3])
        fz_str = "Fz (N):         FL: {:.0f}  FR: {:.0f}  RL: {:.0f}  RR: {:.0f}".format(
            fz[0], fz[1], fz[2], fz[3])
        self.ax_time_inputs.text(0.05, 0.55, temp_str, transform=self.ax_time_inputs.transAxes, fontsize=10, family='monospace')
        self.ax_time_inputs.text(0.05, 0.43, bulk_str, transform=self.ax_time_inputs.transAxes, fontsize=10, family='monospace')
        self.ax_time_inputs.text(0.05, 0.31, fz_str, transform=self.ax_time_inputs.transAxes, fontsize=10, family='monospace')
        if controls is not None:
            tb = controls.get('throttle_brake_cmd', 0.0)
            steer = controls.get('steer_cmd_rad', 0.0)
            tgt_v = controls.get('target_v_mps', 0.0)
            self.ax_time_inputs.text(0.05, 0.19, f"throttle_brake = {tb:.3f}   steer_rad = {steer:.4f}   target_v_mps = {tgt_v:.2f}",
                                     transform=self.ax_time_inputs.transAxes, fontsize=10, family='monospace', verticalalignment='top')

        for i in range(4):
            Fz = max(float(fz[i]), 1.0)
            T = float(temps[i])
            tp = tyre_params_per_wheel[i]

            # Lateral Pacejka: Fy vs alpha
            alpha_deg, Fy_curve = _pacejka_lateral_curve(tp, Fz, T, self.default_pressure)
            alpha_deg_cur = np.rad2deg(float(slip_angles[i]))
            Fy_cur = float(fy[i])
            # Vehicle stores Fy in body convention (negated from model); curve is model Fy. So point on curve = -Fy_cur
            Fy_on_curve = -Fy_cur

            self.ax_lat[i].clear()
            self.ax_lat[i].plot(alpha_deg, Fy_curve, 'b-', label='Fy(α)')
            self.ax_lat[i].plot(alpha_deg_cur, Fy_on_curve, 'ro', markersize=8, label='Current')
            self.ax_lat[i].axhline(0, color='k', linewidth=0.5)
            self.ax_lat[i].axvline(0, color='k', linewidth=0.5)
            self.ax_lat[i].set_xlabel('Slip angle α (deg)')
            self.ax_lat[i].set_ylabel('Fy (N)')
            self.ax_lat[i].set_title(f'{wheel_names[i]} Lateral')
            self.ax_lat[i].legend(loc='upper right', fontsize=7)
            self.ax_lat[i].grid(True, alpha=0.3)

            # Longitudinal Pacejka: Fx vs kappa
            kappa_arr, Fx_curve = _pacejka_longitudinal_curve(tp, Fz, T, self.default_pressure)
            kappa_cur = float(slip_ratios[i])
            Fx_cur = float(fx[i])

            self.ax_lon[i].clear()
            self.ax_lon[i].plot(kappa_arr, Fx_curve, 'b-', label='Fx(κ)')
            self.ax_lon[i].plot(kappa_cur, Fx_cur, 'ro', markersize=8, label='Current')
            self.ax_lon[i].axhline(0, color='k', linewidth=0.5)
            self.ax_lon[i].axvline(0, color='k', linewidth=0.5)
            self.ax_lon[i].set_xlabel('Slip ratio κ')
            self.ax_lon[i].set_ylabel('Fx (N)')
            self.ax_lon[i].set_title(f'{wheel_names[i]} Longitudinal')
            self.ax_lon[i].legend(loc='upper right', fontsize=7)
            self.ax_lon[i].grid(True, alpha=0.3)

            # Traction ellipse: Fy (x) vs Fx (y), ellipse boundary + current point
            Fx_ell, Fy_ell, Fx_max, Fy_max = _traction_ellipse_boundary(
                tp, Fz, T, self.default_pressure
            )

            self.ax_ellipse[i].clear()
            self.ax_ellipse[i].plot(Fy_ell, Fx_ell, 'b-', alpha=0.8, label='Limit')
            self.ax_ellipse[i].plot(Fy_cur, Fx_cur, 'ro', markersize=10, label='Current')
            self.ax_ellipse[i].axhline(0, color='k', linewidth=0.5)
            self.ax_ellipse[i].axvline(0, color='k', linewidth=0.5)
            self.ax_ellipse[i].set_xlabel('Fy (N)')
            self.ax_ellipse[i].set_ylabel('Fx (N)')
            self.ax_ellipse[i].set_title(f'{wheel_names[i]} Traction ellipse')
            self.ax_ellipse[i].legend(loc='upper right', fontsize=7)
            self.ax_ellipse[i].axis('equal')
            self.ax_ellipse[i].grid(True, alpha=0.3)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        if _is_interactive_backend():
            plt.pause(0.01)

    def close(self):
        if self.enabled and self.fig is not None and plt.fignum_exists(self.fig.number):
            plt.close(self.fig)
        self.fig = None
