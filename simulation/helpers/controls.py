#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 18:43:11 2025

@author: lewisblake
"""

# controls.py
import numpy as np
import sys
from pathlib import Path
# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from helpers.track_tools import find_closest_point_on_path

class PIDController:
    """A standard PID controller with integral clamping."""
    def __init__(self, Kp, Ki, Kd, min_output, max_output):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.min_output, self.max_output = min_output, max_output
        self.integral = 0
        self.prev_error = 0
        self.max_integral = 100 # Anti-windup limit
        self.min_integral = -100

    def update(self, error, dt):
        """PID update with conditional integration anti-windup.

        - Only commit the integral update when the tentative output is not
          saturating, or if the integral would help move the output away
          from the saturating bound.
        """
        # Proportional
        P = self.Kp * error

        # Derivative (simple filtered)
        derivative = 0.0
        if dt > 1e-6:
            raw_deriv = (error - self.prev_error) / dt
            # very small 1st-order filter to reduce derivative noise
            tau = 0.02
            alpha = dt / (tau + dt)
            derivative = alpha * raw_deriv + (1.0 - alpha) * getattr(self, 'prev_derivative', 0.0)
            self.prev_derivative = derivative
        D = self.Kd * derivative

        # Tentative integral (do not commit yet)
        potential_integral = self.integral + error * dt
        potential_integral = np.clip(potential_integral, self.min_integral, self.max_integral)
        I_candidate = self.Ki * potential_integral

        # Tentative (unclipped) output using I_candidate
        tentative_output = P + I_candidate + D

        # Check clipping against PID output limits
        clipped = False
        tentative_output_clipped = np.clip(tentative_output, self.min_output, self.max_output)
        if tentative_output != tentative_output_clipped:
            clipped = True

        # Anti-windup decision:
        # - If not clipped, accept the integral update.
        # - If clipped, only accept the integral if it would move the output away
        # from the saturating bound. For example, if clipped to min_output (negative)
        # and error is positive (which would increase output toward zero), we can allow it.
        if (not clipped) or (
            (tentative_output_clipped == self.min_output and error > 0) or
            (tentative_output_clipped == self.max_output and error < 0)
        ):
            self.integral = potential_integral

        # Compute I (using whatever integral state we now have)
        I = self.Ki * self.integral

        output = P + I + D
        self.prev_error = error
        return np.clip(output, self.min_output, self.max_output)


class DriverModel:
    """
    Determines throttle and steering commands to follow a target speed profile.
    Uses a PID for speed control and a Pure Pursuit controller for steering.
    """
    def __init__(self, params):
        self.params = params
        self.path = params['path']
        self.throttle_pid = PIDController(Kp=0.3, Ki=0.1, Kd=0.03, min_output=-1.0, max_output=1.0)
        self.last_steer_cmd = 0.0
        self.last_r = 0.0

    def get_controls(self, state_dict, dt_control):
        """Calculates and returns the throttle and steering commands.
        Added simple human-style traction control (TC) and brake-lock mitigation.
        """

        vx = np.ravel(state_dict['vx_mps'])[0]

        # Find the target velocity from the pre-calculated speed profile
        _, current_path_idx = find_closest_point_on_path(
            state_dict['X_m'], state_dict['Y_m'], self.path
        )
        target_velocity = self.path['target_speed'][current_path_idx]

        velocity_error = target_velocity - vx
        throttle_brake_cmd = self.throttle_pid.update(velocity_error, dt_control)
        
        # if not hasattr(self, 'last_throttle_sign'):
        # self.last_throttle_sign = np.sign(throttle_brake_cmd)
        
        # current_sign = np.sign(throttle_brake_cmd)
        
        # # Detect switch between acceleration (positive) and braking (negative)
        # if self.last_throttle_sign != 0 and current_sign != 0 and self.last_throttle_sign != current_sign:
        # # Reset integral when switching between accel and brake
        # self.throttle_pid.integral = 0.0
        
        # self.last_throttle_sign = current_sign

        target_steer = self._get_pure_pursuit_steer(state_dict)
        
        # Yaw Damping / Stability Control (ESP)
        # Instead of just strictly damping r_dot, we damp the ERROR between actual yaw rate 
        # and the "kinematically intended" yaw rate from the pure pursuit command.
        if self.params.get('driver', {}).get('use_yaw_damping', False):
            kp_stability = self.params.get('driver', {}).get('yaw_damping_gain', 0.5)
            
            # 1. Calculate desired yaw rate based on target steer (Kinematic model: r = v/L * tan(delta))
            # This represents "where the driver/pure-pursuit wants to go"
            L = self.params['vehicle']['wheelbase']
            # Limit vx to avoid singularity or unstable small speed behavior
            vx_safe = max(vx, 1.0) 
            r_target = (vx_safe / L) * np.tan(target_steer)
            
            # 2. Get actual yaw rate
            r_actual = np.ravel(state_dict['r_radps'])[0]
            
            # 3. Calculate yaw rate error (Oversteer = r_actual > r_target)
            r_error = r_actual - r_target
            
            # 4. Correction: Steer AGAINST the error.
            # Modified to apply only when "oversteering" (yaw rate exceeds target) 
            # or when trying to stabilize (straight line).
            # We skip correction during "understeer" (corner entry) where r_actual < r_target,
            # to avoid artificially sharpening the turn-in.
            
            is_oversteer = abs(r_actual) > abs(r_target)
            opposing_signs = (np.sign(r_actual) != np.sign(r_target)) and (abs(r_target) > 0.01)
            
            if is_oversteer or opposing_signs:
                 steer_correction = -kp_stability * r_error
            else:
                 steer_correction = 0.0

            target_steer += steer_correction

        # Apply a slew rate limiter to steering for smoother inputs
        max_steer_rate = np.deg2rad(90.0) # deg/s
        max_change = max_steer_rate * dt_control
        steer_cmd = np.clip(target_steer, self.last_steer_cmd - max_change, self.last_steer_cmd + max_change)

        steer_limit_rad = np.deg2rad(45)
        final_steer_cmd = np.clip(steer_cmd, -steer_limit_rad, steer_limit_rad)
        self.last_steer_cmd = final_steer_cmd

        # Traction Control (TC) logic
        # Parameters taken from params['driver'] with sensible defaults:
        p_driver = self.params.get('driver', {})
        thr_sr_thresh = p_driver.get('tc_throttle_sr_threshold', 0.20)   # e.g. 0.20 = 20% slip
        brk_sr_thresh = p_driver.get('tc_brake_sr_threshold', 0.20)
        tc_gain       = p_driver.get('tc_gain', 0.8)                     # how strongly to reduce input (0..1)
        tc_min_throttle = p_driver.get('tc_min_throttle', 0.0)          # don't reduce throttle below this
        tc_min_brake    = p_driver.get('tc_min_brake', -0.5)            # don't reduce braking below this (negative)
        # Note: throttle_brake_cmd convention in your code: positive = throttle, negative = brake.

        # Compute slip ratios in the same way vehicle_dynamics uses them
        # (use small floor on vx to avoid divide-by-zero)
        R_eff = self.params['vehicle'].get('R_eff', 0.22)
        vx_for_slip_ratio = max(abs(vx), 0.1)  # floor to keep ratio sensible at very low speed
        wheel_w = np.ravel(state_dict['wheel_w_radps'])
        # slip_ratio = w*R / vx - 1  (array len 4: FL,FR,RL,RR)
        slip_ratios = (wheel_w * R_eff) / vx_for_slip_ratio - 1.0
        # slip ratio sign: positive => wheel linear speed > vehicle speed (spinning)
        # negative => wheel slower than vehicle (braking / impending lock)

        # Evaluate per-axle / per-wheel slips:
        sr_front = np.max(np.abs(slip_ratios[0:1]))
        sr_rear  = np.max(np.abs(slip_ratios[2:3]))

        # If we are commanding positive throttle and rear slip too high -> reduce throttle.
        if throttle_brake_cmd > 0.0:
            # focus TC on driven axle (rear wheels indices 2 and 3 in your model)
            driven_sr = np.max(slip_ratios[2:3])  # only positive spin matters for drive
            if driven_sr > thr_sr_thresh:
                # scale factor between 0 and 1: 0 => full cut, 1 => no change
                # smoothly reduce based on how far above threshold we are
                # Use exponential smoothing for smoother transitions
                excess = (driven_sr - thr_sr_thresh) / max(1.0 - thr_sr_thresh, 1e-6)
                reduction = np.clip(tc_gain * excess, 0.0, 1.0)
                # Apply exponential smoothing to reduction (alpha = 0.3 for smooth response)
                if not hasattr(self, 'last_tc_reduction'):
                    self.last_tc_reduction = 0.0
                alpha_tc = 0.3  # Smoothing factor
                reduction = alpha_tc * reduction + (1 - alpha_tc) * self.last_tc_reduction
                self.last_tc_reduction = reduction
                
                new_throttle = throttle_brake_cmd * (1.0 - reduction)
                # ensure we don't reduce below a minimum (so driver still has some control)
                throttle_brake_cmd = max(new_throttle, tc_min_throttle)
            else:
                # Reset reduction when below threshold
                if hasattr(self, 'last_tc_reduction'):
                    self.last_tc_reduction *= 0.9  # Decay

        # If commanding brake (negative), avoid wheel lock by reducing brake if wheels show large negative slip
        if throttle_brake_cmd < 0.0:
            # Negative slip ratio magnitude indicates wheels rotating slower: |sr| large => impending lock
            # Check all wheels for lock tendency (ABS logic)
            brake_lock_excess = np.max(-slip_ratios)  # -slip where slip negative -> positive value if wheel slower
            if brake_lock_excess > brk_sr_thresh:
                excess = (brake_lock_excess - brk_sr_thresh) / max(1.0 - brk_sr_thresh, 1e-6)
                reduction = np.clip(tc_gain * excess, 0.0, 1.0)
                # Apply exponential smoothing to reduction for smoother ABS response
                if not hasattr(self, 'last_abs_reduction'):
                    self.last_abs_reduction = 0.0
                alpha_abs = 0.4  # Slightly faster response for ABS
                reduction = alpha_abs * reduction + (1 - alpha_abs) * self.last_abs_reduction
                self.last_abs_reduction = reduction
                
                # Reduce magnitude of brake (make it less negative)
                new_brake = throttle_brake_cmd * (1.0 - reduction)
                # ensure braking not reduced to zero below min braking (you may want to tune)
                throttle_brake_cmd = min(new_brake, tc_min_brake)  # new_brake is negative; min(...) keeps it >= tc_min_brake
            else:
                # Reset reduction when below threshold
                if hasattr(self, 'last_abs_reduction'):
                    self.last_abs_reduction *= 0.9  # Decay
                
                
        # Anti-windup back-calculation for TC/ABS (applied != pid request)
        # pid_out_unclipped is what PID *wanted* (we don't currently return it),
        # but we can compute pid_out by re-evaluating the PID terms without committing integrator.
        # Simpler: call PID with a small helper that returns the unclipped PID output (or
        # cache it from update if you prefer). Here we'll compute a quick pid_request:
        # NOTE: this relies on PID internals; if you prefer, store pid_request from update().

        # Compute a "pid_request" by re-evaluating P+D + Ki*self.integral (current integral)
        # (This assumes PIDController stores Kp/Ki/Kd and integral/prev_error).
        P_req = self.throttle_pid.Kp * (target_velocity - vx)
        # derivative estimate (use prev_error stored)
        deriv_est = 0.0
        if dt_control > 1e-6:
            deriv_est = ( (target_velocity - vx) - self.throttle_pid.prev_error ) / dt_control
        D_req = self.throttle_pid.Kd * deriv_est
        I_req = self.throttle_pid.Ki * self.throttle_pid.integral
        pid_request = P_req + I_req + D_req

        applied = throttle_brake_cmd   # this is the (possibly reduced) command after TC/ABS

        # If the applied command is less in magnitude than requested (TC reduced it),
        # push the integral towards the applied value (back-calculation).
        # anti_windup_beta in (0..1) controls how quickly to unwind.
        # Increased from 0.001 to 0.1 for faster anti-windup response
        anti_windup_beta = 0.1  # tune this: 0.1..0.9
        diff = applied - pid_request
        if abs(diff) > 1e-4:
            # adjust integral so I_req_new + P + D ≈ applied
            # delta_I_needed = applied - (P_req + D_req) - I_req
            delta_I = diff
            # Convert delta_I to change in integral term (integral state = I / Ki)
            if abs(self.throttle_pid.Ki) > 1e-8:
                integral_adjust = (anti_windup_beta * delta_I) / self.throttle_pid.Ki
                # Clip the adjust so we don't jump integrator wildly
                max_adj = 0.5 * abs(self.throttle_pid.max_integral)
                integral_adjust = np.clip(integral_adjust, -max_adj, max_adj)
                self.throttle_pid.integral += integral_adjust
                # ensure integral stays within bounds
                self.throttle_pid.integral = np.clip(self.throttle_pid.integral,
                                                     self.throttle_pid.min_integral,
                                                    self.throttle_pid.max_integral)

        return {
            'throttle_brake_cmd': throttle_brake_cmd,
            'steer_cmd_rad': final_steer_cmd,
            'target_v_mps': target_velocity,
            'tc_info': {
                'sr_all': slip_ratios.tolist(),
                'sr_front_max': float(sr_front),
                'sr_rear_max': float(sr_rear)
            }
        }


    def _get_pure_pursuit_steer(self, state_dict):
        """
        Implements the Pure Pursuit steering control law.
        """
        k_lookahead = 0.05
        L_lookahead_base = 10.0

        vx = np.ravel(state_dict['vx_mps'])[0]
        x = np.ravel(state_dict['X_m'])[0]
        y = np.ravel(state_dict['Y_m'])[0]
        psi = np.ravel(state_dict['psi_rad'])[0]

        L_lookahead = k_lookahead * abs(vx) + L_lookahead_base

        _, current_idx = find_closest_point_on_path(x, y, self.path)
        
        path_s = self.path['s']
        
        target_s = path_s[current_idx] + L_lookahead
        
        if target_s > path_s[-1]:
            target_s -= path_s[-1]

        target_idx = np.argmin(np.abs(path_s - target_s))
        
        target_x, target_y = self.path['x'][target_idx], self.path['y'][target_idx]

        dx = target_x - x
        dy = target_y - y
        target_y_vehicle = dx * np.sin(-psi) + dy * np.cos(-psi)

        gamma = (2 * target_y_vehicle) / (L_lookahead**2)
        
        wheelbase = self.params['vehicle']['wheelbase']
        steer_angle = np.arctan(gamma * wheelbase)
        
        return steer_angle


class ExcitationController:
    """
    Open-loop controller for excitation mode: applies a time-based sequence of
    steering and throttle/brake commands to excite tyre dynamics for estimation.
    """

    def __init__(self, params):
        self.params = params
        exc = params.get('excitation', {})
        self.phases = list(exc.get('phases', []))
        self.repeat = int(exc.get('repeat_phases', 1))
        self.total_phase_time = sum(p['duration_s'] for p in self.phases)
        self.cycle_duration = self.total_phase_time * self.repeat
        self.initial_speed = exc.get('initial_speed_mps', 15.0)
        self._last_steer = 0.0
        # Slew rate (deg/s): limits how fast steer can change. Set to null or 0 to use exact phase values.
        max_steer_rate_deg_s = exc.get('max_steer_rate_deg_s')
        if max_steer_rate_deg_s is None or max_steer_rate_deg_s == 0:
            self._max_steer_rate_rad_s = None  # no limit: use exact phase steer_deg
        else:
            self._max_steer_rate_rad_s = np.deg2rad(float(max_steer_rate_deg_s))

    def get_controls(self, state_dict, dt_control, sim_time):
        """Return throttle_brake_cmd, steer_cmd_rad, target_v_mps from current phase."""
        if not self.phases:
            return {
                'throttle_brake_cmd': 0.0,
                'steer_cmd_rad': 0.0,
                'target_v_mps': self.initial_speed,
            }

        # Wrap time into [0, total_phase_time) so the phase list repeats every cycle
        t = sim_time % self.total_phase_time if self.total_phase_time > 0 else 0
        elapsed = 0.0
        steer_deg = 0.0
        throttle_brake = 0.0
        for phase in self.phases:
            d = phase['duration_s']
            if elapsed + d > t:
                steer_deg = phase['steer_deg']
                throttle_brake = phase['throttle_brake']
                break
            elapsed += d

        steer_rad = np.deg2rad(steer_deg)
        if self._max_steer_rate_rad_s is not None:
            max_change = self._max_steer_rate_rad_s * dt_control
            steer_cmd = np.clip(
                steer_rad,
                self._last_steer - max_change,
                self._last_steer + max_change
            )
        else:
            steer_cmd = steer_rad
        steer_limit = np.deg2rad(45)
        steer_cmd = np.clip(steer_cmd, -steer_limit, steer_limit)
        self._last_steer = steer_cmd

        vx = np.ravel(state_dict['vx_mps'])[0]
        return {
            'throttle_brake_cmd': throttle_brake,
            'steer_cmd_rad': steer_cmd,
            'target_v_mps': vx,  # not used for control in excitation, just for logging
        }
