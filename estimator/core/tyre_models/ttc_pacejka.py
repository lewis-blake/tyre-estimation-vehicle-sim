"""
Tire Model Functions - Magic Formula 5.2
=========================================

This file contains tire model functions for lateral, longitudinal, and combined slip.
- FY = f(SA, FZ, T, params) for lateral force
- FX = f(SR, FZ, T, params) for longitudinal force

Configuration is imported from config.py - modify that file to change parameters.
"""

import numpy as np

# Import configuration - DISABLED for integration
# We inject parameters via tyre_interface.py, so we don't need config.py here.
CONFIG_AVAILABLE = False
DEBUG_VERBOSE = False
# try:
# from config import (
# get_fixed_params, get_estimated_param_info, DEBUG_VERBOSE,
# MODEL_MODE
# )
# CONFIG_AVAILABLE = True
# except ImportError:
# CONFIG_AVAILABLE = False
# DEBUG_VERBOSE = True


# FALLBACK FIXED PARAMETERS (used if config.py not available)
# This is kept for backwards compatibility - prefer using config.py

FIXED_PARAMS = {
    # Magic Formula shape parameters
    'C': None,
    'E': None,

    # Reference values
    'D_ref': None,
    'C_ref': None,
    'Fz0': 1000.0,

    # Bias parameter (allows origin shift)
    'bias': 0.0,

    # Load sensitivity exponents
    'k_load': None,
    'k_load_stiff': None,

    # Temperature parameters for peak force (D) - NEW Hyperbolic Cosine Model
    'T_opt': None,      # Optimal temperature
    'T_disp': None,     # Temperature dispersion parameter

    # OLD TEMPERATURE MODEL - Asymmetric Gaussian (preserved for reference)
    # 'T_opt_D': None,
    # 'sigma_left_D': None,
    # 'sigma_right_D': None,
    # 'temp_baseline_D': 0.8,

    # Temperature parameters for stiffness (C_alpha) - polynomial form
    'T_stiff0': None,
    'T_stiff1': None,
    'T_stiff2': None,
}


# THERMAL MULTIPLIER FUNCTIONS

# OLD TEMPERATURE MODEL - Asymmetric Gaussian (preserved for reference)
# This model is preserved for future reference. It has been replaced with a
# hyperbolic cosine-based model that integrates temperature effects additively
# into D_ref rather than multiplicatively.
#
# def thermal_multiplier(T, T_opt, sigma_left, sigma_right, temp_baseline):
# """
# Asymmetric Gaussian thermal multiplier with baseline.
#
# lambda_T = temp_baseline + (1 - temp_baseline) * exp(-(T - T_opt)^2 / (2*sigma^2))
#
# where sigma = sigma_left if T < T_opt, else sigma_right
#
# Parameters:
# T : array-like
# Temperature (deg C)
# T_opt : float
# Optimal temperature
# sigma_left : float
# Gaussian width below T_opt
# sigma_right : float
# Gaussian width above T_opt
# temp_baseline : float
# Minimum multiplier value (0=full Gaussian, 1=no temp effect)
#
# Returns:
# lambda_T : np.ndarray
# Thermal multiplier (ranges from temp_baseline to 1)
# """
# sigma = np.where(T < T_opt, sigma_left, sigma_right)
# gaussian = np.exp(-((T - T_opt)**2) / (2 * sigma**2))
# lambda_T = temp_baseline + (1 - temp_baseline) * gaussian
# return lambda_T

# Note: The longitudinal model still uses thermal_multiplier (see below)
def thermal_multiplier(T, T_opt, sigma_left, sigma_right, temp_baseline):
    """
    Asymmetric Gaussian thermal multiplier with baseline.

    NOTE: This function is still used by the longitudinal model.
    The lateral model now uses temperature_effect_cosh instead.

    lambda_T = temp_baseline + (1 - temp_baseline) * exp(-(T - T_opt)^2 / (2*sigma^2))

    where sigma = sigma_left if T < T_opt, else sigma_right

    Parameters:
    -----------
    T : array-like
        Temperature (deg C)
    T_opt : float
        Optimal temperature
    sigma_left : float
        Gaussian width below T_opt
    sigma_right : float
        Gaussian width above T_opt
    temp_baseline : float
        Minimum multiplier value (0=full Gaussian, 1=no temp effect)

    Returns:
    --------
    lambda_T : np.ndarray
        Thermal multiplier (ranges from temp_baseline to 1)
    """
    sigma = np.where(T < T_opt, sigma_left, sigma_right)
    gaussian = np.exp(-((T - T_opt)**2) / (2 * sigma**2))
    lambda_T = temp_baseline + (1 - temp_baseline) * gaussian
    return lambda_T


# NEW TEMPERATURE MODEL - Hyperbolic Cosine (for lateral model)

def temperature_effect_cosh(T, T_opt, T_disp):
    """
    Hyperbolic cosine-based temperature effect (additive to D_ref).

    This model creates a smooth inverted parabola-like curve centered at T_opt.
    The temperature effect is added to D_ref rather than multiplied.

    Formula: (1 - cosh((T - T_opt) / T_disp))

    Parameters:
    -----------
    T : array-like
        Temperature (deg C)
    T_opt : float
        Optimal temperature where effect is maximized (cosh minimum at 0)
    T_disp : float
        Temperature dispersion parameter - controls the width of the curve
        Smaller values = narrower peak, larger values = broader peak

    Returns:
    --------
    temp_effect : np.ndarray
        Temperature effect term to be added to D_ref
        Maximum value is 0 at T = T_opt (cosh(0) = 1)
        Becomes increasingly negative as |T - T_opt| increases

    Example:
    --------
    At T = T_opt: effect = 1 - cosh(0) = 1 - 1 = 0 (no change to D_ref)
    At T ≠ T_opt: effect < 0 (reduces D_ref)
    """
    temp_effect = 1.0 - np.cosh((T - T_opt) / T_disp)
    return temp_effect


# LATERAL FORCE MODEL (Magic Formula 5.2)

def lateral_force_model(SA, FZ, T, P, params, fixed_params=None):
    """
    Magic Formula 5.2 tire model for lateral force.

    FY = D * sin(C * arctan(Bx - E * (Bx - arctan(Bx)))) + bias

    Parameters:
    -----------
    SA : array-like
        Slip angle (deg)
    FZ : array-like
        Normal force (N) - positive values expected (abs taken internally)
    T : array-like
        Average tire temperature (deg C)
    P : array-like
        Tire pressure (kPa)
    params : array-like
        Parameters being optimized (order depends on which are fixed)
    fixed_params : dict, optional
        Dictionary of fixed parameter values (uses config if None)

    Returns:
    --------
    FY : np.ndarray
        Predicted lateral force (N)
    """
    # Get fixed parameters from config or use provided/fallback
    if fixed_params is None:
        if CONFIG_AVAILABLE:
            fixed_params = get_fixed_params('lateral')
        else:
            fixed_params = FIXED_PARAMS

    # Unpack parameters (mix of estimated and fixed)
    param_idx = 0

    def get_param(name):
        nonlocal param_idx
        if fixed_params.get(name) is not None:
            return fixed_params[name]
        else:
            val = params[param_idx]
            param_idx += 1
            return val

    # Extract all parameters
    C = get_param('C')
    E = get_param('E')
    D_ref = get_param('D_ref')
    C_ref = get_param('C_ref')
    Fz0 = get_param('Fz0')
    bias = get_param('bias')
    k_load = get_param('k_load')
    k_load_stiff = get_param('k_load_stiff')

    # Temperature parameters for D (peak force)
    # OLD TEMPERATURE MODEL - Asymmetric Gaussian (preserved for reference)
    # T_opt_D = get_param('T_opt_D')
    # sigma_left_D = get_param('sigma_left_D')
    # sigma_right_D = get_param('sigma_right_D')
    # temp_baseline_D = get_param('temp_baseline_D')

    # NEW TEMPERATURE MODEL - Hyperbolic Cosine
    T_opt = get_param('T_opt')
    T_disp = get_param('T_disp')

    # Temperature parameters for stiffness
    T_stiff0 = get_param('T_stiff0')
    T_stiff1 = get_param('T_stiff1')
    T_stiff2 = get_param('T_stiff2')

    # Pressure parameters for peak force (D)
    p_load0 = get_param('p_load0')
    p_load1 = get_param('p_load1')
    p_load2 = get_param('p_load2')

    # Pressure parameters for stiffness (C_alpha)
    p_stiff0 = get_param('p_stiff0')
    p_stiff1 = get_param('p_stiff1')
    p_stiff2 = get_param('p_stiff2')

    # Convert slip angle to radians
    alpha = np.deg2rad(SA)

    # Ensure FZ is positive (TTC data has negative FZ)
    Fz = np.abs(FZ)

    # PEAK FORCE COEFFICIENT (D) with load and thermal effects
    # OLD TEMPERATURE MODEL - Asymmetric Gaussian (preserved for reference)
    # lambda_T_D = thermal_multiplier(T, T_opt_D, sigma_left_D, sigma_right_D, temp_baseline_D)
    # D = (D_ref * Fz + 1e-3 * k_load * Fz**2) * lambda_T_D * lambda_P_D

    # NEW TEMPERATURE MODEL - Hyperbolic Cosine integrated into D_ref
    temp_effect = temperature_effect_cosh(T, T_opt, T_disp)
    lambda_P_D = p_load0 + p_load1 * P + p_load2 * P**2
    D = ((D_ref + temp_effect) * Fz + 1e-3 * k_load * Fz**2) * lambda_P_D

    # CORNERING STIFFNESS (C_alpha) with load and thermal effects
    lambda_T_stiff = T_stiff0 + T_stiff1 * T + T_stiff2 * T**2
    lambda_P_stiff = p_stiff0 + p_stiff1 * P + p_stiff2 * P**2
    #C_alpha = C_ref * (Fz / Fz0)**k_load_stiff * lambda_T_stiff
    C_alpha = C_ref * np.sin(2*np.arctan(Fz / k_load_stiff)) * lambda_T_stiff * lambda_P_stiff
    # STIFFNESS FACTOR (B)
    denom = C * np.maximum(D, 1e-10)
    B = C_alpha / np.maximum(denom, 1e-10)

    # MAGIC FORMULA (Pure Slip) + Bias
    Bx = B * alpha
    FY = D * np.sin(C * np.arctan(Bx - E * (Bx - np.arctan(Bx)))) + bias

    return FY


# LONGITUDINAL FORCE MODEL (Magic Formula)

def longitudinal_force_model(SR, FZ, T, P, params, fixed_params=None):
    """
    Magic Formula tire model for longitudinal force.

    FX = D * sin(C * arctan(Bx - E * (Bx - arctan(Bx)))) + bias

    Parameters:
    -----------
    SR : array-like
        Slip ratio (dimensionless, typically -1 to 1)
    FZ : array-like
        Normal force (N) - positive values expected (abs taken internally)
    T : array-like
        Average tire temperature (deg C)
    P : array-like
        Tire pressure (kPa)
    params : array-like
        Parameters being optimized (order depends on which are fixed)
    fixed_params : dict, optional
        Dictionary of fixed parameter values (uses config if None)

    Returns:
    --------
    FX : np.ndarray
        Predicted longitudinal force (N)
    """
    # Get fixed parameters from config or use provided/fallback
    if fixed_params is None:
        if CONFIG_AVAILABLE:
            fixed_params = get_fixed_params('longitudinal')
        else:
            # Fallback longitudinal params
            fixed_params = {
                'C_x': 1.65, 'E_x': None, 'D_ref_x': None, 'C_ref_x': None,
                'Fz0': 1000.0, 'bias_x': 0.0, 'k_load_x': None, 'k_load_stiff_x': None,
                'T_opt_x': None, 'T_disp_x': None,
                'T_stiff0_x': None, 'T_stiff1_x': None, 'T_stiff2_x': None,
            }

    # Unpack parameters
    param_idx = 0

    def get_param(name):
        nonlocal param_idx
        if fixed_params.get(name) is not None:
            return fixed_params[name]
        else:
            val = params[param_idx]
            param_idx += 1
            return val

    # Extract parameters
    C_x = get_param('C_x')
    E_x = get_param('E_x')
    D_ref_x = get_param('D_ref_x')
    C_ref_x = get_param('C_ref_x')
    Fz0 = get_param('Fz0')
    bias_x = get_param('bias_x')
    k_load_x = get_param('k_load_x')
    k_load_stiff_x = get_param('k_load_stiff_x')

    # Temperature parameters for D (peak force) - Hyperbolic Cosine Model
    T_opt_x = get_param('T_opt_x')
    T_disp_x = get_param('T_disp_x')

    # Temperature parameters for stiffness
    T_stiff0_x = get_param('T_stiff0_x')
    T_stiff1_x = get_param('T_stiff1_x')
    T_stiff2_x = get_param('T_stiff2_x')

    # Pressure parameters for longitudinal force
    p_load0_x = get_param('p_load0_x')
    p_load1_x = get_param('p_load1_x')
    p_load2_x = get_param('p_load2_x')
    p_stiff0_x = get_param('p_stiff0_x')
    p_stiff1_x = get_param('p_stiff1_x')
    p_stiff2_x = get_param('p_stiff2_x')

    # Ensure FZ is positive
    Fz = np.abs(FZ)

    # PEAK FORCE COEFFICIENT (D) with load and thermal effects
    # NEW TEMPERATURE MODEL - Hyperbolic Cosine integrated into D_ref
    temp_effect_x = temperature_effect_cosh(T, T_opt_x, T_disp_x)
    lambda_P_D_x = p_load0_x + p_load1_x * P + p_load2_x * P**2
    D = ((D_ref_x + temp_effect_x) * Fz + 1e-3 * k_load_x * Fz**2) * lambda_P_D_x

    # LONGITUDINAL STIFFNESS (C_kappa) with load and thermal effects
    lambda_T_stiff_x = T_stiff0_x + T_stiff1_x * T + T_stiff2_x * T**2
    lambda_P_stiff_x = p_stiff0_x + p_stiff1_x * P + p_stiff2_x * P**2
    C_kappa = C_ref_x * np.sin(2*np.arctan(Fz / k_load_stiff_x)) * lambda_T_stiff_x * lambda_P_stiff_x

    # STIFFNESS FACTOR (B)
    denom = C_x * np.maximum(D, 1e-10)
    B = C_kappa / np.maximum(denom, 1e-10)

    # MAGIC FORMULA + Bias
    Bx = B * SR
    FX = D * np.sin(C_x * np.arctan(Bx - E_x * (Bx - np.arctan(Bx)))) + bias_x

    return FX


# COMBINED SLIP MODEL

def combined_force_model(SA, SR, FZ, T, P, params, fixed_params=None):
    """
    Combined slip tire model for both lateral and longitudinal forces.

    Uses simplified coupling between lateral and longitudinal forces.

    Parameters:
    -----------
    SA : array-like
        Slip angle (deg)
    SR : array-like
        Slip ratio (dimensionless)
    FZ : array-like
        Normal force (N)
    T : array-like
        Temperature (deg C)
    P : array-like
        Tire pressure (kPa)
    params : array-like
        Combined parameters (lateral + longitudinal)
    fixed_params : dict, optional
        Combined fixed parameters

    Returns:
    --------
    FX, FY : tuple of np.ndarray
        Predicted longitudinal and lateral forces
    """
    if fixed_params is None:
        if CONFIG_AVAILABLE:
            fixed_params = get_fixed_params('combined')
        else:
            raise ValueError("Combined mode requires config.py")

    # Count lateral params to split
    lateral_fixed = get_fixed_params('lateral') if CONFIG_AVAILABLE else FIXED_PARAMS
    n_lateral = sum(1 for v in lateral_fixed.values() if v is None)

    # Split params
    params_lateral = params[:n_lateral]
    params_longitudinal = params[n_lateral:]

    # Get pure slip forces
    FY_pure = lateral_force_model(SA, FZ, T, P, params_lateral, fixed_params=fixed_params)
    FX_pure = longitudinal_force_model(SR, FZ, T, P, params_longitudinal, fixed_params=fixed_params)

    # Combined slip reduction (simplified Pacejka approach)
    alpha = np.deg2rad(SA)
    sigma_combined = np.sqrt(SR**2 + (alpha * np.cos(alpha))**2 + 1e-10)

    # Weighting factors
    cos_factor = np.abs(SR) / (sigma_combined + 1e-10)
    sin_factor = np.abs(alpha * np.cos(alpha)) / (sigma_combined + 1e-10)

    # Apply coupling
    FX = FX_pure * cos_factor
    FY = FY_pure * sin_factor

    return FX, FY


# MODEL FACTORY FUNCTION

def get_model_func(mode=None):
    """
    Factory function to return the appropriate model function.

    Parameters:
    -----------
    mode : str, optional
        'lateral', 'longitudinal', or 'combined'. Uses config.MODEL_MODE if None.

    Returns:
    --------
    model_func : callable
        The tire model function for the specified mode
    """
    if mode is None:
        if CONFIG_AVAILABLE:
            mode = MODEL_MODE
        else:
            mode = 'lateral'

    if DEBUG_VERBOSE:
        print(f"get_model_func returning model for mode: {mode}")

    if mode == 'lateral':
        return lateral_force_model
    elif mode == 'longitudinal':
        return longitudinal_force_model
    elif mode == 'combined':
        return combined_force_model
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'lateral', 'longitudinal', or 'combined'")


# PARAMETER INFO FUNCTIONS

def get_param_info(mode=None):
    """
    Get information about which parameters are being estimated vs fixed.

    Parameters:
    -----------
    mode : str, optional
        'lateral', 'longitudinal', or 'combined'. Uses config.MODEL_MODE if None.

    Returns:
    --------
    param_names : list of str
        Names of parameters being estimated (not fixed)
    fixed_summary : dict
        Dictionary of fixed parameter names and values
    """
    if CONFIG_AVAILABLE:
        fixed_params = get_fixed_params(mode)
    else:
        fixed_params = FIXED_PARAMS

    param_names = []
    fixed_summary = {}

    for name, value in fixed_params.items():
        if value is None:
            param_names.append(name)
        else:
            fixed_summary[name] = value

    return param_names, fixed_summary


def print_param_config(mode=None):
    """Print current parameter configuration."""
    param_names, fixed_summary = get_param_info(mode)

    print("=" * 70)
    print("PARAMETER CONFIGURATION")
    if mode:
        print(f"Mode: {mode}")
    print("=" * 70)
    print(f"\nParameters to ESTIMATE ({len(param_names)}):")
    for name in param_names:
        print(f"  {name}")

    print(f"\nParameters FIXED ({len(fixed_summary)}):")
    for name, value in fixed_summary.items():
        print(f"  {name:20s} = {value}")
    print("=" * 70)


# EXAMPLE/LEGACY MODEL (kept for backwards compatibility)

def example_pacejka_model(SA, FZ, T, params):
    """
    Example: Simplified Pacejka Magic Formula with load and thermal effects

    This is provided as a reference implementation.

    Parameters in 'params':
    -----------------------
    params[0] = B     : Stiffness factor
    params[1] = C     : Shape factor
    params[2] = D     : Peak factor (friction coefficient)
    params[3] = E     : Curvature factor
    params[4] = k_FZ  : Load sensitivity coefficient
    params[5] = k_T   : Temperature sensitivity coefficient
    """
    B, C, D, E, k_FZ, k_T = params

    # Convert slip angle to radians
    alpha = np.deg2rad(SA)

    # Load-dependent peak force
    Fz_nominal = 1000.0  # N
    D_load = D * np.abs(FZ) * (1 + k_FZ * (np.abs(FZ) - Fz_nominal) / Fz_nominal)

    # Temperature effect (optimal temperature = 60 deg C)
    T_opt = 60.0  # deg C
    temp_multiplier = 1.0 + k_T * (T - T_opt)

    # Magic Formula
    Bx = B * alpha
    FY = D_load * temp_multiplier * np.sin(
        C * np.arctan(Bx - E * (Bx - np.arctan(Bx)))
    )

    return FY
