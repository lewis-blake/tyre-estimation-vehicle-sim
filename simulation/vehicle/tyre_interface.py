
import numpy as np
import sys
import importlib.util
from pathlib import Path

# Resolve paths to the two separate 'core' packages (they clash on sys.path)
project_root = Path(__file__).resolve().parent.parent.parent
estimation_path = project_root / 'estimator'
master_est_path = project_root / 'shared'

# Add estimation path for standard model (still used elsewhere)
if str(estimation_path) not in sys.path:
    sys.path.insert(0, str(estimation_path))

# Import standard model via sys.path (works because it's added first)
try:
    import core.tyre_model as standard_model
except ImportError:
    print("Warning: Could not import core.tyre_model (Standard Model)")

# Import TTC model by explicit file path to avoid core namespace clash
try:
    _ttc_file = estimation_path / "core" / "tyre_models" / "ttc_pacejka.py"
    _spec = importlib.util.spec_from_file_location("ttc_pacejka", _ttc_file)
    ttc_model = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(ttc_model)
except Exception:
    print("Warning: Could not import ttc_pacejka (TTC Model)")

from scipy.optimize import minimize_scalar


def _apply_ellipse_limits(Fy0, Fx0, Fy_max, Fx_max):
    """Scale (Fy0, Fx0) onto ellipse (Fy/Fy_max)^2 + (Fx/Fx_max)^2 <= 1. Preserves signs."""
    Fy_max = np.maximum(np.asarray(Fy_max), 1e-10)
    Fx_max = np.maximum(np.asarray(Fx_max), 1e-10)
    term_y = Fy0 / Fy_max
    term_x = Fx0 / Fx_max
    mag_sq = term_y**2 + term_x**2
    scale = np.ones_like(mag_sq)
    over = mag_sq > 1.0
    scale[over] = 1.0 / np.sqrt(mag_sq[over])
    return Fy0 * scale, Fx0 * scale


def calculate_tyre_forces(tyre_model_type, alpha, kappa, Fz, T, params, default_pressure_kPa=120.0):
    """Unified logic"""
    if tyre_model_type == 'Standard':
        return _calculate_standard_force(alpha, kappa, Fz, T, params)
    elif tyre_model_type == 'TTC_Validation':
        return _calculate_ttc_force(alpha, kappa, Fz, T, params, default_pressure_kPa)
    else:
        raise ValueError(f"Unknown tyre model type: {tyre_model_type}")

def _calculate_standard_force(alpha, kappa, Fz, T, params):
    mode = params.get('traction_ellipse', 'current')

    # Extract params (defaults handled in config or here)
    D_ref = params.get('D_ref', 1.5)
    k_load = params.get('k_load', -0.2)
    Cref = params.get('Cref', 30.0)
    k_load_stiff = params.get('k_load_stiff', -0.6)
    C = params.get('C', 1.3)
    E = params.get('E', -1.0)
    
    D_ref_lon = params.get('D_ref_lon', D_ref)
    k_load_lon = params.get('k_load_lon', k_load)
    Cref_lon = params.get('Cref_lon', Cref)
    k_load_stiff_lon = params.get('k_load_stiff_lon', k_load_stiff)
    C_lon = params.get('C_lon', 1.65)
    E_lon = params.get('E_lon', -1.0)
    
    T_opt = params.get('T_opt', 60.0)
    sigma_left = params.get('sigma_left', 15.0)
    sigma_right = params.get('sigma_right', 15.0)
    T_opt_BCD = params.get('T_opt_BCD', 60.0)
    sigma_left_BCD = params.get('sigma_left_BCD', 80.0)
    sigma_right_BCD = params.get('sigma_right_BCD', 100.0)
    T_factor_min_BCD = params.get('T_factor_min_BCD', 0.9)
    T_factor_min = params.get('T_factor_min', 0.7)
    Fz0 = params.get('Fz0', 1000.0)
    
    # Calculate Lat
    D_lat = standard_model.D_peak(D_ref, Fz, k_load, T, T_opt, sigma_left, sigma_right, T_factor_min, Fz0)
    
    Calpha = standard_model.Calpha_brush(Cref, Fz, k_load_stiff, T, T_opt_BCD, sigma_left_BCD, sigma_right_BCD, T_factor_min_BCD, Fz0)
    B_lat = Calpha / (C * np.maximum(D_lat, 1e-6))
    Fy0 = standard_model.magic_formula_Fy(Fz, alpha, B_lat, C, D_lat, E)
    
    # Calculate Lon
    D_lon = standard_model.D_peak(D_ref_lon, Fz, k_load_lon, T, T_opt, sigma_left, sigma_right, T_factor_min, Fz0)
    Ckappa = standard_model.Ckappa_brush(Cref_lon, Fz, k_load_stiff_lon, T, T_opt_BCD, sigma_left_BCD, sigma_right_BCD, T_factor_min_BCD, Fz0)
    B_lon = Ckappa / (C_lon * np.maximum(D_lon, 1e-6))
    Fx0 = standard_model.magic_formula_Fx(Fz, kappa, B_lon, C_lon, D_lon, E_lon)
    
    if mode == 'none':
        return Fx0, Fy0

    if mode == 'current':
        Fy, Fx = standard_model.apply_traction_ellipse(Fy0, Fx0, Fz, D_lat, D_lon)
        return Fx, Fy

    # slip_dependent: ellipse limits from peak / past-peak
    alpha_peak, Fy_peak = standard_model.calculate_peak_slip_angle(Fz, B_lat, C, D_lat, E)
    kappa_peak, Fx_peak = standard_model.calculate_peak_slip_ratio(Fz, B_lon, C_lon, D_lon, E_lon)
    Fy_max = np.where(np.abs(alpha) > alpha_peak, np.abs(Fy0), np.abs(Fy_peak))
    Fx_max = np.where(np.abs(kappa) > kappa_peak, np.abs(Fx0), np.abs(Fx_peak))
    Fy_max = np.maximum(Fy_max, 1e-10)
    Fx_max = np.maximum(Fx_max, 1e-10)
    Fy, Fx = _apply_ellipse_limits(Fy0, Fx0, Fy_max, Fx_max)
    return Fx, Fy

def _calculate_ttc_force(alpha, kappa, Fz, T, params, default_pressure_kPa):
    mode = params.get('traction_ellipse', 'current')
    SA_deg = np.rad2deg(alpha)
    P = default_pressure_kPa

    if mode == 'none':
        FX_pure = ttc_model.longitudinal_force_model(kappa, Fz, T, P, [], fixed_params=params)
        FY_pure = ttc_model.lateral_force_model(SA_deg, Fz, T, P, [], fixed_params=params)
        return FX_pure, FY_pure

    if mode == 'current':
        Fx, Fy = ttc_model.combined_force_model(SA_deg, kappa, Fz, T, P, [], fixed_params=params)
        return Fx, Fy

    # slip_dependent: pure forces then ellipse with slip-dependent limits
    FY_pure = ttc_model.lateral_force_model(SA_deg, Fz, T, P, [], fixed_params=params)
    FX_pure = ttc_model.longitudinal_force_model(kappa, Fz, T, P, [], fixed_params=params)
    # Peak slip (deg for lateral, dimensionless for long) and force at peak; use scalars for optimizer
    Fz_s = float(np.asarray(Fz).ravel()[0])
    T_s = float(np.asarray(T).ravel()[0])
    def lat_func(sa):
        return -np.abs(ttc_model.lateral_force_model(sa, Fz_s, T_s, P, [], fixed_params=params))
    res_lat = minimize_scalar(lat_func, bounds=(0, 20.0), method='bounded')
    alpha_peak_deg = res_lat.x
    Fy_peak_mag = np.abs(-res_lat.fun)
    def lon_func(sr):
        return -np.abs(ttc_model.longitudinal_force_model(sr, Fz_s, T_s, P, [], fixed_params=params))
    res_lon = minimize_scalar(lon_func, bounds=(0, 0.5), method='bounded')
    kappa_peak = res_lon.x
    Fx_peak_mag = np.abs(-res_lon.fun)
    # Past-peak: shrink ellipse to current force magnitude
    SA_deg = np.asarray(SA_deg)
    kappa = np.asarray(kappa)
    Fy_max = np.where(np.abs(SA_deg) > alpha_peak_deg, np.abs(FY_pure), Fy_peak_mag)
    Fx_max = np.where(np.abs(kappa) > kappa_peak, np.abs(FX_pure), Fx_peak_mag)
    Fy_max = np.maximum(Fy_max, 1e-10)
    Fx_max = np.maximum(Fx_max, 1e-10)
    Fy, Fx = _apply_ellipse_limits(FY_pure, FX_pure, Fy_max, Fx_max)
    return Fx, Fy

def calculate_peak_forces(tyre_model_type, Fz, tyre_params, T=None):
    """
    Calculate peak longitudinal and lateral forces for a given load.
    
    Returns:
        Fx_max, Fy_max
    """
    Fz = np.atleast_1d(Fz)[0] # Extract scalar
    
    if T is None:
         T = tyre_params.get('T_opt', 60.0)

    if tyre_model_type == 'Standard':
        return _calculate_standard_peak_forces(Fz, tyre_params, T)
    elif tyre_model_type == 'TTC_Validation':
        return _calculate_ttc_peak_forces(Fz, tyre_params, T)
    else:
         raise ValueError(f"Unknown tyre model type: {tyre_model_type}")

def _calculate_standard_peak_forces(Fz, tyre_params, T):
    # Reimplementing logic from gg_diagram_generator roughly
    D_ref = tyre_params.get('D_ref', 1.5)
    k_load = tyre_params.get('k_load', -0.2)
    D_ref_lon = tyre_params.get('D_ref_lon', D_ref)
    k_load_lon = tyre_params.get('k_load_lon', k_load)
    
    T_opt = tyre_params.get('T_opt', 60.0)
    sigma_left = tyre_params.get('sigma_left', 15.0)
    sigma_right = tyre_params.get('sigma_right', 15.0)
    Fz0 = tyre_params.get('Fz0', 1000.0)

    # Longitudinal parameters
    Cref_lon = tyre_params.get('Cref_lon', tyre_params.get('Cref', 30.0))
    k_load_stiff_lon = tyre_params.get('k_load_stiff_lon', tyre_params.get('k_load_stiff', 0.0))
    C_lon = tyre_params.get('C_lon', tyre_params.get('C', 1.65))
    E_lon = tyre_params.get('E_lon', tyre_params.get('E', -1.0))
    
    T_opt_BCD = tyre_params.get('T_opt_BCD', 90.0)
    sigma_left_BCD = tyre_params.get('sigma_left_BCD', 80.0)
    sigma_right_BCD = tyre_params.get('sigma_right_BCD', 100.0)
    T_factor_min_BCD = tyre_params.get('T_factor_min_BCD', 0.9)

    # Lat
    D_lat = standard_model.D_peak(D_ref, Fz, k_load, T, T_opt, sigma_left, sigma_right, Fz0)
    Cref = tyre_params.get('Cref', 30.0)
    k_load_stiff = tyre_params.get('k_load_stiff', 0.0)
    C = tyre_params.get('C', 1.65)
    E = tyre_params.get('E', -1.0)
    
    Calpha = standard_model.Calpha_brush(Cref, Fz, k_load_stiff, T, T_opt_BCD, sigma_left_BCD, sigma_right_BCD, T_factor_min_BCD, Fz0)
    B_lat = Calpha / (C * np.maximum(D_lat, 1e-6))
    
    alpha_peak, Fy_max = standard_model.calculate_peak_slip_angle(Fz, B_lat, C, D_lat, E)
    
    # Lon
    D_lon = standard_model.D_peak(D_ref_lon, Fz, k_load_lon, T, T_opt, sigma_left, sigma_right, Fz0)
    Ckappa = standard_model.Ckappa_brush(Cref_lon, Fz, k_load_stiff_lon, T, T_opt_BCD, sigma_left_BCD, sigma_right_BCD, T_factor_min_BCD, Fz0)
    B_lon = Ckappa / (C_lon * np.maximum(D_lon, 1e-6))
    
    kappa_peak, Fx_max = standard_model.calculate_peak_slip_ratio(Fz, B_lon, C_lon, D_lon, E_lon)
    
    return Fx_max, Fy_max

def _calculate_ttc_peak_forces(Fz, params, T):
    P = params.get('default_pressure_kPa', 120.0)
    
    # Grid search for Lat Peak
    # TTC model lateral_force_model(SA, FZ, T, P, params, fixed_params)
    
    def lat_func(sa):
        return -abs(ttc_model.lateral_force_model(sa, Fz, T, P, [], fixed_params=params))
        
    res_lat = minimize_scalar(lat_func, bounds=(0, 20.0), method='bounded') # 0 to 20 deg
    Fy_max = -res_lat.fun
    
    # Grid search for Lon Peak
    def lon_func(sr):
         return -abs(ttc_model.longitudinal_force_model(sr, Fz, T, P, [], fixed_params=params))
         
    res_lon = minimize_scalar(lon_func, bounds=(0, 0.5), method='bounded')
    Fx_max = -res_lon.fun
    
    return Fx_max, Fy_max

