
from typing import Optional
import numpy as np
from .base import TyreModelBase
from . import ttc_pacejka as ttc
import sys
from pathlib import Path

# Add shared directory to path to find tyre_configs
_estimator_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_estimator_dir.parent / "shared"))
from tyre_configs import ttc_config_paper as ttc_config  # Use paper config as the reference

class TtcPacejka(TyreModelBase):
    """
    Wrapper for TTC Pacejka model to conform to TyreModelBase interface.
    
    Adapts the decomposed method calls (D_peak, etc.) to the TTC model implementations.
    """
    
    def __init__(self):
        # Load fixed parameters from ttc_config
        self.config = ttc_config.get_config()
        self.fixed_lat = {k: v for k, v in self.config.items() if k in ['C', 'Fz0', 'bias', 'T_opt', 'T_disp', 'T_stiff0', 'p_load0', 'p_stiff0']}
        self.fixed_lon = {k: v for k, v in self.config.items() if k in ['C_x', 'Fz0', 'bias_x', 'T_opt_x', 'T_disp_x', 'T_stiff0_x', 'p_load0_x', 'p_stiff0_x']}

    def magic_formula_Fy(self, Fz, alpha, B, C, D, E=0.0):
        """
        Calculate lateral force Fy using standard Magic Formula 5.2 equation.
        Fy = D * sin(C * arctan(Bx - E * (Bx - arctan(Bx))))

        Note: alpha is expected in RADIANS (consistent with the rest of the
        estimator pipeline which computes slip angles via atan2).
        """
        # Prevent zero B
        B = B + 1e-10

        Bx = B * alpha
        Fy = D * np.sin(C * np.arctan(Bx - E * (Bx - np.arctan(Bx))))

        return Fy

    def magic_formula_Fx(self, Fz, kappa, B, C, D, E=0.0):
        """
        Calculate longitudinal force Fx.
        """
        # Ensure Fz is positive
        Fz = np.abs(Fz)
        
        # Prevent zero B
        B = B + 1e-10
        
        
        Bx = B * kappa
        Fx = D * np.sin(C * np.arctan(Bx - E * (Bx - np.arctan(Bx))))
        
        return Fx

    def D_peak(self, D_ref, Fz, k_load, T, T_opt, sigma_left, sigma_right, T_factor_min=None, Fz0=1000.0):
        """
        Calculate peak force D using TTC model.

        D = (D_ref + temp_effect) * Fz + 1e-3 * k_load * Fz^2

        In the TTC model, sigma_left maps to T_disp (temperature dispersion).
        If sigma_left is provided and > 0, it is used as T_disp; otherwise
        the value from ttc_config is used as fallback.

        T_opt defaults to config value if not provided (e.g. when not in
        the UKF state vector).
        """
        Fz_abs = np.abs(Fz)

        # Fall back to config T_opt if not provided
        if T_opt is None:
            T_opt = self.config.get('T_opt', 60.0)

        # Use sigma_left as T_disp if provided, else fall back to config
        T_disp = sigma_left if sigma_left is not None and sigma_left > 0 else self.config.get('T_disp', 1e10)
        temp_effect = ttc.temperature_effect_cosh(T, T_opt, T_disp)

        D = (D_ref + temp_effect) * Fz_abs + 1e-3 * k_load * Fz_abs**2

        return D

    def Calpha_brush(self, Cref, Fz, k_load_stiff, T, T_opt, sigma_left, sigma_right, T_factor_min, Fz0=1000.0):
        """
        Calculate cornering stiffness Calpha.

        C_alpha = C_ref * sin(2 * arctan(Fz / k_load_stiff)) * lambda_T

        When T_factor_min is provided and positive, it is used as T_stiff0
        (the constant term in the stiffness temperature polynomial).
        This allows the UKF to estimate T_stiff0 by placing it in the
        T_factor_min slot of the state vector.
        """
        Fz_abs = np.abs(Fz)

        # Use T_factor_min as T_stiff0 if provided and positive, else config fallback
        T_stiff0 = T_factor_min if T_factor_min is not None and T_factor_min > 0 else self.config.get('T_stiff0', 1.0)
        T_stiff1 = self.config.get('T_stiff1', 0.0)
        T_stiff2 = self.config.get('T_stiff2', 0.0)

        lambda_T_stiff = T_stiff0 + T_stiff1 * T + T_stiff2 * T**2

        C_alpha = Cref * np.sin(2 * np.arctan(Fz_abs / k_load_stiff)) * lambda_T_stiff

        return C_alpha

    def Ckappa_brush(self, Cref, Fz, k_load_stiff, T, T_opt, sigma_left, sigma_right, T_factor_min, Fz0=1000.0):
        """
        Calculate longitudinal stiffness Ckappa.
        """
        return self.Calpha_brush(Cref, Fz, k_load_stiff, T, T_opt, sigma_left, sigma_right, T_factor_min, Fz0)

    def thermal_mult(self, T, T_opt, sigma_left, sigma_right, T_factor_min=None):
        """
        Calculate thermal multiplier using TTC cosh model.
        Returns the ADDITIVE term (or multiplier if mapped).
        
        TTC uses additive temp_effect for D. 
        But this function is often called to get a multiplier (0..1).
        
        We will return the value of `temperature_effect_cosh`.
        Note: This is an additive term <= 0.
        """
        T_disp = self.config.get('T_disp', 1e10)
        return ttc.temperature_effect_cosh(T, T_opt, T_disp)

    def apply_traction_ellipse(self, Fy0, Fx0, Fz, D_peak_lat, D_peak_lon, alpha=None, kappa=None, traction_ellipse_mode=None, **kwargs):
        """
        traction_ellipse_mode: 'none' | 'current' | 'slip_dependent'.
        current: sigma-based coupling (matches ttc_pacejka.combined_force_model).
        slip_dependent: ellipse (Fy/Fy_max)^2 + (Fx/Fx_max)^2 <= 1 with slip-dependent limits.
        """
        mode = traction_ellipse_mode or 'current'
        if mode == 'none':
            return Fy0, Fx0

        if mode == 'slip_dependent':
            alpha = kwargs.get('alpha')
            kappa = kwargs.get('kappa')
            B_lat = kwargs.get('B_lat')
            C = kwargs.get('C')
            E = kwargs.get('E')
            B_lon = kwargs.get('B_lon')
            C_lon = kwargs.get('C_lon')
            E_lon = kwargs.get('E_lon')
            if alpha is not None and kappa is not None and B_lat is not None and C is not None and E is not None and B_lon is not None and C_lon is not None and E_lon is not None:
                alpha_peak, Fy_peak = self.calculate_peak_slip_angle(Fz, B_lat, C, D_peak_lat, E)
                kappa_peak, Fx_peak = self.calculate_peak_slip_ratio(Fz, B_lon, C_lon, D_peak_lon, E_lon)
                alpha = np.asarray(alpha)
                kappa = np.asarray(kappa)
                Fy_max = np.where(np.abs(alpha) > alpha_peak, np.abs(Fy0), np.abs(Fy_peak))
                Fx_max = np.where(np.abs(kappa) > kappa_peak, np.abs(Fx0), np.abs(Fx_peak))
                Fy_max = np.maximum(Fy_max, 1e-10)
                Fx_max = np.maximum(Fx_max, 1e-10)
                term_y = Fy0 / Fy_max
                term_x = Fx0 / Fx_max
                mag_sq = term_y**2 + term_x**2
                scale = np.ones_like(mag_sq)
                over = mag_sq > 1.0
                scale[over] = 1.0 / np.sqrt(mag_sq[over])
                return Fy0 * scale, Fx0 * scale
            # else fall through to current

        # current: sigma coupling
        if alpha is None or kappa is None:
            return Fy0, Fx0

        alpha = float(np.asarray(alpha).ravel()[0])
        kappa = float(np.asarray(kappa).ravel()[0])

        sigma = np.sqrt(kappa**2 + (alpha * np.cos(alpha))**2 + 1e-10)
        cos_factor = np.abs(kappa) / (sigma + 1e-10)
        sin_factor = np.abs(alpha * np.cos(alpha)) / (sigma + 1e-10)

        return Fy0 * sin_factor, Fx0 * cos_factor

    def invert_pacejka_fx(self, Fx_target, Fz, B, C, D, E, kappa_guess=0.0):
        """Simple inversion or optimization to find kappa."""
        from scipy.optimize import minimize_scalar
        
        def obj(k):
            return (self.magic_formula_Fx(Fz, k, B, C, D, E) - Fx_target)**2
            
        res = minimize_scalar(obj, bounds=(-1.5, 1.5), method='bounded')
        return res.x

    def calculate_peak_slip_angle(self, Fz, B, C, D, E):
        """Find slip angle (rad) for peak lateral force. Returns (alpha_peak, Fy_peak_magnitude)."""
        from scipy.optimize import minimize_scalar
        def obj(a):
            return -np.abs(self.magic_formula_Fy(Fz, a, B, C, D, E))
        res = minimize_scalar(obj, bounds=(0, 0.35), method='bounded')  # ~20 deg in rad
        Fy_peak = np.abs(self.magic_formula_Fy(Fz, res.x, B, C, D, E))
        return res.x, Fy_peak

    def calculate_peak_slip_ratio(self, Fz, B, C, D, E):
        """Find slip ratio for peak longitudinal force. Returns (kappa_peak, Fx_peak_magnitude)."""
        from scipy.optimize import minimize_scalar
        def obj(k):
            return -np.abs(self.magic_formula_Fx(Fz, k, B, C, D, E))
        res = minimize_scalar(obj, bounds=(0, 1.0), method='bounded')
        Fx_peak = np.abs(self.magic_formula_Fx(Fz, res.x, B, C, D, E))
        return res.x, Fx_peak

    def get_default_params(self) -> dict:
        """Return parameters loaded from ttc_config, mapped to standard naming."""
        c = self.config
        return {
            # Lateral Mappings
            'D_ref': c.get('D_ref', 1.5),
            'k_load': c.get('k_load', 0.0),
            'Cref': c.get('C_ref', 50000.0),
            'k_load_stiff': c.get('k_load_stiff', 1000.0), # Assuming this maps
            'C': c.get('C', 1.3),
            'E': c.get('E', -1.0),
            'T_opt': c.get('T_opt', 60.0),
            'sigma_left': c.get('T_disp', 200.0),
            'sigma_right': c.get('T_disp', 200.0),
            
            # Longitudinal Mappings (TTC uses _x suffix)
            'D_ref_lon': c.get('D_ref_x', 1.5),
            'k_load_lon': c.get('k_load_x', 0.0),
            'Cref_lon': c.get('C_ref_x', 50000.0),
            'k_load_stiff_lon': c.get('k_load_stiff_x', 1000.0),
            'C_lon': c.get('C_x', 1.65),
            'E_lon': c.get('E_x', -1.0),
            
            # Additional fallback mappings
            'T_factor_min': 0.7,
            'Fz0': c.get('Fz0', 1000.0)
        }
