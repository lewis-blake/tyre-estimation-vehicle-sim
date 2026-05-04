import numpy as np
from .base import TyreModelBase

class NewPacejka(TyreModelBase):
    """
    New Pacejka implementation with user-specified custom formulas.
    
    Formula differences from Standard:
    1. D_peak (Force Amplitude) = lambda_temp * (D_ref * Fz + k_load * Fz^2)
       Note: D is treated as PEAK FORCE, not friction coefficient.
    
    2. Stiffness BCD = C_ref * sin(2 * atan(Fz / k_stiff))
    
    3. Magic Formula: Fy = D * sin(C * atan(B * alpha - E*(...)))
       where B = BCD / (C * D)
    """

    def thermal_mult(self, T, T_opt, sigma_left, sigma_right, T_factor_min=None):
        """
        Calculates a thermal multiplier (bell curve) based on tyre temperature.
        Peak is at T_opt. Supports numpy arrays.
        """
        # Ensure inputs are arrays or scalars
        T = np.asarray(T)
        T_opt = float(T_opt)
        sigma_left = float(sigma_left)
        sigma_right = float(sigma_right)
        
        # Gaussian component
        T_diff = T - T_opt
        sigma = np.where(T < T_opt, sigma_left, sigma_right)
        gauss = np.exp(-0.5 * (T_diff / sigma) ** 2)
        
        # Apply thermal floor if provided
        if T_factor_min is not None:
             T_factor_min = float(T_factor_min)
             return T_factor_min + (1.0 - T_factor_min) * gauss
             
        return np.maximum(0.1, gauss)

    def D_peak(self, D_ref, Fz, k_load, T, T_opt, sigma_left, sigma_right, Fz0=1000.0):
        """
        Calculate Peak FORCE D using quadratic load sensitivity.
        Formula: D = lambda_temp * (D_ref * Fz + k_load * Fz^2)
        
        NOTE: Returns Force [N], not Friction Coefficient [-]
        """
        lambda_temp = self.thermal_mult(T, T_opt, sigma_left, sigma_right)
        
        # Quadratic load dependence for Force
        # D_ref should be ~ Friction Coeff (linear term)
        # k_load should be ~ Load sensitivity (quadratic term)
        
        D_force = lambda_temp * (D_ref * Fz + k_load * (Fz**2))
        
        return max(1.0, D_force)

    def Calpha_brush(self, Cref, Fz, k_load_stiff, T, T_opt, sigma_left, sigma_right, T_factor_min, Fz0=1000.0):
        """
        Calculate Cornering Stiffness (BCD) using sinusoidal formula.
        Formula: BCD = C_ref * sin(2 * atan(Fz / k_stiff))
        
        Note: Cref here acts as amplitude/scaling factor.
        k_load_stiff acts as the stiffness factor inside atan.
        """
        # Avoid division by zero
        k_stiff_safe = k_load_stiff if abs(k_load_stiff) > 1e-4 else 1e-4
        
        BCD = Cref * np.sin(2 * np.arctan(Fz / k_stiff_safe))
        
        return max(1.0, BCD)

    def Ckappa_brush(self, Cref, Fz, k_load_stiff, T, T_opt, sigma_left, sigma_right, T_factor_min, Fz0=1000.0):
        """Longitudinal stiffness - using same formula as lateral for now."""
        return self.Calpha_brush(Cref, Fz, k_load_stiff, T, T_opt, sigma_left, sigma_right, T_factor_min, Fz0)

    def magic_formula_Fy(self, Fz, alpha, B, C, D, E=0.0):
        """
        New Magic Formula.
        Fy = D * sin(C * atan(B*alpha - E*(...)))
        
        Crucial difference: D is already Force, do NOT multiply by Fz again.
        """
        # D is Peak Force from D_peak()
        
        Bx = B * alpha
        input_term = Bx - E * (Bx - np.arctan(Bx))
        result = D * np.sin(C * np.arctan(input_term))
        
        return result

    def magic_formula_Fx(self, Fz, kappa, B, C, D, E=0.0):
        """
        New Magic Formula for Longitudinal.
        Fx = D * sin(C * atan(B*kappa - E*(...)))
        """
        Bx = B * kappa
        input_term = Bx - E * (Bx - np.arctan(Bx))
        result = D * np.sin(C * np.arctan(input_term))
        
        return result

    def apply_traction_ellipse(self, Fy0, Fx0, Fz, D_peak_lat, D_peak_lon, alpha=None, kappa=None, traction_ellipse_mode=None, **kwargs):
        """Standard traction ellipse logic. traction_ellipse_mode: 'none' | 'current'."""
        mode = traction_ellipse_mode or 'current'
        if mode == 'none':
            return Fy0, Fx0
        # D_peak_lat/lon are Forces [N] in this model
        Fy_max = D_peak_lat
        Fx_max = D_peak_lon
        
        if Fy_max < 1.0 or Fx_max < 1.0:
            return 0.0, 0.0
            
        term_y = (Fy0 / Fy_max)
        term_x = (Fx0 / Fx_max)
        
        magnitude_sq = term_y**2 + term_x**2
        
        if magnitude_sq > 1.0:
            scale = 1.0 / np.sqrt(magnitude_sq)
            Fy = Fy0 * scale
            Fx = Fx0 * scale
        else:
            Fy = Fy0
            Fx = Fx0
            
        return Fy, Fx

    def invert_pacejka_fx(self, Fx_target, Fz, B, C, D, E, kappa_guess=0.0):
        from scipy.optimize import fsolve
        def error_func(k):
            return self.magic_formula_Fx(Fz, k, B, C, D, E) - Fx_target
        kappa_sol = fsolve(error_func, kappa_guess)
        return kappa_sol[0]

    def calculate_peak_slip_angle(self, Fz, B, C, D, E):
        from scipy.optimize import minimize_scalar
        def func(a):
            return -abs(self.magic_formula_Fy(Fz, a, B, C, D, E))
        res = minimize_scalar(func, bounds=(0, 0.5), method='bounded')
        return res.x, -res.fun

    def calculate_peak_slip_ratio(self, Fz, B, C, D, E):
        from scipy.optimize import minimize_scalar
        def func(k):
            return -abs(self.magic_formula_Fx(Fz, k, B, C, D, E))
        res = minimize_scalar(func, bounds=(0, 0.5), method='bounded')
        return res.x, -res.fun
