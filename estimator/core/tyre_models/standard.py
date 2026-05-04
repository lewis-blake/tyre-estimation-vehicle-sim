import numpy as np
from .base import TyreModelBase

class StandardPacejka(TyreModelBase):
    """Standard Pacejka magic formula implementation."""

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
        gauss = np.exp(-((T_diff / sigma) ** 2))
        
        # Apply thermal floor if provided
        if T_factor_min is not None:
             T_factor_min = float(T_factor_min)
             return T_factor_min + (1.0 - T_factor_min) * gauss
             
        # Legacy behavior (no floor, clamped at 0.1)
        return np.maximum(0.1, gauss)

    def D_peak(self, D_ref, Fz, k_load, T, T_opt, sigma_left, sigma_right, T_factor_min, Fz0=1000.0):
        """
        Calculate peak force coefficient D (approx mu * Fz).
        Uses Power Law: D = D_ref * (Fz/Fz0)^k
        """
        # Legacy Thermal Logic: Weighted average
        # Uses T_factor_min passed from configuration (ground truth: 0.7)
        
        # Gaussian component
        T_diff = T - T_opt
        sigma = np.where(T < T_opt, sigma_left, sigma_right)
        gauss = np.exp(-((T_diff / sigma) ** 2))
        
        # Smooth floor scaling
        therm = T_factor_min + (1.0 - T_factor_min) * gauss
        
        # Power Law load sensitivity
        Fz_safe = np.maximum(Fz, 1e-6)
        D = D_ref * (Fz_safe / Fz0) ** k_load * therm
        
        return max(0.1, D)

    def Calpha_brush(self, Cref, Fz, k_load_stiff, T, T_opt, sigma_left, sigma_right, T_factor_min, Fz0=1000.0):
        """
        Cornering stiffness coefficient C_alpha (proportional to Fy/alpha/Fz).
        Uses Power Law: C = Cref * (Fz/Fz0)^k
        """
        # Legacy Thermal Logic: Weighted average
        # Gaussian component
        T_diff = T - T_opt
        sigma = np.where(T < T_opt, sigma_left, sigma_right)
        gauss = np.exp(-((T_diff / sigma) ** 2))
        
        # Smooth floor scaling
        therm = T_factor_min + (1.0 - T_factor_min) * gauss

        # Power Law load sensitivity
        Fz_safe = np.maximum(Fz, 1e-6)
        result = Cref * (Fz_safe / Fz0) ** k_load_stiff
        
        return result * therm

    def Ckappa_brush(self, Cref, Fz, k_load_stiff, T, T_opt, sigma_left, sigma_right, T_factor_min, Fz0=1000.0):
        """
        Longitudinal stiffness coefficient C_kappa.
        """
        return self.Calpha_brush(Cref, Fz, k_load_stiff, T, T_opt, sigma_left, sigma_right, T_factor_min, Fz0)

    def magic_formula_Fy(self, Fz, alpha, B, C, D, E=0.0):
        """
        Standard Pacejka Formula for Lateral Force.
        Fy = Fz * D * sin(C * atan(B*alpha - E*(B*alpha - atan(B*alpha))))
        
        NOTE: D here is Friction Coefficient (mu), so we multiply by Fz!
        """
        # Magic Formula
        # y = D sin(C atan(Bx - E(Bx - atan(Bx))))
        
        # If D is peak coefficient (mu), then Force = Fz * y
        # If D is peak Force, then Force = y
        
        # In our implementation D_peak returns mu.
        mu_peak = D
        
        # Input alpha is in radians?
        # Typically for Pacejka, alpha might be degrees or radians. The stiffness B is tuned accordingly.
        # Assuming radians for consistency with vehicle models.
        
        Bx = B * alpha
        
        # Protection against numerical errors if Bx is huge?
        # atan is safe.
        
        input_term = Bx - E * (Bx - np.arctan(Bx))
        result = mu_peak * np.sin(C * np.arctan(input_term))
        
        return Fz * result

    def magic_formula_Fx(self, Fz, kappa, B, C, D, E=0.0):
        """
        Standard Pacejka Formula for Longitudinal Force.
        Fx = Fz * D * sin(C * atan(B*kappa - E*(B*kappa - atan(B*kappa))))
        
        NOTE: D here is Friction Coefficient (mu), so we multiply by Fz!
        """
        mu_peak = D
        
        # kappa is dimensionless slip ratio
        
        Bx = B * kappa
        input_term = Bx - E * (Bx - np.arctan(Bx))
        result = mu_peak * np.sin(C * np.arctan(input_term))
        
        return Fz * result

    def apply_traction_ellipse(self, Fy0, Fx0, Fz, D_peak_lat, D_peak_lon, alpha=None, kappa=None, traction_ellipse_mode=None, **kwargs):
        """
        Limits combined forces to lie within the friction ellipse.
        traction_ellipse_mode: 'none' | 'current' | 'slip_dependent'.
        For slip_dependent, pass alpha, kappa, B_lat, C, E, B_lon, C_lon, E_lon via kwargs.
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
            if alpha is None or kappa is None or B_lat is None or C is None or E is None or B_lon is None or C_lon is None or E_lon is None:
                # Fallback to current behaviour if slip-dependent args missing
                mode = 'current'
            else:
                alpha_peak, Fy_peak = self.calculate_peak_slip_angle(Fz, B_lat, C, D_peak_lat, E)
                kappa_peak, Fx_peak = self.calculate_peak_slip_ratio(Fz, B_lon, C_lon, D_peak_lon, E_lon)
                Fy_max = np.where(np.abs(alpha) > alpha_peak, np.abs(Fy0), np.abs(Fy_peak))
                Fx_max = np.where(np.abs(kappa) > kappa_peak, np.abs(Fx0), np.abs(Fx_peak))
                Fy_max = np.maximum(np.asarray(Fy_max), 1e-10)
                Fx_max = np.maximum(np.asarray(Fx_max), 1e-10)
                term_y = Fy0 / Fy_max
                term_x = Fx0 / Fx_max
                mag_sq = term_y**2 + term_x**2
                scale = np.ones_like(mag_sq)
                over = mag_sq > 1.0
                scale[over] = 1.0 / np.sqrt(mag_sq[over])
                return Fy0 * scale, Fx0 * scale

        # current: fixed ellipse
        # Max limits
        Fy_max = Fz * D_peak_lat
        Fx_max = Fz * D_peak_lon
        
        # 1. Scalar case (optimization & simplicity)
        if np.isscalar(Fy0) and np.isscalar(Fx0) and np.isscalar(Fz):
            if Fy_max < 1.0 or Fx_max < 1.0:
                return 0.0, 0.0
            
            term_y = Fy0 / Fy_max
            term_x = Fx0 / Fx_max
            mag_sq = term_y**2 + term_x**2
            
            if mag_sq > 1.0:
                scale = 1.0 / np.sqrt(mag_sq)
                return Fy0 * scale, Fx0 * scale
            return Fy0, Fx0

        # 2. Vectorized case
        Fy0 = np.asarray(Fy0)
        Fx0 = np.asarray(Fx0)
        
        # Initialize output arrays
        Fy = np.array(Fy0)
        Fx = np.array(Fx0)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            term_y = Fy0 / Fy_max
            term_x = Fx0 / Fx_max
            mag_sq = term_y**2 + term_x**2

        # Create scale factor (default 1.0)
        scale = np.ones_like(mag_sq)
        
        # If mag_sq > 1.0, scale down
        mask_over = mag_sq > 1.0
        if np.any(mask_over):
            scale[mask_over] = 1.0 / np.sqrt(mag_sq[mask_over])
        
        # Apply scale
        Fy = Fy * scale
        Fx = Fx * scale
        
        # Enforce hard zero if limits are too small (Tyre lifted or similar)
        # Handle scalar or array max limits
        if np.ndim(Fy_max) == 0 and np.ndim(Fx_max) == 0:
             if Fy_max < 1.0 or Fx_max < 1.0:
                 return np.zeros_like(Fy0), np.zeros_like(Fx0)
        else:
             # Broadcast check
             mask_zero = (np.asarray(Fy_max) < 1.0) | (np.asarray(Fx_max) < 1.0)
             if np.any(mask_zero):
                 Fy = np.where(mask_zero, 0.0, Fy)
                 Fx = np.where(mask_zero, 0.0, Fx)
                 
        return Fy, Fx

    def invert_pacejka_fx(self, Fx_target, Fz, B, C, D, E, kappa_guess=0.0):
        """
        Numerical inversion of Fx formula to find kappa for a requested Force.
        Used for inverse tyre models (e.g. knowing force, finding slip).
        """
        from scipy.optimize import fsolve
        
        def error_func(k):
            return self.magic_formula_Fx(Fz, k, B, C, D, E) - Fx_target
            
        # Limit search?
        kappa_sol = fsolve(error_func, kappa_guess)
        return kappa_sol[0]

    def calculate_peak_slip_angle(self, Fz, B, C, D, E):
        """Find slip angle alpha where lateral force is maximum."""
        from scipy.optimize import minimize_scalar
        
        def func(a):
            return -abs(self.magic_formula_Fy(Fz, a, B, C, D, E))
            
        res = minimize_scalar(func, bounds=(0, 0.5), method='bounded')
        return res.x, -res.fun

    def calculate_peak_slip_ratio(self, Fz, B, C, D, E):
        """Find slip ratio kappa where longitudinal force is maximum."""
        from scipy.optimize import minimize_scalar
        
        def func(k):
            return -abs(self.magic_formula_Fx(Fz, k, B, C, D, E))
            
        res = minimize_scalar(func, bounds=(0, 0.5), method='bounded')
        return res.x, -res.fun
