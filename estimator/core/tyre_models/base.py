from abc import ABC, abstractmethod
from typing import Tuple

class TyreModelBase(ABC):
    """Abstract base class for tyre models."""

    @abstractmethod
    def magic_formula_Fy(self, Fz, alpha, B, C, D, E=0.0):
        """Calculate lateral force Fy."""
        pass

    @abstractmethod
    def magic_formula_Fx(self, Fz, kappa, B, C, D, E=0.0):
        """Calculate longitudinal force Fx."""
        pass

    @abstractmethod
    def D_peak(self, D_ref, Fz, k_load, T, T_opt, sigma_left, sigma_right, Fz0=1000.0):
        """Calculate peak force coefficient D."""
        pass

    @abstractmethod
    def Calpha_brush(self, Cref, Fz, k_load_stiff, T, T_opt, sigma_left, sigma_right, T_factor_min, Fz0=1000.0):
        """Calculate cornering stiffness Calpha."""
        pass

    @abstractmethod
    def Ckappa_brush(self, Cref, Fz, k_load_stiff, T, T_opt, sigma_left, sigma_right, T_factor_min, Fz0=1000.0):
        """Calculate longitudinal stiffness Ckappa."""
        pass

    @abstractmethod
    def thermal_mult(self, T, T_opt, sigma_left, sigma_right, T_factor_min=None):
        """Calculate thermal multiplier."""
        pass

    @abstractmethod
    def apply_traction_ellipse(self, Fy0, Fx0, Fz, D_peak_lat, D_peak_lon, alpha=None, kappa=None, traction_ellipse_mode=None, **kwargs):
        """Apply traction ellipse constraint. traction_ellipse_mode: 'none' | 'current' | 'slip_dependent'."""
        pass

    @abstractmethod
    def invert_pacejka_fx(self, Fx_target, Fz, B, C, D, E, kappa_guess=0.0):
        """Invert Pacejka Fx to find kappa."""
        pass

    @abstractmethod
    def calculate_peak_slip_angle(self, Fz, B, C, D, E):
        """Calculate slip angle for peak lateral force."""
        pass

    @abstractmethod
    def calculate_peak_slip_ratio(self, Fz, B, C, D, E):
        """Calculate slip ratio for peak longitudinal force."""
        pass
