"""
Tyre Model Dispatcher

This module now serves as a facade/dispatcher for the active tyre model.
It maintains backward compatibility by exposing the same functions as before,
but delegates execution to the currently active model instance (Standard or New).

Usage:
    import core.tyre_model as tm
    
    # Initialize with specific model (called by estimator)
    tm.set_active_tyre_model(TyreModelType.PACEJKA_NEW)
    
    # Functions work as before, but use the new logic
    Fy = tm.magic_formula_Fy(...)
"""

from typing import Optional, Union, Any
from config.master_config import TyreModelType
from .tyre_models.base import TyreModelBase
from .tyre_models.standard import StandardPacejka
from .tyre_models.new_pacejka import NewPacejka
from .tyre_models.ttc_wrapper import TtcPacejka

# Global Active Model State

_ACTIVE_MODEL: TyreModelBase = StandardPacejka()
_ACTIVE_MODEL_TYPE: TyreModelType = TyreModelType.PACEJKA_STANDARD
_TRACTION_ELLIPSE_MODE: str = "current"

def set_active_tyre_model(model_type: Union[TyreModelType, str]):
    """
    Set the globally active tyre model.
    Call this at the start of your estimation script.
    """
    global _ACTIVE_MODEL, _ACTIVE_MODEL_TYPE

    # Normalise string input to enum first
    if isinstance(model_type, str):
        try:
            model_type = TyreModelType(model_type)
        except ValueError:
            # Fallback: default to standard if string is unrecognised
            model_type = TyreModelType.PACEJKA_STANDARD

    if model_type == TyreModelType.PACEJKA_TTC:
        _ACTIVE_MODEL = TtcPacejka()
        _ACTIVE_MODEL_TYPE = TyreModelType.PACEJKA_TTC
        print(f" Tyre Model switched to: TTC PACEJKA")
    elif model_type == TyreModelType.PACEJKA_NEW:
        _ACTIVE_MODEL = NewPacejka()
        _ACTIVE_MODEL_TYPE = TyreModelType.PACEJKA_NEW
        print(f" Tyre Model switched to: NEW PACEJKA")
    else:
        _ACTIVE_MODEL = StandardPacejka()
        _ACTIVE_MODEL_TYPE = TyreModelType.PACEJKA_STANDARD
        print(f" Tyre Model switched to: STANDARD PACEJKA")

def get_active_model() -> TyreModelBase:
    return _ACTIVE_MODEL

def set_traction_ellipse_mode(mode: str):
    """Set traction ellipse mode: 'none' | 'current' | 'slip_dependent'. Call after set_active_tyre_model."""
    global _TRACTION_ELLIPSE_MODE
    _TRACTION_ELLIPSE_MODE = mode

# Facade Functions (Delegate to _ACTIVE_MODEL)

def thermal_mult(T, T_opt, sigma_left, sigma_right, T_factor_min=None):
    return _ACTIVE_MODEL.thermal_mult(T, T_opt, sigma_left, sigma_right, T_factor_min)

def D_peak(D_ref, Fz, k_load, T, T_opt, sigma_left, sigma_right, T_factor_min, Fz0=1000.0):
    """Calculate peak force coefficient (or Peak Force in New Model)."""
    return _ACTIVE_MODEL.D_peak(D_ref, Fz, k_load, T, T_opt, sigma_left, sigma_right, T_factor_min, Fz0)

def Calpha_brush(Cref, Fz, k_load_stiff, T, T_opt, sigma_left, sigma_right, T_factor_min, Fz0=1000.0):
    return _ACTIVE_MODEL.Calpha_brush(Cref, Fz, k_load_stiff, T, T_opt, sigma_left, sigma_right, T_factor_min, Fz0)

def Ckappa_brush(Cref, Fz, k_load_stiff, T, T_opt, sigma_left, sigma_right, T_factor_min, Fz0=1000.0):
    return _ACTIVE_MODEL.Ckappa_brush(Cref, Fz, k_load_stiff, T, T_opt, sigma_left, sigma_right, T_factor_min, Fz0)

def magic_formula_Fy(Fz, alpha, B, C, D, E=0.0):
    return _ACTIVE_MODEL.magic_formula_Fy(Fz, alpha, B, C, D, E)

def magic_formula_Fx(Fz, kappa, B, C, D, E=0.0):
    return _ACTIVE_MODEL.magic_formula_Fx(Fz, kappa, B, C, D, E)

def apply_traction_ellipse(Fy0, Fx0, Fz, D_peak_lat, D_peak_lon, alpha=None, kappa=None, **kwargs):
    return _ACTIVE_MODEL.apply_traction_ellipse(
        Fy0, Fx0, Fz, D_peak_lat, D_peak_lon, alpha, kappa,
        traction_ellipse_mode=_TRACTION_ELLIPSE_MODE, **kwargs
    )

def invert_pacejka_fx(Fx_target, Fz, B, C, D, E, kappa_guess=0.0):
    return _ACTIVE_MODEL.invert_pacejka_fx(Fx_target, Fz, B, C, D, E, kappa_guess)

def calculate_peak_slip_angle(Fz, B, C, D, E):
    return _ACTIVE_MODEL.calculate_peak_slip_angle(Fz, B, C, D, E)

def calculate_peak_slip_ratio(Fz, B, C, D, E):
    return _ACTIVE_MODEL.calculate_peak_slip_ratio(Fz, B, C, D, E)
