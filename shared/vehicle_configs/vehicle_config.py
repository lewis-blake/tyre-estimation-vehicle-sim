#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vehicle Configuration - Dynamic Loading Module

This module provides a centralized system for loading vehicle parameters.
Vehicle parameters MUST be loaded from a specific configuration file
(e.g., m4_gt3_vehicle_config.py, FS_vehicle_config.py) using apply_config().

NO DEFAULT VALUES are provided here - you must explicitly load a vehicle config.

Usage:
    import vehicle_config
    import m4_gt3_vehicle_config
    vehicle_config.apply_config(m4_gt3_vehicle_config)
    vehicle_config.reload_dependents()
"""

# VEHICLE PARAMETERS - NO DEFAULTS
# These will be set by apply_config() from a specific vehicle config module
# Attempting to use them before calling apply_config() will raise NameError

# MASS PROPERTIES
VEH_MASS = None  # Must be set by apply_config()
VEH_MASS_UNSPRUNG = None
VEH_MASS_SPRUNG = None

# GEOMETRY
VEH_WHEELBASE = None
VEH_COG_X = None
VEH_TRACK_F = None
VEH_TRACK_R = None
VEH_H_CG = None
VEH_ACKERMAN_PCT = None

# INERTIA
VEH_IZZ = None

# WHEEL PROPERTIES
VEH_R_WHEEL = None
VEH_I_WHEEL = None

# BRAKE SYSTEM
BRAKE_BIAS = None

# LATERAL LOAD TRANSFER DISTRIBUTION
LLTD = None

# AERODYNAMICS
CDA = None
CLA = None
COP_X = None
COP_Z = None
RHO_AIR = None

# STEERING
STEERING_RATIO = None

# TYRE PARAMETERS
# NOTE: Tyre parameters are in estimation/core/parameters.py
# (MagicFormulaParams class and MF_PARAMS_F/MF_PARAMS_R instances)


def apply_config(config_module):
    """Override all vehicle parameters from an alternative config module.

    After calling this, all module-level globals in vehicle_config are updated.
    Modules that already imported via ``from vehicle_config import X`` will NOT
    see the new values — call ``reload_dependents()`` afterwards.

    Parameters
    ----------
    config_module : module
        A module exposing the same top-level names (e.g. m4_gt3_vehicle_config).
    """
    import sys
    this = sys.modules[__name__]
    for name in [
        'VEH_MASS', 'VEH_MASS_UNSPRUNG', 'VEH_MASS_SPRUNG',
        'VEH_WHEELBASE', 'VEH_COG_X', 'VEH_TRACK_F', 'VEH_TRACK_R',
        'VEH_H_CG', 'VEH_ACKERMAN_PCT',
        'VEH_IZZ',
        'VEH_R_WHEEL', 'VEH_I_WHEEL',
        'BRAKE_BIAS',
        'LLTD',
        'CDA', 'CLA', 'COP_X', 'COP_Z', 'RHO_AIR',
        'STEERING_RATIO',
    ]:
        if hasattr(config_module, name):
            setattr(this, name, getattr(config_module, name))


def reload_dependents():
    """Reload modules that import vehicle_config values at top-level.

    This must be called AFTER ``apply_config()`` so that the ``from
    vehicle_config import X`` statements pick up the new values.
    """
    import importlib, sys
    module_names = [
        'algorithms.force_estimator',
        'algorithms.param_estimator',
        'algorithms.measurement_models',
        'core.parameters',
        'algorithms.force_estimation.kinematic',
    ]
    for mod_name in module_names:
        if mod_name in sys.modules:
            importlib.reload(sys.modules[mod_name])
