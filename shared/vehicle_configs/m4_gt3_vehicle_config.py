#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BMW M4 GT3 Vehicle Configuration

Approximate parameters for the BMW M4 GT3 race car.
These override the default (simulator) vehicle_config values
when processing real car telemetry data.

Sources:
- BMW Motorsport public data and homologation documents
- Typical GT3 class specifications
"""

# MASS PROPERTIES
VEH_MASS = 1310.0  # Minimum weight with driver [kg] (BoP dependent)
VEH_MASS_UNSPRUNG = 200.0  # Estimated total unsprung mass [kg] (~50 kg/corner)
VEH_MASS_SPRUNG = VEH_MASS - VEH_MASS_UNSPRUNG  # Sprung mass [kg]

# GEOMETRY
VEH_WHEELBASE = 2.857  # Wheelbase [m] (BMW M4 platform)
VEH_COG_X = 0.53  # COG position ratio — Front-biased (53% Front)
VEH_TRACK_F = 1.630  # Front track width [m]
VEH_TRACK_R = 1.610  # Rear track width [m]
VEH_H_CG = 0.45  # CG height [m] — GT3 cars with roof, higher than open-wheel
VEH_ACKERMAN_PCT = 0.0  # Ackerman steering percentage

# INERTIA
VEH_IZZ = 2800.0  # Yaw moment of inertia [kg·m²] — estimated for GT3-class car

# WHEEL PROPERTIES
VEH_R_WHEEL = 0.3  # Effective rolling radius [m] (18" wheels with slick tyres)
VEH_I_WHEEL = 2.5  # Wheel + tyre rotational inertia [kg·m²] — estimated

# BRAKE SYSTEM
BRAKE_BIAS = 0.60  # Brake bias — fraction to front axle (typical GT3)

# LATERAL LOAD TRANSFER DISTRIBUTION (LLTD)
# LLTD controls how total lateral load transfer is distributed between axles
# LLTD = 0.0: All lateral load transfer at rear axle
# LLTD = 0.5: Equal distribution (50% front, 50% rear)
# LLTD = 1.0: All lateral load transfer at front axle
LLTD = 0.50  # Lateral load transfer distribution (0-1)

# AERODYNAMICS
CDA = 0  # Drag area [m²] (Cd ~0.40 × frontal area ~3.0 m²)
CLA = -0  # Downforce area [m²] (negative = downforce) — GT3 with large wing
COP_X = 0.45  # Center of pressure X position ratio
COP_Z = 0.3  # Center of pressure height [m]
RHO_AIR = 1.225  # Air density [kg/m³]

# STEERING
STEERING_RATIO = 12.5  # Steering ratio (steering wheel angle / road wheel angle)
                      # Set to actual value (e.g., 12.5) if delta in data is steering wheel angle
