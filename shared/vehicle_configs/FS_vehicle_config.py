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
VEH_MASS = 320  # Minimum weight with driver [kg] (BoP dependent)
VEH_MASS_UNSPRUNG = 42  # Total unsprung mass [kg] (2 * (m_us_f + m_us_r) = 2*(9+12))
VEH_MASS_SPRUNG = VEH_MASS - VEH_MASS_UNSPRUNG  # Sprung mass [kg]

# GEOMETRY
VEH_WHEELBASE = 1.6  # Wheelbase [m] (BMW M4 platform)
VEH_COG_X = 0.5  # COG position ratio — slightly front-biased (front-engine)
VEH_TRACK_F = 1.25  # Front track width [m]
VEH_TRACK_R = 1.25  # Rear track width [m]
VEH_H_CG = 0.2  # CG height [m] — GT3 cars with roof, higher than open-wheel
VEH_ACKERMAN_PCT = 0  # Ackerman steering percentage

# INERTIA
VEH_IZZ = 200  # Yaw moment of inertia [kg·m²] — estimated for GT3-class car

# WHEEL PROPERTIES
VEH_R_WHEEL = 0.22  # Effective rolling radius [m] (18" wheels with slick tyres)
VEH_I_WHEEL = 1.0  # Wheel + tyre rotational inertia [kg·m²]

# BRAKE SYSTEM
BRAKE_BIAS = 0.63  # Brake bias — fraction to front axle (typical GT3)

# LATERAL LOAD TRANSFER DISTRIBUTION (LLTD)
# LLTD controls how total lateral load transfer is distributed between axles
# LLTD = 0.0: All lateral load transfer at rear axle
# LLTD = 0.5: Equal distribution (50% front, 50% rear)
# LLTD = 1.0: All lateral load transfer at front axle
LLTD = 0.5223  # Lateral load transfer distribution (front fraction, from suspension roll stiffness)

# AERODYNAMICS
CDA = 0  # Drag area [m²] (Cd ~0.40 × frontal area ~3.0 m²)
CLA = 0  # Downforce area [m²] (negative = downforce) — GT3 with large wing
COP_X = 0.5  # Center of pressure X position ratio
COP_Z = 0.2  # Center of pressure height [m]
RHO_AIR = 1.225  # Air density [kg/m³]

# STEERING
STEERING_RATIO = 1.0  # Steering ratio (steering wheel angle / road wheel angle)
                      # FS cars typically have direct steering (ratio ~1.0)
