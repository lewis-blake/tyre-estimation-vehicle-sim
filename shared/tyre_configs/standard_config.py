
# Standard Tyre Model Configuration (Legacy/Current)

def get_config():
    return {
        'tyre_model_type': 'Standard',
        
        # Lateral parameters
        'D_ref': 1.5,           # Peak force scaling factor
        'k_load': -0.2,          # Load sensitivity exponent
        'Cref': 30.0,            # Cornering stiffness reference
        'k_load_stiff': -0.6,    # Load sensitivity for stiffness
        'C': 1.3,                # Shape factor
        'E': -1.0,               # Curvature factor
        
        # Longitudinal parameters
        'D_ref_lon': 1.5,        # Peak force scaling factor (longitudinal)
        'k_load_lon': -0.2,      # Load sensitivity exponent (longitudinal)
        'Cref_lon': 30.0,        # Longitudinal stiffness reference
        'k_load_stiff_lon': -0.6, # Load sensitivity for stiffness (longitudinal)
        'C_lon': 1.65,           # Shape factor (longitudinal)
        'E_lon': -1.0,           # Curvature factor (longitudinal)
        
        # Temperature parameters (shared)
        'T_opt': 60.0,           # Optimal temperature [°C]
        'sigma_left': 15.0,      # Temperature spread (left side) [°C]
        'sigma_right': 15.0,     # Temperature spread (right side) [°C]
        'T_opt_BCD': 60.0,       # Optimal temperature for BCD [°C]
        'sigma_left_BCD': 80.0,  # Temperature spread for BCD (left) [°C]
        'sigma_right_BCD': 100.0, # Temperature spread for BCD (right) [°C]
        'T_factor_min_BCD': 0.9, # Minimum temperature factor for BCD
        'T_factor_min': 0.7,     # Minimum temperature factor for Peak Force (D)
        'Fz0': 1000.0,           # Nominal load [N]
        
        # Traction ellipse: "none" | "current" | "slip_dependent"
        'traction_ellipse': 'slip_dependent',
        
        # Degradation parameters
        'deg': {
            'D_ref': {
                'k_time': 0.00,      # Time-based degradation rate [1/s]
                'k_Fz': 0.000,       # Load-based degradation rate [1/(N·s)]
                'k_alpha': 0.0,      # Slip angle-based degradation rate [1/(rad·s)]
                'k_T': 0.000,        # Temperature-based degradation rate [1/(°C·s)]
                'min_val': 0.3        # Minimum value
            },
            'Cref': {
                'k_time': 0.00,
                'k_Fz': 0.0000,
                'k_alpha': 0.000,
                'k_T': 0.000,
                'min_val': 20.0
            },
            'D_ref_lon': {
                'k_time': 0.00,
                'k_Fz': 0.000,
                'k_kappa': 0.0,      # Slip ratio-based degradation rate [1/(·s)]
                'k_T': 0.000,
                'min_val': 0.3
            },
            'Cref_lon': {
                'k_time': 0.00,
                'k_Fz': 0.0000,
                'k_kappa': 0.000,
                'k_T': 0.000,
                'min_val': 20.0
            }
        }
    }
