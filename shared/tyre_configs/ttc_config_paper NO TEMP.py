
# TTC Validation Tyre Model Configuration

def get_config():
    return {
        'tyre_model_type': 'TTC_Validation',

        # Lateral Parameters
        # Fixed Parameters (from FIXED_PARAMS_LATERAL)
        'C': 1.3,
        'Fz0': 800.0,
        'bias': 0.0,
        'T_opt': 60.0,
        'T_disp': 1e6,
        'T_stiff0': 1.0,
        'p_load0': 1.0,
        'p_stiff0': 1.0,

        # Estimated Parameters (from INITIAL_GUESS_LATERAL)
        'E': -1.5,
        'D_ref': 1.74,
        'C_ref': 59137.0,
        'k_load': -0.15,
        'k_load_stiff': 2400.0,
        'T_stiff1': -0.0,
        'T_stiff2': 0.0,
        'p_load1': 0.0,
        'p_load2': 0.0,
        'p_stiff1': 0.0,
        'p_stiff2': 0.0,

        # Longitudinal Parameters
        # Fixed Parameters (from FIXED_PARAMS_LONGITUDINAL)
        'C_x': 1.65,
        'Fz0': 1000.0, # Note: Fz0 can be different for lon/lat in TTC config structure
        'bias_x': 0.0,
        'T_opt_x': 60.0,
        'T_disp_x': 1e6,
        'T_stiff0_x': 1.0,
        'p_load0_x': 1.0,
        'p_stiff0_x': 1.0,
        
        # Estimated Parameters (from INITIAL_GUESS_LONGITUDINAL)
        'E_x': -1.0,
        'D_ref_x': 1.97,
        'C_ref_x': 31301.0,
        'k_load_x': -0.15,
        'k_load_stiff_x': 2400.0,
        'T_stiff1_x': -0.0,
        'T_stiff2_x': 0.0,
        'p_load1_x': 0.0,
        'p_load2_x': 0.0,
        'p_stiff1_x': 0.0,
        'p_stiff2_x': 0.0,

        # Default Pressure (since main_sim doesn't simulate it yet)
        'default_pressure_kPa': 120.0,
        
        # Traction ellipse: "none" | "current" | "slip_dependent"
        'traction_ellipse': 'slip_dependent',
        
         # Degradation parameters (Placeholder for TTC, using same structure as standard)
        'deg': {
            'D_ref': { 'k_time': 0.0, 'k_Fz': 0.0, 'k_alpha': 0.0, 'k_T': 0.0, 'min_val': 0.3 },
            'Cref': { 'k_time': 0.0, 'k_Fz': 0.0, 'k_alpha': 0.0, 'k_T': 0.0, 'min_val': 20.0 },
            'D_ref_lon': { 'k_time': 0.0, 'k_Fz': 0.0, 'k_kappa': 0.0, 'k_T': 0.0, 'min_val': 0.3 },
            'Cref_lon': { 'k_time': 0.0, 'k_Fz': 0.0, 'k_kappa': 0.0, 'k_T': 0.0, 'min_val': 20.0 }
        }
    }
