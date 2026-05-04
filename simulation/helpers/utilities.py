#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 18:44:26 2025

@author: lewisblake
"""

# utilities.py
import numpy as np

def initialise_state_history(initial_state_dict, max_steps):
    """
    Creates a pre-allocated NumPy array for logging and a map that links
    state names to columns in the array.
    """
    history_map = []
    initial_state_row_vector = np.array([])
    current_column_index = 0
    
    sorted_keys = sorted(initial_state_dict.keys())
    
    for key in sorted_keys:
        value = initial_state_dict[key]
        value_as_row = np.ravel(value)
        num_elements = value_as_row.size
        
        initial_state_row_vector = np.concatenate((initial_state_row_vector, value_as_row))
        
        entry = {
            'name': key,
            'startIndex': current_column_index,
            'endIndex': current_column_index + num_elements - 1,
            'originalSize': np.shape(value)
        }
        history_map.append(entry)
        current_column_index += num_elements
        
    total_cols = initial_state_row_vector.size
    history_log = np.zeros((max_steps, total_cols))
    
    return history_log, history_map

def update_step_data(history_log, current_data_dict, history_map, step_index):
    """
    Stores the current step's data into the correct row and columns of the history log.
    """
    if step_index >= history_log.shape[0]:
        print(f"Warning: Step index {step_index} is out of bounds for history log.")
        return history_log

    for entry in history_map:
        name = entry['name']
        start_col = entry['startIndex']
        end_col = entry['endIndex']
        
        if name in current_data_dict:
            value = current_data_dict[name]
            flat_value = np.ravel(value)
            
            expected_size = end_col - start_col + 1
            if flat_value.size != expected_size:
                if flat_value.size > expected_size:
                    flat_value = flat_value[:expected_size]
                else:
                    padded_value = np.full(expected_size, np.nan)
                    padded_value[:flat_value.size] = flat_value
                    flat_value = padded_value
            
            history_log[step_index, start_col:end_col+1] = flat_value
        else:
            history_log[step_index, start_col:end_col+1] = np.nan
            
    return history_log
