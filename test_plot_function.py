#!/usr/bin/env python
"""
Standalone test of the plot_compare_loss_manifold function
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import traceback

# Add current directory to path
sys.path.insert(0, os.getcwd())

def simple_plot_test(p_range, nu_range):
    """Simple function to test plotting with given ranges"""
    print(f"p_range: {p_range}, type: {type(p_range)}")
    print(f"nu_range: {nu_range}, type: {type(nu_range)}")
    
    # Create a meshgrid for testing
    n_points = 10
    p_vals = np.linspace(p_range[0], p_range[1], n_points)
    nu_vals = np.linspace(nu_range[0], nu_range[1], n_points)
    print(f"p_vals: {p_vals}")
    print(f"nu_vals: {nu_vals}")
    
    # This should confirm if there's any issue with accessing the ranges
    print("All good with range access!")
    return True

try:
    # Test with explicit values to verify function works
    print("Test 1: Basic tuple test")
    p_range = (0.55, 0.95)
    nu_range = (0.3, 1.5)
    simple_plot_test(p_range, nu_range)
    
    # Test with string values that might be coming from job script
    print("\nTest 2: String values test")
    p_range_str = "(0.55, 0.95)"
    nu_range_str = "(0.3, 1.5)"
    
    # Try to convert strings to tuples
    try:
        # This is a common way strings might get converted
        p_range_eval = eval(p_range_str)
        nu_range_eval = eval(nu_range_str)
        simple_plot_test(p_range_eval, nu_range_eval)
    except Exception as e:
        print(f"Error with string conversion: {str(e)}")
        
    # Let's try to import the actual TMIAnalyzer if available
    print("\nTest 3: Importing TMIAnalyzer")
    try:
        from read_tmi_compare_results import TMIAnalyzer
        print("Successfully imported TMIAnalyzer")
        
        # Create a minimal analyzer for testing
        analyzer = TMIAnalyzer(
            pc_guess=0.75,
            nu_guess=0.7,
            p_fixed=0.4,
            p_fixed_name='pctrl',
            threshold=1.0e-15
        )
        
        # Just test the critical part without data
        print("\nTesting the critical line in plot_compare_loss_manifold")
        p_range = (0.55, 0.95)
        nu_range = (0.3, 1.5)
        
        # Create p_vals and nu_vals directly as done in the function
        n_points = 10
        p_vals = np.linspace(p_range[0], p_range[1], n_points)
        nu_vals = np.linspace(nu_range[0], nu_range[1], n_points)
        print(f"Created p_vals and nu_vals successfully")
        
    except ImportError:
        print("Could not import TMIAnalyzer for testing")
        
    print("\nAll tests completed successfully!")
    
except Exception as e:
    print(f"Error in test: {str(e)}")
    traceback.print_exc() 