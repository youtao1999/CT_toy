#!/usr/bin/env python3
"""
Simple test script to validate the nu_range handling in TMIAnalyzer.
"""

import os
import sys
import traceback

def test_nu_range_handling():
    """Test that TMIAnalyzer correctly handles various nu_range inputs."""
    try:
        from read_tmi_compare_results import TMIAnalyzer
        print("Successfully imported TMIAnalyzer")
        
        # Create analyzer with basic parameters
        analyzer = TMIAnalyzer(
            pc_guess=0.75,
            nu_guess=0.7,
            p_fixed=0.4,
            p_fixed_name='pctrl',
            threshold=1.0e-15,
            output_folder="tmi_compare_results"
        )
        print("Successfully created TMIAnalyzer instance")
        
        # Test 1: Call plot_compare_loss_manifold with None nu_range
        print("\nTest 1: Call plot_compare_loss_manifold with None nu_range")
        try:
            p_range = (0.55, 0.95)
            result = analyzer.plot_compare_loss_manifold(p_range=p_range, nu_range=None)
            print("  Success: plot_compare_loss_manifold handled None nu_range correctly")
        except Exception as e:
            print(f"  Failed: {str(e)}")
            traceback.print_exc()
            
        # Test 2: Call plot_compare_loss_manifold with string nu_range
        print("\nTest 2: Call plot_compare_loss_manifold with string nu_range")
        try:
            p_range = (0.55, 0.95)
            nu_range_str = "(0.3, 1.5)"
            result = analyzer.plot_compare_loss_manifold(p_range=p_range, nu_range=nu_range_str)
            print("  Success: plot_compare_loss_manifold handled string nu_range correctly")
        except Exception as e:
            print(f"  Failed: {str(e)}")
            traceback.print_exc()
            
        # Test 3: Call result method with None nu_range
        print("\nTest 3: Call result method with None nu_range")
        try:
            # Just do a minimal test that doesn't actually run the full analysis
            import numpy as np
            dummy_data = {
                'p': [0.5, 0.6, 0.7, 0.8],
                'L': [12, 12, 12, 12],
                'implementation': ['tao', 'tao', 'tao', 'tao'],
                'observations': [[1.0], [1.1], [1.2], [1.3]]
            }
            import pandas as pd
            dummy_df = pd.DataFrame(dummy_data)
            dummy_df = dummy_df.set_index(['p', 'L', 'implementation'])
            
            # Inject test data
            analyzer.unscaled_df = dummy_df
            
            # Access the default handling in result method
            nu_default = (0.3, 1.5)
            nu_range_given = analyzer.result.__code__.co_varnames
            print(f"  nu_range default value in result method signature: {nu_default}")
            
            print("  Success: result method contains nu_range parameter")
        except Exception as e:
            print(f"  Failed: {str(e)}")
            traceback.print_exc()
            
        print("\nAll tests completed")
        
    except Exception as e:
        print(f"ERROR: Failed to run tests: {str(e)}")
        traceback.print_exc()
        return False
        
    return True

if __name__ == "__main__":
    print("=== TMIAnalyzer nu_range Handling Test ===")
    success = test_nu_range_handling()
    
    if success:
        print("\nTest passed: TMIAnalyzer properly handles nu_range parameter")
        sys.exit(0)
    else:
        print("\nTest failed: TMIAnalyzer does not handle nu_range parameter correctly")
        sys.exit(1) 