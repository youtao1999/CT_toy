#!/usr/bin/env python3
"""
Simple diagnostic script to check data files and environment for TMI analysis.
"""

import os
import sys
import glob
import traceback

# Configuration - same as analysis parameters
p_fixed = 0.4
p_fixed_name = "pctrl"
threshold = 1.0e-15
output_folder = "tmi_compare_results"

def check_environment():
    """Check Python environment and dependencies."""
    print("=== Environment Check ===")
    print(f"Python version: {sys.version}")
    
    # Check NumPy
    try:
        import numpy as np
        print(f"NumPy version: {np.__version__}")
        if np.__version__.startswith('2.'):
            print("WARNING: Using NumPy 2.x which may cause compatibility issues")
    except ImportError:
        print("ERROR: NumPy not found")
    
    # Check Pandas
    try:
        import pandas as pd
        print(f"Pandas version: {pd.__version__}")
    except ImportError:
        print("ERROR: Pandas not found")
    
    # Check Matplotlib
    try:
        import matplotlib
        print(f"Matplotlib version: {matplotlib.__version__}")
    except ImportError:
        print("ERROR: Matplotlib not found")
    
    # Check h5py
    try:
        import h5py
        print(f"h5py version: {h5py.__version__}")
    except ImportError:
        print("ERROR: h5py not found")
    
    # Check FSS.DataCollapse
    try:
        from FSS.DataCollapse import DataCollapse
        print("FSS.DataCollapse module found")
    except ImportError:
        print("ERROR: FSS.DataCollapse module not found")
    
    print()

def check_data_files():
    """Check for data files needed for analysis."""
    print("=== Data Files Check ===")
    
    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)
    print(f"Output folder: {output_folder} (exists: {os.path.exists(output_folder)})")
    
    # Check CSV file
    csv_filename = os.path.join(
        output_folder,
        f'tmi_compare_results_{p_fixed_name}{p_fixed:.3f}_threshold{threshold:.1e}.csv'
    )
    csv_exists = os.path.exists(csv_filename)
    print(f"CSV file: {csv_filename} (exists: {csv_exists})")
    if csv_exists:
        print(f"  Size: {os.path.getsize(csv_filename) / 1024:.2f} KB")
    
    # Check H5 files
    file_pattern = f'sv_comparison_L*_{p_fixed_name}{p_fixed:.3f}_p*'
    h5_files = glob.glob(file_pattern)
    print(f"H5 file pattern: {file_pattern}")
    print(f"Found {len(h5_files)} matching directories")
    
    if h5_files:
        for directory in h5_files[:5]:  # Show first 5 directories only
            print(f"  {directory}")
            l_value = directory.split('_L')[1].split('_')[0]
            h5_filename = os.path.join(directory, f'final_results_L{l_value}.h5')
            if os.path.exists(h5_filename):
                print(f"    Found H5 file: {h5_filename} (Size: {os.path.getsize(h5_filename) / 1024:.2f} KB)")
            else:
                print(f"    H5 file not found: {h5_filename}")
    
        if len(h5_files) > 5:
            print(f"  ... and {len(h5_files) - 5} more directories")
    
    print()

def check_read_tmi_compare_results():
    """Check if read_tmi_compare_results.py is properly set up."""
    print("=== read_tmi_compare_results.py Check ===")
    script_path = "read_tmi_compare_results.py"
    
    if not os.path.exists(script_path):
        print(f"ERROR: {script_path} not found")
        return
    
    # Check file size and modification time
    size = os.path.getsize(script_path)
    mtime = os.path.getmtime(script_path)
    import time
    print(f"{script_path} found:")
    print(f"  Size: {size / 1024:.2f} KB")
    print(f"  Last modified: {time.ctime(mtime)}")
    
    # Check nu_range parameter
    try:
        with open(script_path, 'r') as f:
            content = f.read()
            if "nu_range=None" in content:
                print("  nu_range parameter is properly defined with default value")
            else:
                print("  WARNING: nu_range parameter might not have a default value")
            
            if "if nu_range is None:" in content:
                print("  nu_range handling code found (using default if None)")
            else:
                print("  WARNING: No code to handle None nu_range found")
    except Exception as e:
        print(f"  Error reading file: {e}")
    
    print()

if __name__ == "__main__":
    print("=== TMI Analysis Diagnostic Tool ===")
    print("Running diagnostics to help debug analysis issues...")
    print()
    
    try:
        check_environment()
        check_data_files()
        check_read_tmi_compare_results()
        
        print("=== Diagnostics Complete ===")
        print("If you see any ERROR messages above, those need to be fixed before running the analysis.")
        
    except Exception as e:
        print(f"ERROR during diagnostics: {e}")
        traceback.print_exc()
        sys.exit(1) 