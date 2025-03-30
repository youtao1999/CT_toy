#!/bin/bash

# Set up environment - first try to use conda environment if available
if command -v conda &> /dev/null; then
    echo "Conda found, setting up environment..."
    # Create or update conda environment
    conda env update -f environment.yml --prune
    conda activate tmi_analysis
else
    echo "Conda not found, using pip..."
    # Set up virtual environment with pip
    python3 -m venv tmi_env
    source tmi_env/bin/activate
    
    # Install dependencies with pip
    pip install -r requirements.txt
fi

# Make sure we have numpy < 2.0.0
pip uninstall -y numpy
pip install 'numpy<2.0.0'

# Run analysis
echo "Running TMI analysis..."
python run_analysis.py

# Cleanup
echo "Analysis complete." 