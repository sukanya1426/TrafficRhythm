#!/bin/bash

# Create a virtual environment
python3 -m venv traffic_env

# Activate the virtual environment
source traffic_env/bin/activate

# Install required Python packages
pip install scikit-learn numpy pandas matplotlib seaborn

echo "Virtual environment 'traffic_env' created and dependencies installed."
echo "To activate the environment, run: source traffic_env/bin/activate"