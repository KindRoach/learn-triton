#!/bin/bash

# Check if a Python script is provided as argument
if [ $# -eq 0 ]; then
    echo "Usage: $0 <python_script.py>"
    exit 1
fi

python_script_name="$1"

# Check if the Python script exists
if [ ! -f "$python_script_name" ]; then
    echo "Error: Python script '$python_script_name' not found"
    exit 1
fi

# Extract the base name without extension for the output file
base_name=$(basename "$python_script_name" .py)
output_file="${base_name}_profile"

echo "Profiling $python_script_name with NCU..."
ncu --set full --export "$output_file" python "$python_script_name"