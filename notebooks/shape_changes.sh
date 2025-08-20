#!/bin/bash

# Request resources interactively
salloc --gres=gpu:1 --cpus-per-task=6

# Load Conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate knee_pipeline

# Set DISPLAY variable for X11 if needed
export DISPLAY=:0.0

# Check if GPU is available
if ! python -c "import torch; assert torch.cuda.is_available()" 2> /dev/null; then
    echo "CUDA is not available. Mapping model to CPU."
    model_loading_kwargs="map_location=torch.device('cpu')"
else
    model_loading_kwargs=""
fi

echo "Running shape_change_between_knees_raw_by_visit.py...................."
python shape_change_between_knees_raw_by_visit.py
echo "Completed shape_change_between_knees_raw_by_visit.py"

echo "Running shape_change_overtime_raw.py...................."
python shape_change_overtime_raw.py
echo "Completed shape_change_overtime_raw.py"

# echo shape_changes_of_CTRLs_along_Bvectors.py
# python shape_changes_of_CTRLs_along_Bvectors.py
# echo "Completed shape_changes_of_CTRLs_along_Bvectors.py"

# echo "Running shape_change_between_knees_raw.py...................."
# python shape_change_between_knees_raw.py
# echo "Completed shape_change_between_knees_raw.py"

# echo "Running shape_change_overtime_SD.py...................."
# python shape_change_overtime_SD.py
# echo "Completed shape_change_overtime_SD.py"

# echo "Running shape_change_between_knees_SD.py...................."
# python shape_change_between_knees_SD.py
# echo "Completed shape_change_between_knees_SD.py"

# echo "Running shape_change_between_knees_SD_by_visit.py...................."
# python shape_change_between_knees_SD_by_visit.py
# echo "Completed shape_change_between_knees_SD_by_visit.py"
