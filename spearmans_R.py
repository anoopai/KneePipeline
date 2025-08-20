
from math import log
from pathlib import Path
import os
import subprocess
import json
import sys
import time
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt
import shutil
from pymskt.mesh import get_icp_transform, cpd_register, non_rigidly_register
from pymskt import mesh
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from pymskt.mesh.meshTools import smooth_scalars_from_second_mesh_onto_base, transfer_mesh_scalars_get_weighted_average_n_closest
from pymskt.mesh import Mesh, BoneMesh

# Compite Spearman correlation of Change in Thickness and Change in T2 relaxation time at each point
# Constants and configurations
knee_to_use = ['aclr', 'clat', 'ctrl']
dir_path = '/dataNAS/people/anoopai/DESS_ACL_study'
code_dir_path = '/dataNAS/people/anoopai/KneePipeline/'
data_path = os.path.join(dir_path, 'data')
log_path = '/dataNAS/people/anoopai/KneePipeline/logs'
output_file = os.path.join(log_path, f't2_and_thickness_change.txt')
log_file_path = os.path.join(log_path, f'pipeline_DESS_errors.txt')
mean_path = os.path.join(code_dir_path, 'mean_data')
save_path = os.path.join(mean_path, f't2_and_thickness_change')
if not os.path.exists(save_path):
    os.makedirs(save_path)
mean_femur_path = os.path.join(mean_path, 'MeanFemur_mesh_B-only.vtk')

parameter1= 'Diff_thickness (mm)'
parameter2= 'Diff_T2_mean'
visits = ['VISIT-2', 'VISIT-3', 'VISIT-4', 'VISIT-5']
knees = ['aclr', 'clat', 'ctrl']


analysis_complete = []
data_all = []

for subject in os.listdir(data_path):
    subject_path = os.path.join(data_path, subject)

    if os.path.isdir(subject_path):
        # Iterate through each visit in the subject path
        for visit in os.listdir(subject_path):
            if visit not in ['VISIT-6', 'VISIT-1']:  # Exclude VISIT-6
                visit_path = os.path.join(subject_path, visit)

                if os.path.isdir(visit_path):
                    # Iterate through each knee type in the visit path
                    for knee in os.listdir(visit_path):
                        knee_path = os.path.join(visit_path, knee)
                        
                        if os.path.isdir(knee_path) and knee in knee_to_use:
                            path_image = os.path.join(knee_path, 'scans/qdess')
                            path_save = os.path.join(knee_path, f'results_nsm')
                            femur_path = os.path.join(path_save, 'femur_mesh_NSM_orig_reg2mean.vtk')
                            
                            sub_component= Path(knee_path).parts[6]  # '11-P'
                            visit_component = Path(knee_path).parts[7]  # 'VISIT-1'
                            knee_component = Path(knee_path).parts[8]  # 'clat'
                            try:
                            
                                if os.path.exists(femur_path):
                                    femur_mesh = BoneMesh(femur_path)
                                    thickness_change_all = {}
                                    t2_change_all = {}
                                    scalar_names =[parameter1, parameter2]
                                    data= {'sub': sub_component,
                                            'visit': visit_component,
                                            'knee': knee_component
                                            }

                                    for scalar_name in scalar_names:
                                        data[scalar_name] = vtk_to_numpy(femur_mesh.GetPointData().GetArray(scalar_name))
                                    
                                    data_all.append(data)
                                else:
                                    print(f"Femur mesh not found for {subject} {visit} {knee}")
                                    continue    
                            except Exception as e:
                                print(f"Error processing {subject} {visit} {knee}: {e}")
                                continue

# Create a dictionary to store filtered results
filtered_results = {knee: {visit: [] for visit in visits} for knee in knees}

# Filter data
for d in data_all:
    knee_type = d['knee']
    visit_type = d['visit']
    if knee_type in knees and visit_type in visits:
        filtered_results[knee_type][visit_type].append(d)

# Access filtered results
aclr_results = filtered_results['aclr']
clat_results = filtered_results['clat']
ctrl_results = filtered_results['ctrl']

# Dictionary to hold results for each knee and visit
results = {knee: {visit: {"spearman_r": None, "p_value": None} for visit in visits} for knee in knees}

# Calculate Pearson's r, R^2, and p-value for each knee and visit
for knee in knees:
    for visit in visits:
        # Gather all parameter2 and parameter1 data
        parameter2_list = []
        parameter1_list = []
        
        for d in filtered_results[knee][visit]:
            parameter2_list.append(d[parameter2])
            parameter1_list.append(d[parameter1])
        
        # Convert to NumPy arrays
        parameter2_array = np.array(parameter2_list)  # Shape (n_subjects, 10000)
        parameter1_array = np.array(parameter1_list)  # Shape (n_subjects, 10000)

        # Initialize arrays to store results
        r_values = np.empty(parameter1_array.shape[1])  # Shape (10000,)
        p_values = np.empty(parameter1_array.shape[1])  # Shape (10000,)

        # Calculate Pearson's r, R^2, and p-value for each point
        for i in range(parameter1_array.shape[1]):
            # Calculate the ratio of zeros in the columns
            bone_ratio_param1 = np.sum(parameter1_array[:, i] == -200) / len(parameter1_array[:, i])
            bone_ratio_param2 = np.sum(parameter2_array[:, i] == -200)  / len(parameter2_array[:, i])
            fc_edges_ratio_param1 = np.sum(parameter1_array[:, i] == -100) / len(parameter1_array[:, i])
            fc_edges_ratio_param2 = np.sum(parameter2_array[:, i] == -100) / len(parameter2_array[:, i])
            
            # Check if more than x% of the values are zeros
            if bone_ratio_param1 > 0.1 or bone_ratio_param2 > 0.1:
                r_values[i] = -200.0  # Assign 0.0 for r-values
                p_values[i] = 1.0  # Assign 1.0 for p-values
            elif fc_edges_ratio_param1 > 0.1 or fc_edges_ratio_param2 > 0.1:
                r_values[i] = -100.0  # Assign 0.0 for r-values
                p_values[i] = 1.0  # Assign 1.0 for p-values
            else:
                # Calculate Pearson's r and p-value
                r_values[i], p_values[i] = spearmanr(parameter1_array[:, i], parameter2_array[:, i])

        # Store results in the dictionary
        results[knee][visit] = {
            "spearman_r": r_values,
            "p_value": p_values
        }
        
correlation_scalars= list(results[knee][visit].keys())

for knee in knees:
    for visit in visits:
        femur_mesh = BoneMesh(mean_femur_path)
        for scalar_name in correlation_scalars:
            # print(knee, visit, scalar_name)
            # Add the scalar data to the femur mesh
            femur_mesh.point_data[scalar_name] = results[knee][visit][scalar_name]
        femur_mesh.save_mesh(os.path.join(save_path, f'MeanFemur_mesh_B-only_{knee}_{visit}.vtk'))