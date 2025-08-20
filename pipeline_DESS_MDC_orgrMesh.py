
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

scalar_names= ['Diff_thickness (mm)']
visits = ['VISIT-2']
knees_to_use = ['ctrl']
knee = 'ctrl'
visit = 'VISIT-2'

dir_path = '/dataNAS/people/anoopai/DESS_ACL_study'
data_path = os.path.join(dir_path, 'data')
code_dir_path = '/dataNAS/people/anoopai/KneePipeline/'
log_path = '/dataNAS/people/anoopai/KneePipeline/logs'
output_file = os.path.join(log_path, f'pipleine_DESS_MDC_orgmesh.txt')
mean_data_path = os.path.join(code_dir_path, 'mean_data')
save_path = os.path.join(mean_data_path, 'spatial_autocorrelation/orgMesh')
mean_femur_path = os.path.join(mean_data_path, 'MeanFemur_mesh_B-only.vtk')
mdc_femur_path =os.path.join(save_path, f'MDC_Mean_orgMesh_femur_mesh_{knee}_{visit}.vtk')

if not os.path.exists(save_path):
    os.makedirs(save_path)

analysis_complete = []
data_all = []

for subject in os.listdir(data_path):
    subject_path = os.path.join(data_path, subject)

    if os.path.isdir(subject_path):
        # Iterate through each visit in the subject path
        for visit in visits:
            visit_path = os.path.join(subject_path, visit)

            if os.path.isdir(visit_path):
                # Iterate through each knee type in the visit path
                for knee in os.listdir(visit_path):
                    knee_path = os.path.join(visit_path, knee)
                    
                    if os.path.isdir(knee_path) and knee in knees_to_use:
                        path_image = os.path.join(knee_path, 'scans/qdess')
                        path_save = os.path.join(knee_path, f'results_nsm')
                        femur_path = os.path.join(path_save, 'femur_mesh_NSM_orig_reg2mean.vtk')
                        
                        sub_component= Path(knee_path).parts[6]  # '11-P'
                        visit_component = Path(knee_path).parts[7]  # 'VISIT-1'
                        knee_component = Path(knee_path).parts[8]  # 'clat'
                        try:
                        
                            if os.path.exists(femur_path):
                                femur_mesh = BoneMesh(femur_path)
                            
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

for scalar_name in scalar_names:
    print(f"Processing scalar: {scalar_name}")
    
    # Create a dictionary to store filtered results
    filtered_results = {knee: {visit: [] for visit in visits} for knee in knees_to_use}

    # Filter data
    for d in data_all:
        knee_type = d['knee']
        visit_type = d['visit']
        if knee_type in knees_to_use and visit_type in visits:
            filtered_results[knee_type][visit_type].append(d)

    # Access filtered results
    ctrl_results = filtered_results['ctrl']

    # Dictionary to hold results for each knee and visit
    results = {knee: {visit: {f'{scalar_name}': None} for visit in visits} for knee in knees_to_use}

    # Calculate MDC at each point on the mesh across all subjects
    for knee in knees_to_use:
        for visit in visits:
            # Gather all parameter2 and parameter1 data
            scalar_list = []
            
            for d in filtered_results[knee][visit]:
                scalar_list.append(d[scalar_name])
            
            # Convert to NumPy arrays
            scalar_array = np.array(scalar_list)  # Shape (n_subjects, 10000)

            # Initialize arrays to store results
            MDC = np.empty(scalar_array.shape[1])  # Shape (10000,)

            # Calculate Pearson's r, R^2, and p-value for each point
            for i in range(scalar_array.shape[1]):

                valid_thickness = scalar_array[:, i][(scalar_array[:, i] != -100) & (scalar_array[:, i] != -200)]
                total_sub= scalar_array[:, i].size
                valid_sub= valid_thickness.size
                percent_data = 100* (valid_sub/total_sub)
                                       
                # Calculate Pearson's r and p-value
                # Minimum detectable change = 1.96*SEM*sqrt(2)
                # SEM = SD/sqrt(2)                      
                if percent_data > 50:
                    MDC[i] = 1.96* (np.std(valid_thickness[:]))
                else:
                    MDC[i] = 0

            # Store results in the dictionary
            results[knee][visit] = {
                f'{scalar_name}': MDC,
            }

            if os.path.exists(mdc_femur_path):
                femur_mesh = BoneMesh(mdc_femur_path)
            else:
                femur_mesh = BoneMesh(mean_femur_path)

            femur_mesh.point_data[f'{scalar_name}'] = results[knee][visit][f'{scalar_name}']
            femur_mesh.save_mesh(mdc_femur_path)