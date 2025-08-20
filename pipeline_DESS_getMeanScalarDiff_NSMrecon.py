
from cProfile import label
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
knee_to_use = ['aclr', 'clat', 'ctrl']
dir_path = '/dataNAS/people/anoopai/DESS_ACL_study'
code_dir_path = '/dataNAS/people/anoopai/KneePipeline/'
data_path = os.path.join(dir_path, 'data')
log_path = '/dataNAS/people/anoopai/KneePipeline/logs'
output_file = os.path.join(log_path, f'pipeline_DESS_getMeanScalarDiff_NSMrecon.txt')
log_file_path = os.path.join(log_path, f'pipeline_DESS_getMeanScalarDiff_NSMrecon.txt')
sub_graft_path = os.path.join(dir_path, 'results/subjects_data/subject_data_graft_types.json')
mean_path = os.path.join(code_dir_path, 'mean_data')
save_path = os.path.join(mean_path, f'spatial_autocorrelation/NSMrecon')

if not os.path.exists(save_path):
    os.makedirs(save_path)
    
mean_femur_path = os.path.join(mean_path, 'MeanFemur_mesh_B-only.vtk')

parameter1= 'Diff_thickness (mm)'
parameter3= 'labels'

knees = ['aclr', 'clat', 'ctrl']
visits = ['VISIT-2', 'VISIT-3', 'VISIT-4', 'VISIT-5']
grafts= ['aclr_BPBT', 'aclr_other', 'clat', 'ctrl']

with open(sub_graft_path) as f:
        sub_grafts = json.load(f)
        
       
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
                        if knee == 'aclr':
                            for key, values in sub_grafts.items():
                                if subject in values :
                                    graft = key
                                    break
                        else:
                            graft = knee
                            
                        if os.path.isdir(knee_path) and knee in knee_to_use:
                            path_image = os.path.join(knee_path, 'scans/qdess')
                            path_save = os.path.join(knee_path, f'results_nsm')
                            femur_path = os.path.join(path_save, 'NSM_recon_femur_mesh_NSM_orig_reg2mean.vtk')
                            
                            sub_component= Path(knee_path).parts[6]  # '11-P'
                            visit_component = Path(knee_path).parts[7]  # 'VISIT-1'
                            knee_component = Path(knee_path).parts[8]  # 'clat'
                            graft_component = graft
                            try:
                                if os.path.exists(femur_path):
                                    femur_mesh = BoneMesh(femur_path)
                                    thickness_change_all = {}
                                    t2_change_all = {}
                                    scalar_names =[parameter1, parameter3]
                                    data= {'sub': sub_component,
                                            'visit': visit_component,
                                            'knee': knee_component,
                                            'graft': graft_component,
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


filtered_results = {knee: {visit: [] for visit in visits} for knee in knees}

# Filter data
for d in data_all:
    knee_type = d['knee']
    visit_type = d['visit']
    if knee_type in knees and visit_type in visits:
        filtered_results[knee_type][visit_type].append(d)

# Dictionary to hold results for each knee and visit
results = {knee: {visit: {f'{parameter1}': None, f'{parameter3}': None} for visit in visits} for knee in knees}

# Calculate Pearson's r, R^2, and p-value for each knee and visit
for knee in knees:
    for visit in visits:
        # Gather all parameter2 and parameter1 data
        parameter1_list = []
        parameter3_list= []
        
        for d in filtered_results[knee][visit]:
            parameter1_list.append(d[parameter1])
            parameter3_list.append(d[parameter3])
        
        # Convert to NumPy arrays
        parameter1_array = np.array(parameter1_list)  # Shape (n_subjects, 10000)
        parameter3_array = np.array(parameter3_list)  # Shape (n_subjects, 10000)

        # Initialize arrays to store results
        Thickness_mean_all = np.empty(parameter1_array.shape[1])
        labels = np.empty(parameter1_array.shape[1])

        # Calculate Pearson's r, R^2, and p-value for each point
        for i in range(parameter1_array.shape[1]):
            # Calculate the mean of the parameters (remove background/bone values from the mean)
            valid_thickness = parameter1_array[:, i][(parameter1_array[:, i] != -100) & (parameter1_array[:, i] != -200)]
            valid_labels = parameter3_array[:, i][(parameter3_array[:, i] != -100) & (parameter3_array[:, i] != -200)]
            
            total_sub= parameter1_array[:, i].size
            valid_sub= valid_thickness.size
            percent_data = 100* (valid_sub/total_sub)
                    
            if percent_data > 50:
                Thickness_mean_all[i] = valid_thickness.mean()
            else:
                Thickness_mean_all[i] = -200  # or 0 or whatever you prefer 
                
            total_sub= parameter3_array[:, i].size
            valid_sub= valid_labels.size
            percent_data = 100* (valid_sub/total_sub)
                
            if percent_data > 20:
                labels[i] = valid_labels.mean()
            else:
                labels[i] = -200
                
        # Store results in the dictionary
        results[knee][visit] = {
            # "spearman_r": r_values,
            # "p_value": p_values,
            f'{parameter1}': Thickness_mean_all,
            f'{parameter3}': labels,
        }
        
correlation_scalars= list(results[knee][visit].keys())

for knee in knees:
    for visit in visits:
        femur_mesh = BoneMesh(mean_femur_path)
        for scalar_name in correlation_scalars:
            # Add the scalar data to the femur mesh
            femur_mesh.point_data[scalar_name] = results[knee][visit][scalar_name]
        femur_mesh.save_mesh(os.path.join(save_path, f'Mean_NSMrecon_femur_mesh_{knee}_{visit}.vtk'))