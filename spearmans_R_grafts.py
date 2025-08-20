
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
output_file = os.path.join(log_path, f't2_and_thickness_change.txt')
log_file_path = os.path.join(log_path, f'pipeline_DESS_errors.txt')
mean_path = os.path.join(code_dir_path, 'mean_data')
save_path = os.path.join(mean_path, f't2_and_thickness_change_graft')
sub_graft_path =  os.path.join(dir_path, 'results/subjects_data/subject_data_graft_types.json')

if not os.path.exists(save_path):
    os.makedirs(save_path)
    
mean_femur_path = os.path.join(mean_path, 'MeanFemur_mesh_B-only.vtk')

parameter1= 'Diff_thickness (mm)'
parameter2= 'Diff_T2_mean_filt'
parameter3= 'labels'
visits = ['VISIT-2', 'VISIT-3', 'VISIT-4', 'VISIT-5']
knees = ['aclr', 'clat', 'ctrl']
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
                            femur_path = os.path.join(path_save, 'femur_mesh_NSM_orig_reg2mean.vtk')
                            
                            sub_component= Path(knee_path).parts[6]  # '11-P'
                            visit_component = Path(knee_path).parts[7]  # 'VISIT-1'
                            knee_component = Path(knee_path).parts[8]  # 'clat'
                            graft_component = graft
                            try:
                                if os.path.exists(femur_path):
                                    femur_mesh = BoneMesh(femur_path)
                                    thickness_change_all = {}
                                    t2_change_all = {}
                                    scalar_names =[parameter1, parameter2, parameter3]
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

# Create a dictionary to store filtered results
# filtered_results = {knee: {visit: [] for visit in visits} for knee in knees}
filtered_results = {graft: {visit: [] for visit in visits} for graft in grafts}

# Filter data
for d in data_all:
    graft_type = d['graft']
    visit_type = d['visit']
    if graft_type in grafts and visit_type in visits:
        filtered_results[graft_type][visit_type].append(d)

# Access filtered results
aclr_BPBT_results = filtered_results['aclr_BPBT']
aclr_other_results = filtered_results['aclr_other']
clat_results = filtered_results['clat']
ctrl_results = filtered_results['ctrl']

# Dictionary to hold results for each knee and visit
results = {graft: {visit: {"spearman_r": None, "p_value": None} for visit in visits} for graft in grafts}

# Calculate Pearson's r, R^2, and p-value for each knee and visit
for graft in grafts:
    for visit in visits:
        # Gather all parameter2 and parameter1 data
        parameter2_list = []
        parameter1_list = []
        parameter_3_list= []
        
        for d in filtered_results[graft][visit]:
            parameter2_list.append(d[parameter2])
            parameter1_list.append(d[parameter1])
            parameter_3_list.append(d[parameter3])
        
        # Convert to NumPy arrays
        parameter2_array = np.array(parameter2_list)  # Shape (n_subjects, 10000)
        parameter1_array = np.array(parameter1_list)  # Shape (n_subjects, 10000)
        parameter3_array = np.array(parameter_3_list)  # Shape (n_subjects, 10000)

        # Initialize arrays to store results
        r_values = np.empty(parameter1_array.shape[1])  # Shape (10000,)
        p_values = np.empty(parameter1_array.shape[1])  # Shape (10000,)
        T2_mean_all = np.empty(parameter1_array.shape[1])
        Thickness_mean_all = np.empty(parameter1_array.shape[1])
        labels = np.empty(parameter1_array.shape[1])
        direction = np.empty(parameter1_array.shape[1])

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
                T2_mean_all[i] = -200
                Thickness_mean_all[i] = -200
                labels[i]=-200
                direction[i] = -200
            
            elif fc_edges_ratio_param1 > 0.1 or fc_edges_ratio_param2 > 0.1:
                r_values[i] = -100.0  # Assign 0.0 for r-values
                p_values[i] = 1.0  # Assign 1.0 for p-values
                T2_mean_all[i] = -100
                Thickness_mean_all[i] = -100
                labels[i]=-200
                direction[i] = -100
            else:
                # Calculate Pearson's r and p-value
                r_values[i], p_values[i] = spearmanr(parameter1_array[:, i], parameter2_array[:, i])

                # Calculate the mean of the parameters (remove background/bone values from the mean)
                valid_thickness = parameter1_array[:, i][(parameter1_array[:, i] != -100) & (parameter1_array[:, i] != -200)]
                valid_t2 = parameter2_array[:, i][(parameter2_array[:, i] != -100) & (parameter2_array[:, i] != -200)]
                valid_labels = parameter3_array[:, i][(parameter3_array[:, i] != -100) & (parameter3_array[:, i] != -200)]
                
                # Calculate the mean excluding -100 and -200
                if valid_thickness.size > 0:
                    Thickness_mean_all[i] = valid_thickness.mean()
                else:
                    Thickness_mean_all[i] = -200  # or 0 or whatever you prefer

                if valid_t2.size > 0:
                    T2_mean_all[i] = valid_t2.mean()
                else:
                    T2_mean_all[i] = -200
                    
                if valid_labels.size > 0:
                    labels[i] = valid_labels.mean()
                else:
                    labels[i] = -200
                
                if Thickness_mean_all[i] > 0 and T2_mean_all[i] > 0:
                    direction[i] = 1
                elif Thickness_mean_all[i] < 0 and T2_mean_all[i] > 0:
                    direction[i] = 2
                elif Thickness_mean_all[i] < 0 and T2_mean_all[i] < 0:
                    direction[i] = 3
                elif Thickness_mean_all[i] > 0 and T2_mean_all[i] < 0:
                    direction[i] = 4
                

        # Store results in the dictionary
        results[graft][visit] = {
            "spearman_r": r_values,
            "p_value": p_values,
            "Diff_T2_mean_filt_all": T2_mean_all,
            "Diff_thickness_all": Thickness_mean_all,
            "labels": labels,
            "direction": direction
        }
        
correlation_scalars= list(results[graft][visit].keys())

for graft in grafts:
    for visit in visits:
        femur_mesh = BoneMesh(mean_femur_path)
        for scalar_name in correlation_scalars:
            # print(knee, visit, scalar_name)
            # Add the scalar data to the femur mesh
            femur_mesh.point_data[scalar_name] = results[graft][visit][scalar_name]
        femur_mesh.save_mesh(os.path.join(save_path, f'MeanFemur_mesh_B-only_{graft}_{visit}.vtk'))