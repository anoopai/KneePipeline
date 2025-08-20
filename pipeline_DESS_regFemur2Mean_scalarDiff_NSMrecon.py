from math import log
from pathlib import Path
import os
import subprocess
import json
import sys
import time
import numpy as np
import shutil
from pymskt.mesh import get_icp_transform, cpd_register, non_rigidly_register
from pymskt import mesh
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from pymskt.mesh.meshTools import smooth_scalars_from_second_mesh_onto_base, transfer_mesh_scalars_get_weighted_average_n_closest
from pymskt.mesh import Mesh, BoneMesh


# Constants and configurations
knee_to_use = 'ctrl'
dir_path = '/dataNAS/people/anoopai/DESS_ACL_study'
# dir_path = '/dataNAS/people/anoopai/KneePipeline/'
data_path = os.path.join(dir_path, 'data_correction')
log_path = '/dataNAS/people/anoopai/KneePipeline/logs'
output_file = os.path.join(log_path, f'pipeline_DESS_regFemur2Mean_scalarDiff_NSMrecon_{knee_to_use}.txt')
log_file_path = os.path.join(log_path, f'pipeline_DESS_regFemur2Mean_scalarDiff_NSMrecon.txt')
mean_path = '/dataNAS/people/anoopai/KneePipeline/mean_data/MeanFemur_mesh_B-only.vtk'

if os.path.exists(output_file):
    os.remove(output_file)
    
if os.path.exists(log_file_path):
    os.remove(log_file_path)

# Clear the output file if it exists or create a new one
with open(output_file, 'w') as f:
    f.write('')

# List to keep track of the dictionaries
analysis_complete = []

# Log function for both console and log file
def log_message(message):
    print(message)
    with open(output_file, 'a') as logfile:
        logfile.write(message + '\n')

for subject in os.listdir(data_path):
    subject_path = os.path.join(data_path, subject)

    if os.path.isdir(subject_path):
        # Iterate through each visit in the subject path
        for visit in os.listdir(subject_path):
            if visit not in ['VISIT-6']:  # Exclude VISIT-6
                visit_path = os.path.join(subject_path, visit)
                visit1_path = os.path.join(subject_path, 'VISIT-1')

                if os.path.isdir(visit_path):
                    # Iterate through each knee type in the visit path
                    for knee in os.listdir(visit_path):
                        knee_path = os.path.join(visit_path, knee)
                        knee_path_V1= os.path.join(visit1_path, knee)

                        if os.path.isdir(knee_path) and knee == knee_to_use:
                            path_image = os.path.join(knee_path, 'scans/qdess')
                            path_save = os.path.join(knee_path, f'results_nsm')
                            path_save_v1 = os.path.join(knee_path_V1, f'results_nsm')
                            femur_path = os.path.join(path_save, 'NSM_recon_femur_mesh_NSM_orig.vtk')

                            # Split the path into components
                            sub_component = Path(knee_path).parts[6]  # '11-P'
                            visit_component = Path(knee_path).parts[7]  # 'VISIT-1'
                            knee_component = Path(knee_path).parts[8]  # 'clat'

                            # if os.path.exists(path_image) and not os.listdir(path_save):
                            if os.path.exists(path_image):
                                analysis_info = {}
                                analysis_info = {
                                    'sub': sub_component,
                                    'visit': visit_component,
                                    'knee': knee_component
                                }

                                print(f"Performing analysis on {analysis_info}")
                                
                                # Start tracking time
                                start_time = time.time()

                                ########################################################################################
                                # Register Mean Knee to each subject knee (to get point-to-point correspondance)
                                ########################################################################################     
                                mean_path_reg_org = os.path.join(path_save, 'femur_mesh_NSM_orig_reg2mean.vtk')
                                mean_path_reg = os.path.join(path_save, 'NSM_recon_femur_mesh_NSM_orig_reg2mean.vtk')
                                
                                try:
                                    if os.path.exists(mean_path_reg):
                                        os.remove(mean_path_reg)

                                    femur_mesh = BoneMesh(femur_path)
                                    mean_mesh = BoneMesh(mean_path)
                                    # mean_mesh.resample_surface(clusters=10000)

                                    print("Registering to mean Visit")
                                    mean_mesh_reg, _ = cpd_register(
                                        target_mesh= mean_mesh,
                                        source_mesh= femur_mesh,
                                        icp_register_first=True,
                                        icp_reg_target_to_source= True,
                                        transfer_scalars=True)
                                    
                                    mean_mesh_org= BoneMesh(mean_path_reg_org)
                                    labels = vtk_to_numpy(mean_mesh_org.GetPointData().GetArray('labels'))                                    

                                    mean_mesh_reg = BoneMesh(mean_mesh_reg)
                                    mean_mesh_reg.point_data['labels'] = labels
                                    mean_mesh_reg.save_mesh(mean_path_reg)                               

                                    # Time tracking
                                    end_time = time.time()
                                    total_time = end_time - start_time
                                    minutes = int(total_time // 60)
                                    seconds = total_time % 60
                                    print(f'Total time taken: {minutes} minutes and {seconds:.2f} seconds')
                                    
                                    analysis_info['time'] = f'{minutes} mins and {seconds:.2f} secs'
                                    
                                    with open(output_file, 'a') as f:
                                        f.write('\n Completed registration' + str(analysis_info))
                                        
                                except Exception as e:
                                    with open(log_file_path, 'a') as f:
                                        f.write(f"Error processing {analysis_info}: {e}\n")
                                    
# ########################################################################################
# Compute difference in scalar values w.r.t visit-1
########################################################################################
for subject in os.listdir(data_path):
    subject_path = os.path.join(data_path, subject)

    if os.path.isdir(subject_path):        
        
        for visit in os.listdir(subject_path):
            if visit not in ['VISIT-1', 'VISIT-6']:
                visit_path = os.path.join(subject_path, visit)
                visit1_path = os.path.join(subject_path, 'VISIT-1')

                if os.path.isdir(visit_path):
                    # Iterate through each knee type in the visit path
                    for knee in os.listdir(visit_path):
                        knee_path = os.path.join(visit_path, knee)
                        knee_path_V1= os.path.join(visit1_path, knee)

                        if os.path.isdir(knee_path) and knee == knee_to_use:
                            path_image = os.path.join(knee_path, 'scans/qdess')
                            path_save = os.path.join(knee_path, f'results_nsm')
                            path_save_v1 = os.path.join(knee_path_V1, f'results_nsm')
                            femur_path = os.path.join(path_save, 'NSM_recon_femur_mesh_NSM_orig_reg2mean.vtk')
                            femur_path_v1 = os.path.join(path_save_v1, 'NSM_recon_femur_mesh_NSM_orig_reg2mean.vtk')
                                
                            # Split the path into components
                            sub_component = Path(knee_path).parts[6]  # '11-P'
                            visit_component = Path(knee_path).parts[7]  # 'VISIT-1'
                            knee_component = Path(knee_path).parts[8]  # 'clat'

                            # if os.path.exists(path_image) and not os.listdir(path_save):
                            if os.path.exists(path_image):
                                analysis_info = {}
                                analysis_info = {
                                    'sub': sub_component,
                                    'visit': visit_component,
                                    'knee': knee_component
                                }

                            try:
                                print(f"Computing difference in scalar values w.r.t visit-1 for {analysis_info}")
                                femur_mesh = BoneMesh(femur_path)
                                femur_mesh_v1= BoneMesh(femur_path_v1)
                                
                                n_arrays = femur_mesh.GetPointData().GetNumberOfArrays()
                                scalar_names = [femur_mesh.GetPointData().GetArray(array_idx).GetName() for array_idx in range(n_arrays)]
                                
                                diff_scalar_names = ['thickness (mm)','labels']
                                
                                for scalar_name in scalar_names:
                                    if scalar_name in diff_scalar_names:
                                        femur_points_bone_only = vtk_to_numpy(femur_mesh.GetPointData().GetArray('labels'))
                                        femur_points_bone_only_v1 = vtk_to_numpy(femur_mesh_v1.GetPointData().GetArray('labels'))
                                        femur_points= vtk_to_numpy(femur_mesh.GetPointData().GetArray(scalar_name)) 
                                        femur_points_v1= vtk_to_numpy(femur_mesh_v1.GetPointData().GetArray(scalar_name))
                                        scalar_diff = np.zeros_like(femur_points)
                                        scalar_t2_pos = np.zeros_like(femur_points)
                                        scalar_diff_dict ={}
                                        
                                        for i, _ in enumerate(femur_points):

                                            if femur_points[i] == 0 and femur_points_v1[i] == 0:
                                                # print("Two zero values")
                                                scalar_diff[i] = -200.0
                                                scalar_t2_pos[i] = -200.0
                                            elif femur_points[i] == 0 or femur_points_v1[i] == 0:
                                                # print("One zero value")
                                                scalar_diff[i] = -100.0
                                                scalar_t2_pos[i] = -100.0
                                            else:   
                                                if scalar_name == 'labels':
                                                    scalar_diff[i] = int(femur_points[i])
                                                else:
                                                    scalar_diff[i] = femur_points[i] - femur_points_v1[i]
                                                                                                                                                                            
                                        scalar_diff_dict = scalar_diff
                                        
                                        if scalar_name == 'labels':
                                            femur_mesh.point_data[f'{scalar_name}'] = scalar_diff_dict
                                        else:
                                            femur_mesh.point_data[f'Diff_{scalar_name}'] = scalar_diff_dict
                                    
                                femur_mesh.save_mesh(femur_path)
                                                                
                            except Exception as e:
                                    with open(log_file_path, 'a') as f:
                                        f.write(f"Error processing {analysis_info}: {e}\n")
                                                                    