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
from utils.project_t2_thickness_on_FC import project_t2_thickness_on_FC

# Constants and configurations
knee_to_use = 'aclr'
# dir_path = '/dataNAS/people/anoopai/DESS_ACL_study'
dir_path = '/dataNAS/people/anoopai/KneePipeline/'
data_path = os.path.join(dir_path, 'data_trial')
log_path = '/dataNAS/people/anoopai/KneePipeline/logs'
output_file = os.path.join(log_path, f'femur_reg_{knee_to_use}.txt')
log_file_path = os.path.join(log_path, f'pipeline_DESS_errors.txt')

if os.path.exists(output_file):
    os.remove(output_file)
    
# if os.path.exists(log_file_path):
#     os.remove(log_file_path)


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

                if os.path.isdir(visit_path):
                    # Iterate through each knee type in the visit path
                    for knee in os.listdir(visit_path):
                        knee_path = os.path.join(visit_path, knee)

                        if os.path.isdir(knee_path) and knee == knee_to_use:
                            
                            path_image = os.path.join(knee_path, 'scans/qdess')
                            path_save = os.path.join(knee_path, f'results_nsm')
                            seg_path = os.path.join(path_save, f'qdess_subregions-labels.nrrd')
                            
                            project_t2_thickness_on_FC(map_path, seg_path)
                            source_path = os.path.join(path_save, 'femur_cart_mesh_0.vtk')
                            source_path = os.path.join(path_save, 'femur_cart_mesh_0_scalars.vtk')

                            
                            source_norm_path = os.path.join(path_save, 'femur_mesh_NSM_orig_norm.vtk')
                            target_path = '/dataNAS/people/anoopai/KneePipeline/MeanBone_mesh_B-only.vtk'
                            source_save_path_reg = os.path.join(path_save, 'femur_mesh_NSM_reg2MeanBone.vtk')
                            target_save_path_reg = os.path.join(path_save, 'MeanBone_mesh_B-only_reg2femur.vtk')
                        
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

                                # try: 
                                
                                orig_mesh = BoneMesh(source_path)
                                mean_mesh= BoneMesh(target_path)
                                mean_mesh.resample_surface(clusters=10000)
                                # orig_mesh.resample_surface(clusters=10000)
                                
                                points = orig_mesh.point_coords
                                center = np.mean(points, axis=0)
                                centered_points = points - center
                                max_rad_dist = np.max(np.linalg.norm(centered_points, axis=1))
                                centered_points /= max_rad_dist
                                orig_mesh.point_coords = centered_points
                                orig_mesh.save_mesh(source_norm_path)
                                
                                orig_mesh_norm = BoneMesh(source_norm_path)
                                
                                print("Registering")
                                mean_mesh_reg, _ = cpd_register(
                                    target_mesh= orig_mesh_norm,
                                    source_mesh= mean_mesh,
                                    transfer_scalars=True
                                )
                                
                                orig_mesh_reg, _ = cpd_register(
                                    target_mesh= mean_mesh,
                                    source_mesh= orig_mesh_norm,
                                    transfer_scalars=True
                                )
                                                               
                                orig_mesh_reg = BoneMesh(orig_mesh_reg)
                                orig_mesh_reg.save_mesh(source_save_path_reg)

                                # mean_mesh_reg = BoneMesh(mean_mesh_reg)
                                
                                print("Transfering scalars")
                                mean_mesh_reg_scalars = transfer_mesh_scalars_get_weighted_average_n_closest(
                                    new_mesh= mean_mesh_reg,
                                    old_mesh= orig_mesh_norm,
                                    return_mesh=True,
                                )
                                
                                mean_mesh_reg_scalars = BoneMesh(mean_mesh_reg_scalars)
                                # # mean_mesh_reg.resample_surface(clusters=10000)
                                mean_mesh_reg_scalars.save_mesh(target_save_path_reg)

                                # Time tracking
                                end_time = time.time()
                                total_time = end_time - start_time
                                minutes = int(total_time // 60)
                                seconds = total_time % 60
                                print(f'Total time taken: {minutes} minutes and {seconds:.2f} seconds')
                                
                                analysis_info['time'] = f'{minutes} mins and {seconds:.2f} secs'
                                
                                with open(output_file, 'a') as f:
                                    f.write('\n Completed' + str(analysis_info))
                                    
                                # except Exception as e:
                                #     with open(log_file_path, 'a') as f:
                                #         f.write(f"Error processing {analysis_info}: {e}\n")