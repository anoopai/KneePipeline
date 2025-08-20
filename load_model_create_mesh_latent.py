from NSM.models import TriplanarDecoder
import torch
from NSM.mesh import create_mesh
import json
import os
from pathlib import Path
import pandas as pd
import sys
import time
import shutil


# model_type = 'bone_only' # 'bone_cartilage'
model_type = 'bone_cartilage'
knees_to_use = ['aclr', 'ctrl', 'clat']

def load_model(config, path_model_state, model_type='triplanar'):

    if model_type == 'triplanar':
        model_class = TriplanarDecoder
        params = {
            'latent_dim': config['latent_size'],
            'n_objects': config['objects_per_decoder'],
            'conv_hidden_dims': config['conv_hidden_dims'],
            'conv_deep_image_size': config['conv_deep_image_size'],
            'conv_norm': config['conv_norm'], 
            'conv_norm_type': config['conv_norm_type'],
            'conv_start_with_mlp': config['conv_start_with_mlp'],
            'sdf_latent_size': config['sdf_latent_size'],
            'sdf_hidden_dims': config['sdf_hidden_dims'],
            'sdf_weight_norm': config['weight_norm'],
            'sdf_final_activation': config['final_activation'],
            'sdf_activation': config['activation'],
            'sdf_dropout_prob': config['dropout_prob'],
            'sum_sdf_features': config['sum_conv_output_features'],
            'conv_pred_sdf': config['conv_pred_sdf'],
        }
    elif model_type == 'deepsdf':
        model_class = Decoder
        params = {
            'latent_size': config['latent_size'],
            'dims': config['layer_dimensions'],
            'dropout': config['layers_with_dropout'],
            'dropout_prob': config['dropout_prob'],
            'norm_layers': config['layers_with_norm'],
            'latent_in': config['layer_latent_in'],
            'weight_norm': config['weight_norm'],
            'xyz_in_all': config['xyz_in_all'],
            'latent_dropout': config['latent_dropout'],
            'activation': config['activation'],
            'final_activation': config['final_activation'],
            'concat_latent_input': config['concat_latent_input'],
            'n_objects': config['objects_per_decoder'],
            'progressive_add_depth': config['progressive_add_depth'],
            'layer_split': config['layer_split'],
        }
    else:
        raise ValueError(f'Unknown model type: {model_type}')


    model = model_class(**params)
    saved_model_state = torch.load(path_model_state)
    model.load_state_dict(saved_model_state["model"])
    model = model.cuda()
    model.eval()
    return model


dir_path = '/dataNAS/people/anoopai/DESS_ACL_study'
data_path = os.path.join(dir_path, 'data')
log_path = '/dataNAS/people/anoopai/KneePipeline/logs'
output_file = os.path.join(log_path, f'recon_complete_B+C.txt')
log_file_path = os.path.join(log_path, f'pipeline_DESS_errors.txt')

# List to keep track of the dictionaries
analysis_complete = []

qdess_df_all= pd.DataFrame()
nsm_df_all= pd.DataFrame()

for subject in os.listdir(data_path):
    if subject == '42-C':
        continue
    subject_path = os.path.join(data_path, subject)
    
    if os.path.isdir(subject_path):
        # Iterate through each visit in the subject path
        for visit in os.listdir(subject_path):
            # Exclude VISIT-1 and VISIT-6
            if visit not in ['VISIT-6']:
                visit_path = os.path.join(subject_path, visit)
                
                if os.path.isdir(visit_path):
                    # Iterate through each knee type in the visit path
                    for knee in os.listdir(visit_path):
                        knee_path = os.path.join(visit_path, knee)
                        
                        if os.path.isdir(knee_path) and knee in knees_to_use:
            
                            path_image =  os.path.join(knee_path,  'scans/qdess')
                            results_path =  os.path.join(knee_path, 'results_nsm')
                            qdess_results_file = os.path.join(results_path,  'qdess_results.json')
                            nsm_recon_file = os.path.join(results_path, 'NSM_recon_params.json')
                            nsm_bone_only_recon_file = os.path.join(results_path, 'NSM_bone_only_recon_params.json')
                            folder_save = results_path
                            
                            # GET MODEL PATHS AND N_OBJECTS
                            if model_type == 'bone_cartilage': 
                                # Bone + Cartilage Model
                                path_model_state = "/dataNAS/people/aagatti/projects/nsm_femur/training_run_files/experiment_results/647_nsm_femur_v0.0.1/model/2000.pth"
                                path_model_config = "/dataNAS/people/aagatti/projects/nsm_femur/training_run_files/experiment_results/647_nsm_femur_v0.0.1/model_params_config.json",
                                n_objects = 2
                                if os.path.exists(nsm_recon_file):
                                    with open(nsm_recon_file) as f:
                                        nsm_recon = json.load(f)
                                        latent_vector = torch.tensor(nsm_recon['latent']).cuda()
                            
                            elif model_type == 'bone_only':
                                # Bone only model - UNCOMMENT TO DO BONE ONLY MESH
                                path_model_config= "/bmrNAS/people/aagatti/projects/auto_seg_server/Segmentation/NSM_MODELS/551_nsm_femur_bone_v0.0.1/model_params_config.json",
                                path_model_state = "/bmrNAS/people/aagatti/projects/auto_seg_server/Segmentation/NSM_MODELS/551_nsm_femur_bone_v0.0.1/model/1150.pth"
                                n_objects = 1
                                if os.path.exists(nsm_bone_only_recon_file):
                                    with open(nsm_bone_only_recon_file) as f:
                                        nsm_bone_only_recon = json.load(f) 
                                        latent_vector = torch.tensor(nsm_bone_only_recon['latent']).cuda()
                            
                            if latent_vector is not None:
                                # Split the path into components
                                sub_component = Path(results_path).parts[6]  # '11-P'
                                visit_component = Path(results_path).parts[7]  # 'VISIT-1'
                                knee_component = Path(results_path).parts[8]  # 'clat'
                                
                                analysis_info = {}
                                analysis_info = {
                                    'sub': sub_component,
                                    'visit': visit_component,
                                    'knee': knee_component
                                }
                                print(f'Processing: {sub_component} - {visit_component} - {knee_component}')
                                
                                start_time = time.time()
                                
                                # LOAD CONFIG
                                with open(path_model_config, 'r') as f:
                                    model_config = json.load(f)

                                # LOAD MODEL
                                model = load_model(model_config, path_model_state, model_type='triplanar')

                                # CREATE MESHES & SAVE THEM TO DISK
                                if n_objects == 1:
                                    bone_mesh = create_mesh(decoder=model, latent_vector=latent_vector, objects=n_objects)
                                    bone_mesh.save_mesh(os.path.join(folder_save, 'bone_mesh_B-only.vtk'))
                                elif n_objects == 2:
                                    bone_mesh, cart_mesh = create_mesh(decoder=model, latent_vector=latent_vector, objects=n_objects)
                                    bone_mesh.save_mesh(os.path.join(folder_save, 'bone_mesh_B+C.vtk'))
                                    cart_mesh.save_mesh(os.path.join(folder_save, 'cart_mesh_B+C.vtk'))
                                    
                                                                        # Time tracking
                                end_time = time.time()
                                total_time = end_time - start_time
                                minutes = int(total_time // 60)
                                seconds = total_time % 60
                                print(f'Total time taken: {minutes} minutes and {seconds:.2f} seconds')
                                
                                analysis_info['time'] = f'{minutes} mins and {seconds:.2f} secs'
                                
                                with open(output_file, 'a') as f:
                                    f.write('\n Completed' + str(analysis_info))
                            
                            else:
                                print(f'Latent vector not found for {sub_component} - {visit_component} - {knee_component} ')

