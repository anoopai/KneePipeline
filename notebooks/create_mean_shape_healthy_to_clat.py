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
import numpy as np

model_type = 'bone_cartilage' # 'bone_cartilage'
recon_type = 'ctrl_to_aclr'

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
data_path = os.path.join(dir_path, 'results/T2C_and_FC_thickness/ISB_2025_shaft_crop_B-only/mean_shape_recon')
log_path = '/dataNAS/people/anoopai/KneePipeline/logs'
output_file = os.path.join(log_path, f'recon_complete_{recon_type}.txt')
log_file_path = os.path.join(log_path, f'create_mean_shapes_healthy_to_aclr_errors.txt')
latent_file_path = os.path.join(data_path, f'latents_{recon_type}.npz')
save_path = os.path.join(data_path, f'{recon_type}_shapes')

if not os.path.exists(save_path):
    os.makedirs(save_path)
    
# GET MODEL PATHS AND N_OBJECTS
if model_type == 'bone_cartilage':
    # Bone + Cartilage Model
    path_model_state = "/bmrNAS/people/aagatti/projects/auto_seg_server/Segmentation/NSM_MODELS/231_nsm_femur_cartilage_v0.0.1/model/2000.pth"
    path_model_config = "/bmrNAS/people/aagatti/projects/auto_seg_server/Segmentation/NSM_MODELS/231_nsm_femur_cartilage_v0.0.1/model_config.json"
    n_objects = 2

elif model_type == 'bone_only':
    # Bone only model - UNCOMMENT TO DO BONE ONLY MESH
    path_model_state = "/bmrNAS/people/aagatti/projects/auto_seg_server/Segmentation/NSM_MODELS/551_nsm_femur_bone_v0.0.1/model/1150.pth"
    path_model_config = "/bmrNAS/people/aagatti/projects/auto_seg_server/Segmentation/NSM_MODELS/551_nsm_femur_bone_v0.0.1/model_params_config.json"
    n_objects = 1

analysis_info=[]
std_intervals = np.arange(0, 3.5, 0.5)

latent = np.load(latent_file_path)
files = latent.files

for file in files:
    print(f'Processing {file}')
    latent_vector = torch.tensor(latent[file].tolist()).cuda()

    if latent_vector is not None:
        start_time = time.time()
        
        # LOAD CONFIG
        with open(path_model_config, 'r') as f:
            model_config = json.load(f)

        # LOAD MODEL
        model = load_model(model_config, path_model_state, model_type='triplanar')

        # CREATE MESHES & SAVE THEM TO DISK
        if n_objects == 1:
            bone_mesh = create_mesh(decoder=model, latent_vector=latent_vector, objects=n_objects)
            bone_mesh.save_mesh(os.path.join(save_path, f'{file}_bone_mesh_B-only.vtk'))
        elif n_objects == 2:
            bone_mesh, cart_mesh = create_mesh(decoder=model, latent_vector=latent_vector, objects=n_objects)
            bone_mesh.save_mesh(os.path.join(save_path, f'{file}_bone_mesh_B+C.vtk'))
            cart_mesh.save_mesh(os.path.join(save_path, f'{file}_cart_mesh_B+C.vtk'))
            
            # Time tracking
        end_time = time.time()
        total_time = end_time - start_time
        minutes = int(total_time // 60)
        seconds = total_time % 60
        print(f'Total time taken: {minutes} minutes and {seconds:.2f} seconds')
        
    else:
        print(f'Latent vector not found for {file}')

