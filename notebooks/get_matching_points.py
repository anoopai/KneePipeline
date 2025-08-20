# %%
import os
import shutil
import json

from pyparsing import C
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from statistics import mean
import torch
import json
import sys
import time
from NSM.models import TriplanarDecoder
from NSM.mesh import create_mesh
from pymskt.mesh import get_icp_transform, cpd_register, non_rigidly_register
from pymskt import mesh
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from pymskt.mesh.meshTools import smooth_scalars_from_second_mesh_onto_base, transfer_mesh_scalars_get_weighted_average_n_closest
from pymskt.mesh import Mesh, BoneMesh, CartilageMesh
from NSM.mesh.interpolate import interpolate_points

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

knee= 'aclr'
std_interval= 0.0
model_type= 'bone_only'

latent_file_path= f'/dataNAS/people/anoopai/DESS_ACL_study/results/BScore_and_FC_metrics/mean_shape_recon/change_overtime/BO/latents_change_{knee}.npz'
ref_mesh_path = '/dataNAS/people/anoopai/DESS_ACL_study/results/BScore_and_FC_metrics/mean_shape_recon/change_overtime/BO/change_overtime_aclr/aclr_0.0.vtk'
interp_mesh_path = f'/dataNAS/people/anoopai/DESS_ACL_study/results/BScore_and_FC_metrics/mean_shape_recon/change_overtime/BO/change_overtime_aclr/aclr_3.0_interp.vtk'

# GET MODEL PATHS AND N_OBJECTS
if model_type == 'bone_cartilage': 
    # Bone + Cartilage Model
    path_model_state = "/dataNAS/people/aagatti/projects/nsm_femur/training_run_files/experiment_results/647_nsm_femur_v0.0.1/model/2000.pth"
    path_model_config = "/dataNAS/people/aagatti/projects/nsm_femur/training_run_files/experiment_results/647_nsm_femur_v0.0.1/model_params_config.json"
    n_objects = 2

elif model_type == 'bone_only':
    # Bone only model - UNCOMMENT TO DO BONE ONLY MESH
    path_model_config= "/bmrNAS/people/aagatti/projects/auto_seg_server/Segmentation/NSM_MODELS/551_nsm_femur_bone_v0.0.1/model_params_config.json"
    path_model_state = "/bmrNAS/people/aagatti/projects/auto_seg_server/Segmentation/NSM_MODELS/551_nsm_femur_bone_v0.0.1/model/1150.pth"
    n_objects = 1
    
else:
    raise ValueError(f'Unknown model type: {model_type}')

with open(path_model_config, 'r') as f:
    model_config = json.load(f)
                     
print(f"Loading model")     
model = load_model(model_config, path_model_state, model_type='triplanar')

latent= np.load(latent_file_path)
ref_latent = latent[f'latent_change_0.0SD'].squeeze()
target_latent = latent[f'latent_change_3.0SD'].squeeze()
ref_mesh = BoneMesh(ref_mesh_path)
ref_mesh_points= ref_mesh.points

print("Interpolating points...")
interpolated_points = interpolate_points(
    model=model,
    latent1 = ref_latent,
    latent2=target_latent,
    n_steps=100,
    points1=ref_mesh_points,
    surface_idx=0,
    verbose=False,
    spherical=True
)

print("Creating and saving mesh from interpolated points...")
# create copies of the meshes, update the points, and save them to disk
interpolated_mesh = ref_mesh.copy()
interpolated_mesh.point_coords = interpolated_points
# interpolated_mesh.resample_surface(subdivisions=1, clusters=10000)
interpolated_mesh.save_mesh(interp_mesh_path)

