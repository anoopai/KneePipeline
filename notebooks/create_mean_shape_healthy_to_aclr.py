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

from NSM.mesh import create_mesh
from pymskt.mesh.meshTools import get_cartilage_properties_at_points
from pymskt import mesh
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from pymskt.mesh.meshTools import get_cartilage_properties_at_points
from pymskt.mesh import Mesh, BoneMesh, CartilageMesh
from NSM.mesh.interpolate import interpolate_points

model_type = 'bone_only' # 'bone_cartilage'
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

def project_fc_thickness_on_femur_mesh(bone_mesh_path, cart_mesh_path):

    bone_mesh= Mesh(bone_mesh_path)
    cart_mesh = Mesh(cart_mesh_path)

    # IMPORTANT: The meshes should be "unnormailised" o patient size. Rough factor for scale-up is 45
    bone_mesh.point_coords = bone_mesh.point_coords * 45
    cart_mesh.point_coords = cart_mesh.point_coords * 45

    # setup the probe that we are using to get data from the T2 file 
    line_resolution = 500   # number of points along the line that the T2 data is sampled at
    no_thickness_filler = 0.0              # if no data is found, what value to fill the data with
    ray_cast_length= 20          # how far to extend the ray from the surface (using negative to go inwards/towards the other side)
    percent_ray_length_opposite_direction = 0.25  # extend the other way a % of the line to make sure get both edges. 1.0 = 100%|
    n_intersections = 2  # how many intersections to expect. If 2, then we expect to find two points on the ray that intersect the cartilage surface.

    thicknesses = np.zeros(bone_mesh.GetNumberOfPoints())

    node_data = get_cartilage_properties_at_points(
        bone_mesh,
        cart_mesh,
        t2_vtk_image=None,
        seg_vtk_image=None,
        ray_cast_length=ray_cast_length,
        percent_ray_length_opposite_direction=percent_ray_length_opposite_direction,
    )
    thicknesses += node_data

    # Assign the thickness scalars to the bone mesh surface.
    bone_mesh.point_data["thickness (mm)"] = thicknesses
    bone_mesh.save_mesh(bone_mesh_path)
    
# Setup paths
dir_path = '/dataNAS/people/anoopai/DESS_ACL_study'
data_path = os.path.join(dir_path, 'results/BScore_and_FC_metrics/mean_shape_recon')
log_path = '/dataNAS/people/anoopai/KneePipeline/logs'
output_file = os.path.join(log_path, f'recon_complete_{recon_type}.txt')
log_file_path = os.path.join(log_path, f'create_mean_shapes_healthy_to_aclr_errors.txt')
latent_file_path = os.path.join(data_path, f'latents_{recon_type}.npz')
path_config= '/dataNAS/people/anoopai/KneePipeline/config.json'


if not os.path.exists(save_path):
    os.makedirs(save_path)
    
with open(path_config) as f:
    config = json.load(f)
    
if model_type == 'bone_cartilage':

    # Setup file paths - relative to this script
    path_model_config = config['nsm']['path_model_config']
    path_model_state = config['nsm']['path_model_state']

    # Get BScore model path information
    path_model_folder = config['bscore']['path_model_folder'] 
    
    n_objects = 2

elif model_type == 'bone_only':
    # Setup file paths - relative to this script
    path_model_config = config['nsm_bone_only']['path_model_config']
    path_model_state = config['nsm_bone_only']['path_model_state']

    # Get BScore model path information
    path_model_folder = config['bscore_bone_only']['path_model_folder'] 
    
    n_objects = 1
else:
    raise ValueError(f'Unknown model type: {model_type}')

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
            ref_femur_mesh_path = os.path.join(save_path, f'{file}_bone_mesh_B+C.vtk')
            ref_cart_mesh_path = os.path.join(save_path, f'{file}_cart_mesh_B+C.vtk')
            bone_mesh.save_mesh(ref_femur_mesh_path)
            cart_mesh.save_mesh(ref_cart_mesh_path)
            project_fc_thickness_on_femur_mesh(ref_femur_mesh_path, ref_cart_mesh_path)
               
        # Time tracking
        end_time = time.time()
        total_time = end_time - start_time
        minutes = int(total_time // 60)
        seconds = total_time % 60
        print(f'Total time taken: {minutes} minutes and {seconds:.2f} seconds')
        
    else:
        print(f'Latent vector not found for {file}')

