import os
import sys
import time
import shutil
import json
import traceback
from weakref import ref
import numpy as np
import pandas as pd
from pathlib import Path
from statistics import mean
import torch
import pyvista as pv
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from NSM.models import TriplanarDecoder
from NSM.mesh import create_mesh
from pymskt.mesh.meshTools import get_cartilage_properties_at_points 
from pymskt.mesh import Mesh, BoneMesh, CartilageMesh
from NSM.mesh.interpolate import interpolate_points, interpolate_mesh, interpolate_common
from pymskt import mesh


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

def create_mean_mesh(latent_vector, model, model_type):

    latent_vector = torch.from_numpy(latent_vector).float().cuda()

    # CREATE MESHES & SAVE THEM TO DISK
    if model_type == 'bone_only':
        bone_mesh = create_mesh(decoder=model, latent_vector=latent_vector, objects=n_objects)
        bone_mesh.resample_surface(clusters=10000)
        bone_mesh.smooth
        return bone_mesh
        
    elif model_type == 'bone_cartilage':
        bone_mesh, cart_mesh = create_mesh(decoder=model, latent_vector=latent_vector, objects=n_objects)
        bone_mesh.smooth
        cart_mesh.smooth
        n_sample = 10000
        bone_mesh.resample_surface(clusters=n_sample)
        cart_mesh.resample_surface(clusters=n_sample)
        return bone_mesh, cart_mesh  
    else:
        raise ValueError(f'Input NSM model type')
     
def interpolate_matching_points(ref_latent, ref_mesh_path_femur, target_latent, model, model_type):
    if model_type == 'bone_only':
        ref_mesh = BoneMesh(ref_mesh_path_femur[0])
        # ref_mesh_points= ref_mesh.points
        # mesh_points_interp = interpolate_mesh(
        #     model= model,
        #     latent1= ref_latent,
        #     latent2= target_latent,
        #     n_steps= 100,
        #     points1= ref_mesh_points,
        #     surface_idx= 0,
        #     verbose= False,
        #     spherical= False
        # )
        
        interpolated_mesh= interpolate_mesh(
            model= model,
            latent1= ref_latent,
            latent2= target_latent,
            n_steps= 100,
            mesh= ref_mesh, 
            surface_idx= 0, 
            verbose= False, 
            spherical= True, 
            max_edge_len= 0.04, 
            adaptive= True, 
            smooth= True, 
            smooth_type= 'laplacian')
        
        interpolated_mesh.resample_surface(clusters=10000)
        
        # create copies of the meshes, update the points, and save them to disk
        # interpolated_mesh = ref_mesh.copy()
        # interpolated_mesh.point_coords = mesh_points_interp
        return interpolated_mesh
        
    elif model_type == 'bone_cartilage':
        ref_mesh = BoneMesh(ref_mesh_path_femur[0])
        # ref_mesh_points= ref_mesh.points
        # mesh_points_interp_femur= interpolate_points(
        #     model=model,
        #     latent1= ref_latent,
        #     latent2= target_latent,
        #     n_steps= 100,
        #     points1= ref_mesh_points,
        #     surface_idx= 0,
        #     verbose= False,
        #     spherical= False
        # )
        
        interpolated_mesh_femur= interpolate_mesh(
            model= model,
            latent1= ref_latent,
            latent2= target_latent,
            n_steps= 100,
            mesh= ref_mesh, 
            surface_idx= 0, 
            verbose= False, 
            spherical= True, 
            max_edge_len= 0.04, 
            adaptive= True, 
            smooth= True, 
            smooth_type= 'laplacian')
        
        interpolated_mesh_femur.resample_surface(clusters=10000)
        
        interpolated_mesh_femur.copy_scalars_from_other_mesh_to_current(ref_mesh)
        
        # # create copies of the meshes, update the points, and save them to disk
        # interpolated_mesh_femur= ref_mesh.copy()
        # interpolated_mesh_femur.point_coords = mesh_points_interp_femur
        
        ref_mesh_cart = CartilageMesh(ref_mesh_path_femur[1])
        # ref_mesh_points_cart= ref_mesh_cart.points
        # mesh_points_interp_cart= interpolate_points(
        #     model=model,
        #     latent1= ref_latent,
        #     latent2= target_latent,
        #     n_steps= 100,
        #     points1= ref_mesh_points_cart,
        #     surface_idx= 1,
        #     verbose= False,
        #     spherical= False
        # )
        
        interpolated_mesh_cart= interpolate_mesh(
            model= model,
            latent1= ref_latent,
            latent2= target_latent,
            n_steps= 100,
            mesh= ref_mesh_cart, 
            surface_idx= 1, 
            verbose= False, 
            spherical= True, 
            max_edge_len= 0.04, 
            adaptive= True, 
            smooth= True, 
            smooth_type= 'laplacian')
        
        interpolated_mesh_cart.resample_surface(clusters=10000)        
        # # create copies of the meshes, update the points, and save them to disk
        # interpolated_mesh_cart= ref_mesh_cart.copy()
        # interpolated_mesh_cart.point_coords = mesh_points_interp_cart
        
        return interpolated_mesh_femur, interpolated_mesh_cart
    else:
        raise ValueError(f'Unknown model type: {model_type}')

def compute_signed_distance_and_normal(mesh1_path, mesh2_path):
    """
    Compute signed distance from each point in mesh2 to the surface of mesh1.
    Signed distance is Positive outside, negative inside, zero on surface.
    Compute normal vector from mesh1 to mesh2.
    """
    
    mesh1 = pv.read(mesh1_path)
    mesh2 = pv.read(mesh2_path)
    
    # Ensure matching number of points after resampling
    assert mesh1.n_points == mesh2.n_points, "Resampling failed to produce matching number of points"

    # Build implicit distance function from surface of mesh1
    implicit_distance = vtk.vtkImplicitPolyDataDistance()
    surf = mesh1.extract_surface().triangulate().clean()
    implicit_distance.SetInput(surf)  # For older VTK versions

    # Evaluate signed distance at each point in mesh2
    signed_distances = np.array([
        implicit_distance.EvaluateFunction(point) for point in mesh2.points
    ])

    # Store result as point scalar
    mesh2.point_data["signed_distance"] = signed_distances
    
    # Compute displacement vectors
    disp_vectors = mesh2.points - mesh1.points  # shape (N, 3)
    normx = disp_vectors[:, 0]
    normy = disp_vectors[:, 1]
    normz = disp_vectors[:, 2]
    magnitude = np.linalg.norm(disp_vectors, axis=1)

    # Attach as scalars to mesh2
    mesh2.point_data['normx'] = normx
    mesh2.point_data['normy'] = normy
    mesh2.point_data['normz'] = normz
    
    displacement_vectors = np.column_stack([
    mesh2.point_data['normx'],
    mesh2.point_data['normy'],
    mesh2.point_data['normz']
    ])
    mesh2.point_data['displacement'] = displacement_vectors  # single vector field
    mesh = BoneMesh(mesh2)
    return mesh 

def get_icp_transform(model_path, bscore):
    
    '''
    This function generates mean mesh for a given Bscore along the Bscore vector from Mean healthy --> Mean OA
    '''
    
    import numpy as np
    import os
    import json

    path_json = os.path.join(model_path, 'model.json')

    with open(path_json, 'r') as f:
        model = json.load(f)
        
        bscore_vector = np.array(model['bscore_vector'])
        mean_healthy = np.array(model['mean_healthy'])
        std_healthy = np.array(model['std_healthy'])
        
        projection = (bscore * mean_healthy ) / std_healthy

        latent_vector = projection * bscore_vector
        
        return latent_vector

def interpolate_latent_for_bscore(model_path, bscore):
    """
    Interpolate latent vector for a given Bscore.
    """
    path_json = os.path.join(model_path, 'model.json')

    with open(path_json, 'r') as f:
        model = json.load(f)
        
        bscore_vector = np.array(model['bscore_vector'])
        mean_healthy = np.array(model['mean_healthy'])
        std_healthy = np.array(model['std_healthy'])
        
        # Compute norm squared of the bscore_vector
        bscore_vector_norm = np.linalg.norm(bscore_vector)

        # Your existing code
        projection = bscore * std_healthy + mean_healthy
        latent_vector = (projection / bscore_vector_norm) * bscore_vector
        
        # latent_vector = projection * bscore_vector
        
        return latent_vector
    
def get_bscore(model_path, latent):
    """
    Interpolate latent vector for a given Bscore.
    """
    path_json = os.path.join(model_path, 'model.json')

    with open(path_json, 'r') as f:
        model = json.load(f)
        
    bscore_vector = np.array(model['bscore_vector'])
    mean_healthy = np.array(model['mean_healthy'])
    std_healthy = np.array(model['std_healthy'])
    
    # project data onto coeffs
    projection = latent @ bscore_vector.T
    
    # standardize data
    bscore = (projection - mean_healthy) / std_healthy
    
    return bscore

def project_fc_thickness_on_femur_mesh(bone_mesh_path, cart_mesh_path):

    bone_mesh= BoneMesh(bone_mesh_path)
    cart_mesh = CartilageMesh(cart_mesh_path)

    # IMPORTANT: The meshes should be "unnormailised" o patient size. Rough factor for scale-up is 45
    bone_mesh.point_coords = bone_mesh.point_coords * 45
    cart_mesh.point_coords = cart_mesh.point_coords * 45

    # setup the probe that we are using to get data from the T2 file 
    line_resolution = 100   # number of points along the line that the T2 data is sampled at
    no_thickness_filler = 0.0  # if no data is found, what value to fill the data with
    ray_cast_length= 10          # how far to extend the ray from the surface (using negative to go inwards/towards the other side)
    percent_ray_length_opposite_direction = 0.05  # extend the other way a % of the line to make sure get both edges. 1.0 = 100%|
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
    bone_mesh.point_coords = bone_mesh.point_coords / 45
    cart_mesh.point_coords = cart_mesh.point_coords / 45
    bone_mesh.save_mesh(bone_mesh_path)
    cart_mesh.save_mesh(cart_mesh_path)  
    
def calculate_fc_thickess_changes(bone_mesh_path_ref, bone_mesh_path_target):
    
    """
    Calculate the difference in cartilage thickness between two bone meshes.
    This function copies the cartilage thickness scalars from the reference mesh to the target mesh,
    calculates the difference in thickness, and saves the modified target mesh with the new scalars.
    Args:
        bone_mesh_path_ref (str): Path to the reference bone mesh file.
        bone_mesh_path_target (str): Path to the target bone mesh file.
    Returns:
        None: The function modifies the target mesh in place and saves it with the new scalars.
    
    """
    # Load the meshes
    bone_mesh1 = BoneMesh(bone_mesh_path_ref) 
    bone_mesh2 = BoneMesh(bone_mesh_path_target)
    
    bone_mesh2.copy_scalars_from_other_mesh_to_current(
        other_mesh=bone_mesh1,  # The mesh to copy scalars from
        orig_scalars_name='thickness (mm)',
        new_scalars_name= 'thickness (mm) Ref'
    )
    
    thickness_ref = vtk_to_numpy(bone_mesh2.GetPointData().GetArray('thickness (mm) Ref'))
    thickness = vtk_to_numpy(bone_mesh2.GetPointData().GetArray('thickness (mm)'))
    
    # Add edge detection - points with very low thickness are likely edge points
    edge_threshold = 0.25  # mm - adjust this value based on your data
    potential_edges = (thickness < edge_threshold) | (thickness_ref < edge_threshold)
    
    # Calculate the difference in thickness
    thickness_diff = thickness - thickness_ref
    
    # Set the difference to zero for edge points and where either thickness is zero
    thickness_diff[(thickness_ref == 0) | (thickness == 0) | potential_edges] = 0
    # thickness_diff[(thickness_ref == 0) | (thickness == 0)] = 0
    
    # Optional: Apply smoothing near edges to reduce artifacts
    # Find points near edges (within 2 points connectivity)
    near_edges = np.zeros_like(potential_edges)
    mesh_points = bone_mesh2.GetPoints()
    locator = vtk.vtkPointLocator()
    locator.SetDataSet(bone_mesh2)
    locator.BuildLocator()
    
    for i in range(len(potential_edges)):
        if potential_edges[i]:
            point = mesh_points.GetPoint(i)
            # Find points within a small radius
            id_list = vtk.vtkIdList()
            locator.FindPointsWithinRadius(0.5, point, id_list)  # 0.5mm radius
            for j in range(id_list.GetNumberOfIds()):
                near_edges[id_list.GetId(j)] = 1
    
    # Smooth the transition near edges
    for i in range(len(thickness_diff)):
        if near_edges[i] and not potential_edges[i]:
            # Find neighboring points
            id_list = vtk.vtkIdList()
            locator.FindClosestNPoints(5, mesh_points.GetPoint(i), id_list)
            # Calculate weighted average excluding edge points
            valid_diffs = []
            for j in range(id_list.GetNumberOfIds()):
                idx = id_list.GetId(j)
                if not potential_edges[idx]:
                    valid_diffs.append(thickness_diff[idx])
            if valid_diffs:
                thickness_diff[i] = np.mean(valid_diffs)
    
    # Add the difference scalar to the mesh
    bone_mesh2['thickness_diff (mm)'] = thickness_diff
    
    # Save the modified mesh
    bone_mesh2.save_mesh(bone_mesh_path_target)

# Gather all the latent vectors
dir_path= '/dataNAS/people/anoopai/DESS_ACL_study'
data_path= os.path.join(dir_path, 'data')
log_path = '/dataNAS/people/anoopai/KneePipeline/logs'
output_file = os.path.join(log_path, f'shape_change_of_KL_grades_from_OAI.txt')
path_config= '/dataNAS/people/anoopai/KneePipeline/config.json'

with open(path_config) as f:
    config = json.load(f)

model_types = ['bone_cartilage']
# model_types = ['bone_only']

model_type_names= {
    'bone_only': 'BO',
    'bone_cartilage': 'BC'
}

kl_grades = [0, 1, 2, 3, 4]

for model_type in model_types:
    print(f'Processing model type: {model_type}') 
    
    data_results_path= os.path.join(dir_path, 'results/BScore_and_FC_metrics')
    save_dir_path = os.path.join(data_results_path, f'mean_shape_recon')
    save_path = os.path.join(save_dir_path, f'{model_type_names[model_type]}_model/shape_change_of_KL_grades_from_OAI_wrt_previous_KLgrade')
    data_path = os.path.join(save_path, 'KL_grades_and_Bscores.txt')
    
    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
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

    # LOAD CONFIG
    with open(path_model_config, 'r') as f:
        model_config = json.load(f)
    
    # Load Model
    model = load_model(model_config, path_model_state, model_type='triplanar')
        
    # List to keep track of the dictionaries
    analysis_complete = []

    qdess_df_all= pd.DataFrame()
    nsm_df_all= pd.DataFrame()

    nsm_results_all = []

    try:
        
        #######################################################################################################
        # Load all baseline latents from the OAI. Create mean mesh for each KL grade.
        # Calculate Bscore for each mean KL grade. 
        # Compute signed distance of each  KL= 1, 2, 3 and 4 mean mesh w.r.t KL=0 mean mesh.
        ########################################################################################################
        
        folder_demographics = '/dataNAS/people/aagatti/projects/OAI_DESS/aging_trajectories/data/demographics'
        path_df = '/dataNAS/people/aagatti/projects/OAI_DESS/fit_nsm/results/647_nsm_femur_v0.0.1/2000/00m/latents.pkl'
        # /dataNAS/people/aagatti/projects/OAI_DESS/fit_nsm/results/551_nsm_femur_bone_v0.0.1_clipped/1150/00m/latents.pkl

        dict_demographic_filenames = {
            'baseline': '0_demographics_baseline.csv',
            '12_month': '1_demographics_12_month.csv',
            '24_month': '3_demographics_24_month.csv',
            '36_month': '5_demographics_36_month.csv',
            '48_month': '6_demographics_48_month.csv',
        }

        # load in the demographics data from baseline, 12, 24, and 48 months, and get the IDs of subjects (and knees) that are either healthy for all timepoints
        # kl = 0, and that had OA for all timepoints (kl >=2)

        for timepoint in ['baseline']:
            df_baseline = pd.read_csv(os.path.join(folder_demographics, dict_demographic_filenames[timepoint]))
            # create id_knee column so we include all individual knees
            df_baseline['id_knee'] = df_baseline['id'].astype(int).astype(str) + '_' + df_baseline['side'].astype(str)
                
                
        df_baseline['id_side'] = df_baseline['id'].astype(int).astype(str) + '_' + df_baseline['side'].astype(str)
        df_latents = pd.read_pickle(path_df)
        df_latents = df_latents.merge(df_baseline, on='id_side', how='left')
        
        
        # Convert strings in the 'latent' column to actual lists
        def parse_latents(latent_str):
            return eval(latent_str) if isinstance(latent_str, str) else latent_str

        # Apply the function to the 'latent' column
        df_latents['latent'] = df_latents['latent'].apply(parse_latents)

        # Prepare a DataFrame to store average latents
        average_latents = []

        # Iterate through each KL value (0 to 4, inclusive)
        for kl_grade in kl_grades:
            # Filter the DataFrame for the current KL grade
            filtered_latents = df_latents[df_latents['kl'] == kl_grade]['latent']

            # Check if there are any latents for the current KL
            if not filtered_latents.empty:
                # Stack the latents as arrays and calculate the mean
                latents_array = np.stack(filtered_latents.values)
                avg_latent = np.mean(latents_array, axis=0)  # Average across patients
            else:
                avg_latent = np.zeros((512,))  # Handle case where there are no values

            bscore = get_bscore(path_model_folder, avg_latent)
            
            # Append the KL grade and avg_latent to the list
            average_latents.append({'kl': kl_grade, 'B-Score': np.round(bscore,2), 'avg_latent': avg_latent})

            # Save the Bscore to a text file
            with open(data_path, 'a') as f:
                f.write(f'kl: {kl_grade}, B-Score: {np.round(bscore,2)} \n')
                
        ####################################################################################################################################
        # Generate Mean meshes for Bscores = 0, 1, 2, 3, 4... along the OA Bscore vector for 
        # Regsiter this B score meah to mean CTRL at Visit-1 mesh
        # Compute the deformation of each B-score mesh w.r.t. CTRL mesh at Visit-1
        ####################################################################################################################################

        for i, kl_grade in enumerate(kl_grades):
           
            latent_vector= average_latents[i]['avg_latent']
            latent_vector = latent_vector.squeeze()
            
            if i == 0: 

                if model_type == 'bone_only':
                    
                    ref_mesh_path_femur =os.path.join(save_path, f'MeanFemur_NSMrecon_OAI_KLgrade{kl_grade}.vtk')
                    if not os.path.exists(ref_mesh_path_femur):
                        bone_mesh= create_mean_mesh(latent_vector, model, model_type)
                        bone_mesh.save_mesh(ref_mesh_path_femur)

                elif model_type == 'bone_cartilage':
                    ref_mesh_path_femur = os.path.join(save_path, f'MeanFemur_NSMrecon_OAI_KLgrade{kl_grade}.vtk')
                    ref_mesh_path_cart = os.path.join(save_path, f'MeanCart_NSMrecon_OAI_KLgrade{kl_grade}.vtk')
                    if not os.path.exists(ref_mesh_path_femur) or not os.path.exists(ref_mesh_path_cart):
                        bone_mesh, cart_mesh= create_mean_mesh(latent_vector, model, model_type)
                        bone_mesh.save_mesh(ref_mesh_path_femur)
                        cart_mesh.save_mesh(ref_mesh_path_cart)
                        project_fc_thickness_on_femur_mesh(ref_mesh_path_femur, ref_mesh_path_cart)                   
                        
                else:
                    raise ValueError(f'Unknown model type: {model_type}')
            
            else:    
                if model_type == 'bone_only':
                    
                    mesh_path_femur =os.path.join(save_path, f'MeanFemur_NSMrecon_OAI_KLgrade{kl_grade}.vtk')
                    if not os.path.exists(mesh_path_femur):
                        bone_mesh= create_mean_mesh(latent_vector, model, model_type)
                        bone_mesh.save_mesh(mesh_path_femur)
                    
                    ref_latent_vector = average_latents[i-1]['avg_latent'].squeeze()
                    target_latent_vector = latent_vector 
                    ref_mesh_path_femur = os.path.join(save_path, f'MeanFemur_NSMrecon_OAI_KLgrade0.vtk')
                    
                    print(f'Interpolating between MeanFemur_NSMrecon_OAI_KLgrade0 and MeanFemur_NSMrecon_OAI_KLgrade{kl_grade}')
                    interp_mesh= interpolate_matching_points(ref_latent_vector, [ref_mesh_path_femur], target_latent_vector, model, model_type)
                    interp_mesh_path_femur =os.path.join(save_path, f'MeanFemur_NSMrecon_OAI_KLgrade0_to_KLgrade{kl_grade}.vtk')
                    interp_mesh.save_mesh(interp_mesh_path_femur)
                    print(f'Computing deformation of MeanFemur_NSMrecon_OAI_KLgrade0 along OA vector with KLgrade={kl_grade}')
                    deformed_mesh = compute_signed_distance_and_normal(ref_mesh_path_femur, interp_mesh_path_femur)
                    deformed_mesh.save_mesh(interp_mesh_path_femur)

                elif model_type == 'bone_cartilage':
                    mesh_path_femur = os.path.join(save_path, f'MeanFemur_NSMrecon_OAI_KLgrade{kl_grade}.vtk')
                    mesh_path_cart = os.path.join(save_path, f'MeanCart_NSMrecon_OAI_KLgrade{kl_grade}.vtk')
                    if not os.path.exists(mesh_path_femur) or not os.path.exists(mesh_path_cart):
                        bone_mesh, cart_mesh= create_mean_mesh(latent_vector, model, model_type)
                        bone_mesh.save_mesh(mesh_path_femur)
                        cart_mesh.save_mesh(mesh_path_cart)
                        project_fc_thickness_on_femur_mesh(mesh_path_femur, mesh_path_cart) 
                    
                    ref_latent_vector =  average_latents[i-1]['avg_latent'].squeeze()
                    target_latent_vector = latent_vector 
                    ref_mesh_path_femur = os.path.join(save_path, f'MeanFemur_NSMrecon_OAI_KLgrade0.vtk')
                    ref_mesh_path_cart = os.path.join(save_path, f'MeanCart_NSMrecon_OAI_KLgrade0.vtk')

                    interp_mesh_femur, interp_mesh_cart= interpolate_matching_points(ref_latent_vector, [ref_mesh_path_femur, ref_mesh_path_cart], target_latent_vector, model, model_type)
                    interp_mesh_path_femur =os.path.join(save_path, f'MeanFemur_NSMrecon_OAI_KLgrade0_to_KLgrade{kl_grade}.vtk')
                    interp_mesh_path_cart = os.path.join(save_path, f'MeanCart_NSMrecon_OAI_KLgrade0_to_KLgrade{kl_grade}.vtk')
                    interp_mesh_femur.save_mesh(interp_mesh_path_femur)
                    interp_mesh_cart.save_mesh(interp_mesh_path_cart)
                    project_fc_thickness_on_femur_mesh(interp_mesh_path_femur, interp_mesh_path_cart) 
    
                    print(f'Computing deformation of MeanFemur_NSMrecon_OAI_KLgrade0 along OA vector with KLgrade={kl_grade}')
                    deformed_mesh_femur = compute_signed_distance_and_normal(ref_mesh_path_femur, interp_mesh_path_femur)
                    deformed_mesh_femur.save_mesh(interp_mesh_path_femur) 
                    deformed_mesh_cart = compute_signed_distance_and_normal(ref_mesh_path_cart, interp_mesh_path_cart)
                    deformed_mesh_cart.save_mesh(interp_mesh_path_cart)
                    calculate_fc_thickess_changes(ref_mesh_path_femur, interp_mesh_path_femur)                    
                        
                else:
                    raise ValueError(f'Unknown model type: {model_type}')
                
    except Exception as e:
        # Print the error message
        print("An error occurred:", e)
        # Print the traceback
        traceback.print_exc()
        # Write the error message to the output file
        with open(output_file, 'a') as f:
            f.write(f"An error occurred: {e}\n")
            