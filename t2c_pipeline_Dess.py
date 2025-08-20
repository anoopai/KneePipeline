#!/usr/bin/env python
import sys
import os
import json
from dosma.scan_sequences import QDess
from dosma import MedicalVolume
import dosma as dm
from dosma.models import StanfordQDessBoneUNet2D, StanfordCubeBoneUNet2D
from dosma.tissues import FemoralCartilage

import SimpleITK as sitk
import pymskt as mskt
import numpy as np
import pandas as pd
import pyvista as pv
import warnings
import torch
import gc
import time
from utils.filter_t2maps import *

import pymskt.mesh.io as io
from pymskt.image import read_nrrd
from pymskt.mesh.meshTransform import SitkVtkTransformer
from pymskt.mesh.meshTools import ProbeVtkImageDataAlongLine
from pymskt.mesh.meshTools import get_surface_normals, n2l, l2n
from pymskt.mesh.meshes import BoneMesh, CartilageMesh
from pymskt.mesh.utils import is_hit, get_intersect, get_surface_normals, get_obb_surface

def main(path_image, path_save, path_save_t2c, path_config, model_name='acl_qdess_bone_july_2024'):
    print('Loading Inputs and Configurations...')
    # read two inputs arguments - where to get data, and where to save it. 
    path_image = sys.argv[1]
    path_save = sys.argv[2]
    path_save_t2c = sys.argv[3]
    
    # Get the model name... by default to most recent bone seg model. 
    if len(sys.argv) > 5:
        model_name = sys.argv[5]
    else: 
        model_name = 'acl_qdess_bone_july_2024'
        
    print('Path to image analyzing:', path_image)

    # Add path of current file to sys.path
    # sys.path.append(os.path.dirname(os.path.realpath(__file__)))

    # READ IN CONFIGURATION STUFF
    with open(path_config) as f:
        config = json.load(f)
    # get parameters for bone/cartilage reconstructions
    dict_bones = config['bones']
    # get the lists of the region names and their seg labels
    dict_regions_ = config['regions']
    # convert the regions to integers
    dict_regions = {}
    for tissue, tissue_dict in dict_regions_.items():
        dict_regions[tissue] = {}
        for region_num, region_name in tissue_dict.items():
            dict_regions[tissue][int(region_num)] = region_name


    print('Loading Image...')
    # figure out if the image is dicom, or nifti, or nrrd & load / get filename
    # as appropriate
    if os.path.isdir(path_image):
        filename_save = os.path.basename(path_image)
    
    ####################################################################################################################
    # Project T2C on Femur
    ####################################################################################################################
    path_save_t2c =  os.path.join(os.path.join(os.path.dirname(path_save)), 'results_t2c_new')
    
    bone_name = 'femur'
    femur_mesh_path = os.path.join(path_save, f'femur_mesh.vtk')
    fc_mesh_path  = os.path.join(path_save, f'femur_cart_0_mesh.vtk')
    sitk_seg_subregions_path  =  os.path.join(path_save, f'{filename_save}_subregions-labels.nrrd')
    sitk_seg_path = os.path.join(path_save, f'{filename_save}_all-labels.nrrd')
    
    # load meshes
    mesh_femur = BoneMesh(femur_mesh_path)
    mesh_fc = CartilageMesh(fc_mesh_path)
    sitk_seg_subregions = sitk.ReadImage(sitk_seg_subregions_path)
    sitk_seg = sitk.ReadImage(sitk_seg_path)

    # Assign mesh objects
    mesh_femur.seg_image = sitk_seg_subregions
    cart_labels = [11, 12, 13, 14, 15]
    mesh_femur.list_cartilage_labels=cart_labels
    mesh_femur.seg_image = sitk_seg
    mesh_femur.list_cartilage_meshes = [mesh_fc]

    # store this mesh in dict for later use
    dict_bones[bone_name]['mesh'] = mesh_femur

    # get the T2 map   
    map_path_nii= os.path.join(path_save, f'{filename_save}_t2map.nii.gz')
    map_path_filt_nii= os.path.join(path_save, f'{filename_save}_t2map_filt.nii.gz')
    map_path_filt_nrrd= map_path_filt_nii.replace('.nii.gz', '.nrrd')
    map_path_nrrd= os.path.join(path_save, f'{filename_save}_t2map.nrrd')
    
    def realign_seg_masks_SITK_format(map_path, seg_path):

        ''' 
        realigns the segmentation masks to the format used by SimpleITK
        Input: seg_path = (str), path of the segmentation file in .nii and .nrrd
        
        '''
        import SimpleITK as sitk
        import numpy as np
        
        map = sitk.ReadImage(map_path)
        seg = sitk.ReadImage(seg_path)
        array = sitk.GetArrayFromImage(map)
        array = np.transpose(array, (0, 2, 1)).astype(int)
        map_ = sitk.GetImageFromArray(array)
        map_.SetSpacing(seg.GetSpacing())
        map_.SetOrigin(seg.GetOrigin())
        map_.SetDirection(seg.GetDirection())
            
        # cast to int
        map_ = sitk.Cast(map_, sitk.sitkInt16)
        # sitk.WriteImage(map_, map_save_path, useCompression=True)

        return map_
   
    t2c_map_path = os.path.join(path_save_t2c, 't2c_subregions.nii.gz')
    map_sitk = realign_seg_masks_SITK_format(t2c_map_path, sitk_seg_path)
    t2c_map_path_nrrd= t2c_map_path.replace('.nii.gz', '.nrrd')
    sitk.WriteImage(map_sitk, t2c_map_path_nrrd)
    
    def project_T2_data(mesh_femur, mesh_fc, map_path_nrrd):
        
        nrrd_t2 = read_nrrd(map_path_nrrd, set_origin_zero=True).GetOutput()

        # apply inverse transform to the mesh (so its also at the origin)
        sitk_image = sitk.ReadImage(map_path_nrrd)
        nrrd_transformer = SitkVtkTransformer(sitk_image)
        mesh_fc.apply_transform_to_mesh(transform=nrrd_transformer.get_inverse_transform())
        mesh_femur.apply_transform_to_mesh(transform=nrrd_transformer.get_inverse_transform())

        # setup the probe that we are using to get data from the T2 file 
        line_resolution = 10000   # number of points along the line that the T2 data is sampled at
        filler = 0              # if no data is found, what value to fill the data with
        ray_length= 20.0          # how far to extend the ray from the surface (using negative to go inwards/towards the other side)
        percent_ray_length_opposite_direction = 0.25  # extend the other way a % of the line to make sure get both edges. 1.0 = 100%|

        data_probe = ProbeVtkImageDataAlongLine(
            line_resolution,
            nrrd_t2,
            save_mean=True,         # if we want mean. 
            save_max=True,          # if we want max
            save_std=True,          # if we want to see variation in the data along the line. 
            save_most_common=False, # for segmentations - to show the regions on the surface. 
            filler=filler
        )

        # get the points and normals from the mesh - this is what we'll iterate over to apply the probe to. 
        points = mesh_femur.mesh.GetPoints()
        normals = get_surface_normals(mesh_femur.mesh)
        point_normals = normals.GetOutput().GetPointData().GetNormals()

        # create an bounding box that we can query for intersections.
        obb_cartilage = get_obb_surface(mesh_fc.mesh)

        # iterate over the points & their normals. 
        for idx in range(points.GetNumberOfPoints()):
            # for each point get its x,y,z and normal
            point = points.GetPoint(idx)
            normal = point_normals.GetTuple(idx) 

            # get the start/end of the ray that we are going to use to probe the data.
            # this is based on the ray length info defind above. 
            end_point_ray = n2l(l2n(point) + ray_length*l2n(normal))
            start_point_ray = n2l(l2n(point) + ray_length*percent_ray_length_opposite_direction*(-l2n(normal)))

            # get the number of intersections and the cell ids that intersect.
            points_intersect, cell_ids_intersect = get_intersect(obb_cartilage, start_point_ray, end_point_ray)

            # if 2 intersections (the inside/outside of the cartilage) then probe along the line between these
            # intersections. Otherwise, fill the data with the filler value.
            if len(points_intersect) == 2:
                # use the intersections, not the ray length info
                # this makes sure we only get values inside of the surface. 
                start = np.asarray(points_intersect[0])
                end = np.asarray(points_intersect[1])

                # start = start + (start-end)
                # end = end + (end-start) 
                data_probe.save_data_along_line(start_pt=start, end_pt=end)
            else:
                data_probe.append_filler()
                
        # undo the transforms from above so that the mesh is put back to its original position.
        mesh_femur.reverse_all_transforms()
        
        return mesh_femur, data_probe

    # mesh_femur, data_probe_t2 = project_T2_data(mesh_femur, mesh_fc, map_path_nrrd)
    mesh_femur, data_probe_t2c = project_T2_data(mesh_femur, mesh_fc, t2c_map_path_nrrd)
    # mesh_femur.set_scalar('T2_max', data_probe_t2.max_data)
    # mesh_femur.set_scalar('T2_mean', data_probe_t2.mean_data)
    # mesh_femur.set_scalar('T2_std', data_probe_t2.std_data)
    
    mesh_femur.set_scalar('T2C', data_probe_t2c.mean_data)
    
    save_path = os.path.join(path_save, f'femur_mesh.vtk')
    mesh_femur.save_mesh(save_path)
    
    femur_mesh_org= dict_bones['femur']['mesh']
    femur_mesh_org.mesh = mesh_femur
    #####################################################################################################################        
    
    #####################################################################################################################
    # Chop Femur shaft to 90% of the S-I direction: Code added by Anoosha
    #####################################################################################################################
    mesh_org = pv.read(os.path.join(path_save, f'femur_mesh.vtk') )
    
    # Calculate the centroid of the mesh
    centroid = np.mean(mesh_org.points, axis=0)

    # Translate the mesh to center it at the origin (0, 0, 0)
    mesh = mesh_org.translate(-centroid)  # Move to origin

    # Get the bounds of the mesh
    bounds = mesh.bounds

    # Calculate the dimensions in each axis
    dimension_x = bounds[1] - bounds[0]
    dimension_y = bounds[3] - bounds[2]
    dimension_z = bounds[5] - bounds[4]

    print(f"Dimensions: x={dimension_x}, y={dimension_y}, z={dimension_z}")

    if dimension_z > 0.7 * dimension_y: # if the S-I dimension is greater than 70% of the R-L dimension
        # Define the cropping parameters
        z_min = bounds[4]   
        z_max = bounds[4] + 0.7 * abs(bounds[1] - bounds[0])  # Define your maximum x value

        # Debugging printed values
        # print(f"Org z bounds: z_min={bounds[4]}, z_max={bounds[5]}")
        # print(f"New z bounds: z_min= {z_min}, z_max={z_max}")

        # Clip along z-axis
        clipped_mesh = mesh.clip("z", value=z_max, invert=True) 

        clip_bounds= clipped_mesh.bounds
        # print(f"Clip z bounds: z_min={clip_bounds[4]}, z_max={clip_bounds[5]}")
        # print(f"Num of points before clipping: {mesh.n_points}")
        # print(f"Num of points remaining after clipping: {clipped_mesh.n_points}")
    
    else: # if the S-I dimension is less than 70% then just crop to 90% of the current S-I height to open up the top surface
        # Define the cropping parameters
        z_min = bounds[4]   
        z_max = bounds[4] + 0.95 * abs(bounds[5] - bounds[4])  # Define your maximum x value

        # Debugging printed values
        print(f"Org z b/s: z_min= {z_min}, z_max={z_max}")

        # Clip along z-axis
        clipped_mesh = mesh.clip("z", value=z_max, invert=True) 

        clip_bounds= clipped_mesh.bounds
        # print(f"Clip z bounds: z_min={clip_bounds[4]}, z_max={clip_bounds[5]}")
        # print(f"Num of points before clipping: {mesh.n_points}")
        # print(f"Num of points remaining after clipping: {clipped_mesh.n_points}")

    # Translate the mesh back to its original position
    clipped_mesh = clipped_mesh.translate(centroid)
    
    bone_name= 'femur'
    file_path= os.path.join(path_save, f'{bone_name}_mesh_clipped.vtk')
    io.write_vtk(clipped_mesh, file_path, points_dtype=float, write_binary=False)

    femur_mesh_org= dict_bones['femur']['mesh']
    femur_mesh_org.mesh = clipped_mesh
    dict_bones['femur']['mesh'] = femur_mesh_org
    #########################################################################################################################
    
    print('Saving Meshes for NSM Fitting...')
    seg_subregion_label_path = os.path.join(path_save, f'{filename_save}_subregions-labels.nrrd')
    sitk_seg_subregions = sitk.ReadImage(seg_subregion_label_path)
    seg_array = sitk.GetArrayFromImage(sitk_seg_subregions)
    
    # look at the config  says to analyze nsm bone or bone and cartilage... 
    if config['perform_bone_only_nsm'] or config['perform_bone_and_cart_nsm']:
        # figure out if right or left leg. Use the medial/lateral tibial cartilage to determine side
        loc_med_cart = np.mean(np.where(seg_array == 3), axis=1)
        loc_lat_cart = np.mean(np.where(seg_array == 4), axis=1)

        # get rotation matrix
        rotation_matrix = np.array(sitk_seg_subregions.GetDirection()).reshape(3,3)

        # flip the ijk becuase simpleitk returns them in the opposite order of the image space
        # then apply rotation matrix to get them in the correct xyz space orientation
        # not worrying about translations - only care about relative position along x (med/lat) axis.
        loc_med_cart_xyz = rotation_matrix @ loc_med_cart[::-1]
        loc_lat_cart_xyz = rotation_matrix @ loc_lat_cart[::-1]

        loc_med_cart_x = loc_med_cart_xyz[0]
        loc_lat_cart_x = loc_lat_cart_xyz[0]

        # get the xyz location, not the ijk index, for the medial and lateral tibial cartilages
        # then use this to determine the side of the knee.

        if loc_med_cart_x > loc_lat_cart_x:
            side = 'right'
        elif loc_med_cart_x < loc_lat_cart_x:
            side = 'left'

        # if side is left, flip the mesh to be a right knee
        if side == 'left':
            femur = dict_bones['femur']['mesh'] # in the future, copy this when BoneMesh has own copy method
            # get the center of the mesh - so we can translate it back to have the same "center"
            center = np.mean(femur.point_coords, axis=0)[0]
            femur.point_coords = femur.point_coords * [-1, 1, 1]
            # move the mesh back so the center is the same as before the flip along x-axis. 
            femur.point_coords = femur.point_coords + [2*center, 0, 0]
            # apply transformation to the cartilage mesh
            fem_cart = femur.list_cartilage_meshes[0].copy()
            fem_cart.point_coords = fem_cart.point_coords * [-1, 1, 1]
            fem_cart.point_coords = fem_cart.point_coords + [2*center, 0, 0]
        else:
            femur = dict_bones['femur']['mesh']
            fem_cart = femur.list_cartilage_meshes[0]

        # save the femur and cartilage meshes - this is in "NSM" format
        femur.save_mesh(os.path.join(path_save, 'femur_mesh_NSM_orig.vtk'))
        fem_cart.save_mesh(os.path.join(path_save, 'fem_cart_mesh_NSM_orig.vtk'))

if __name__ == "__main__":
    # Read command line arguments
    path_image = sys.argv[1]
    path_save = sys.argv[2]
    path_save_t2c = sys.argv[3]
    model_name = sys.argv[5] if len(sys.argv) > 5 else 'acl_qdess_bone_july_2024'
    
    # set path to config as config.json in current directory
    path_config = os.path.join(os.path.dirname(__file__), 'config.json')
    
    main(path_image, path_save, path_save_t2c, path_config, model_name)
