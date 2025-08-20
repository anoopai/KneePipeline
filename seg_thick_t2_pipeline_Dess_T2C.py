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

import pymskt.mesh.io as io
from pymskt.image import read_nrrd
from pymskt.mesh.meshTransform import SitkVtkTransformer
from pymskt.mesh.meshTools import ProbeVtkImageDataAlongLine
from pymskt.mesh.meshTools import get_surface_normals, n2l, l2n
from pymskt.mesh.meshes import BoneMesh, CartilageMesh
from pymskt.mesh.utils import is_hit, get_intersect, get_surface_normals, get_obb_surface

def main(path_image, path_save, path_config, model_name='acl_qdess_bone_july_2024'):
    print('Loading Inputs and Configurations...')
    # read two inputs arguments - where to get data, and where to save it. 
    path_image = sys.argv[1]
    path_save = sys.argv[2]
    # Get the model name... by default to most recent bone seg model. 
    if len(sys.argv) > 4:
        model_name = sys.argv[4]
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
        # read in dicom image using DOSMA
        try:
            # qdess = QDess.from_dicom(path_image)
            qdess= QDess.load(path_image)
        except KeyError:
            qdess = QDess.from_dicom(path_image, group_by='EchoTime')
        volume = qdess.calc_rss()
        filename_save = os.path.basename(path_image)
    elif path_image.endswith(('nii', 'nii.gz')):
        # read in nifti using DOSMA
        # ASSUMING NIFTI SAVED AS RSS /POSTPROCESSED ALREADY
        qdess = None
        nr = dm.NiftiReader()
        volume = nr.load(path_image)
        filename_save = os.path.basename(path_image).split('.nii')[0]
    elif path_image.endswith('nrrd'):
        # read in using SimpleITK, then convert to DOSMA
        qdess = None
        image = sitk.ReadImage(path_image)
        volume = MedicalVolume.from_sitk(image)
        filename_save = os.path.basename(path_image).split('.nrrd')[0]
    else:
        raise ValueError('Image format not supported.')

    seg_path= os.path.join(path_save, filename_save + '_all-labels.nrrd')
    
    if os.path.exists(seg_path):
        seg = sitk.ReadImage(seg_path)
        model = None
    else:
        print('Loading Model...')
        # load the appropriate segmentation model & its weights
        if 'bone' in model_name:
            # set the actual model class being used
            if 'cube' in model_name:
                model_class = StanfordCubeBoneUNet2D
            else:
                model_class = StanfordQDessBoneUNet2D
            # load the model. 
            model = model_class(config['models'][model_name], orig_model_image_size=(512,512))
        else:
            raise ValueError('Model name not supported.')

        print('Segmenting Image...')
        # SEGMENT THE MRI
        seg = model.generate_mask(volume)

        # save the segmentation as nifti
        nw = dm.NiftiWriter()
        nw.save(seg['all'], os.path.join(path_save, filename_save + '_all-labels.nii.gz'))

        # convert seg to sitk format for pymskt processing
        sitk_seg = seg['all'].to_sitk(image_orientation='sagittal')
        # save the segmentation as nrrd
        sitk.WriteImage(sitk_seg, seg_path, useCompression=True)
    
    print('Creating Meshes and Computing Cartilage Thickness...')
    
    seg_subregion_label_path = os.path.join(path_save, f'{filename_save}_subregions-labels.nrrd')
    if os.path.exists(seg_subregion_label_path):
        sitk_seg_subregions = sitk.ReadImage(seg_subregion_label_path)
    else:
        # break segmentation into subregions
        sitk_seg_subregions = mskt.image.cartilage_processing.get_knee_segmentation_with_femur_subregions(
            sitk_seg,
            fem_cart_label_idx=2,
            wb_region_percent_dist=0.6,
            femur_label=7,
            med_tibia_label=3,
            lat_tibia_label=4,
            ant_femur_mask=11,
            med_wb_femur_mask=12,
            lat_wb_femur_mask=13,
            med_post_femur_mask=14,
            lat_post_femur_mask=15,
            verify_med_lat_tib_cart=True,
            tibia_label=8,
            ml_axis=0,
        )
        
        # save the subregions to disk
        sitk.WriteImage(sitk_seg_subregions, os.path.join(path_save, f'{filename_save}_subregions-labels.nrrd'), useCompression=True)
        sitk.WriteImage(sitk_seg_subregions, os.path.join(path_save, f'{filename_save}_subregions-labels.nii.gz'), useCompression=True)
            
    # create 3D surfaces w/ cartilage thickness & compute thickness metrics
    dict_results = {}
    # for bone_name, dict_ in dict_bones.items():
    #     # create bone mesh and crop as appropriate 
    #     bone_mesh = mskt.mesh.BoneMesh(
    #         seg_image=sitk_seg,
    #         label_idx=dict_['tissue_idx'],
    #         list_cartilage_labels=dict_['list_cart_labels'],
    #         bone=bone_name,
    #         crop_percent=dict_['crop_percent'],
    #     )
    #     bone_mesh.create_mesh(smooth_image_var=0.5)
    #     bone_mesh.resample_surface(clusters=dict_['n_points'])
        
    #     # fix bone mesh
    #     bone_mesh.fix_mesh()
        
    #     # compute cartilage thickness metrics - on surfaces
    #     bone_mesh.calc_cartilage_thickness(image_smooth_var_cart=0.3125)
    #     bone_mesh.seg_image = sitk_seg_subregions
        
    #     # fix cartilage surface
    #     for cart_mesh in bone_mesh.list_cartilage_meshes:
    #         cart_mesh.fix_mesh()
        
    #     # get labels to compute thickness metrics
    #     if bone_name == 'femur':
    #         cart_labels = [11, 12, 13, 14, 15]
    #         bone_mesh.list_cartilage_labels=cart_labels
    #     else: 
    #         cart_labels = dict_['list_cart_labels']
    #     # assign labels to bone surface
    #     bone_mesh.assign_cartilage_regions() 
        
    #     # store this mesh in dict for later use
    #     dict_bones[bone_name]['mesh'] = bone_mesh
        
    #     # get thickness and region for each bone vertex
    #     thickness = np.array(bone_mesh.get_scalar('thickness (mm)'))
    #     regions = np.array(bone_mesh.get_scalar('labels'))
        
    #     # for each region, compute thicknes statics. 
    #     for region in cart_labels:
    #         dict_results[f"{dict_regions['cart'][region]}_mm_mean"] = np.nanmean(thickness[regions == region])
    #         dict_results[f"{dict_regions['cart'][region]}_mm_std"] = np.nanstd(thickness[regions == region])
    #         dict_results[f"{dict_regions['cart'][region]}_mm_median"] = np.nanmedian(thickness[regions == region])
        
    #     # save the bone and cartilage meshes. 
    #     bone_mesh.save_mesh(os.path.join(path_save, f'{bone_name}_mesh.vtk'))
    #     # iterate over the cartilage meshes associated with the bone_mesh and save: 
    #     for cart_idx, cart_mesh in enumerate(bone_mesh.list_cartilage_meshes):
    #         cart_mesh.save_mesh(os.path.join(path_save, f'{bone_name}_cart_{cart_idx}_mesh.vtk'))
            
    # print('Computing T2 Maps and Metrics...')
    # # need seg_array for preprocessing related to NSM fitting. 
    # seg_array = sitk.GetArrayFromImage(sitk_seg_subregions)
    
    # if (qdess is not None):
    #     include_required_tags = (
    #         (qdess.get_metadata(qdess.__GL_AREA_TAG__, None) is not None)
    #         and (qdess.get_metadata(qdess.__TG_TAG__, None) is not None)
    #     )
    #     if include_required_tags:
    #         # See if gl and tg private tags are present, if not, skip T2 computation
    #         # create T2 map and clip values
    #         cart = FemoralCartilage()
    #         # t2map = qdess.generate_t2_map(cart, suppress_fat=True, suppress_fluid=True)
    #         t2map= qdess.generate_t2_map(cart, tr= 0.01766e3, te= 0.005924e3, tg= 0.001904e6, alpha= 20, gl_area=3132)

    #         # convert to sitk for mean T2 computation
    #         sitk_t2map = t2map.volumetric_map.to_sitk(image_orientation='sagittal')
            
    #         # save the t2 map
    #         sitk.WriteImage(sitk_t2map, os.path.join(path_save, f'{filename_save}_t2map.nii.gz'), useCompression=True)
    #         sitk.WriteImage(sitk_t2map, os.path.join(path_save, f'{filename_save}_t2map.nrrd'), useCompression=True)

    #         seg_array = sitk.GetArrayFromImage(sitk_seg_subregions)

    #         # get T2 as array and set values outside of expected/reasonable range to nan
    #         t2_array = sitk.GetArrayFromImage(sitk_t2map)
    #         t2_array[t2_array>=80] = np.nan
    #         t2_array[t2_array<=0] = np.nan
            
    #         # t2_array[t2_array>=80] = 80
    #         # t2_array[t2_array<=0] = 0
            
    #         # compute T2 metrics for each region & store in results dictionary
    #         for cart_idx, cart_region in dict_regions['cart'].items():
    #             if cart_idx in seg_array:
    #                 mean_t2 = np.nanmean(t2_array[seg_array == cart_idx])
    #                 std_t2 = np.nanstd(t2_array[seg_array == cart_idx])
    #                 median_t2 = np.nanmedian(t2_array[seg_array == cart_idx])
    #                 dict_results[f'{cart_region}_t2_ms_mean'] = mean_t2
    #                 dict_results[f'{cart_region}_t2_ms_std'] = std_t2
    #                 dict_results[f'{cart_region}_t2_ms_median'] = median_t2
            
    #         # convert segmentation into simple depth dependent version of the segmentation.
    #         for bone_name, dict_ in dict_bones.items():
    #             bone_mesh = dict_['mesh']
    #             # update bone_mesh list_cartilage_labels to be the original ones
    #             # this is only really needed for the femur, but we do it for all bones... just in case. 
    #             bone_mesh.list_cartilage_labels = dict_['list_cart_labels']
    #             # assign the segmentation mask to be the original one.. 
    #             bone_mesh.seg_image = sitk_seg
    #             bone_new_seg, bone_rel_depth = bone_mesh.break_cartilage_into_superficial_deep(rel_depth_thresh=0.5, return_rel_depth=True, resample_cartilage_surface=10_000)
    #             dict_['bone_new_seg'] = bone_new_seg
    #             dict_['bone_rel_depth'] = bone_rel_depth
    #         new_seg_combined = mskt.image.cartilage_processing.combine_depth_region_segs(
    #             sitk_seg_subregions,
    #             [x['bone_new_seg'] for x in dict_bones.values()],
    #         )
            
    #         # save the depth dependent segmentation to disk 
    #         sitk.WriteImage(new_seg_combined, os.path.join(path_save, f'{filename_save}_depth_seg.nrrd'), useCompression=True)
            
    #         # compute T2 metrics for each region & store in results dictionary
    #         # store as superficial / deep T2 maps. 
    #         seg_array_depth = sitk.GetArrayFromImage(new_seg_combined)
    #         for cart_idx, cart_region in dict_regions['cart'].items():
    #             for depth_idx, depth_name in [(100, 'deep'), (200, 'superficial')]:
    #                 cart_idx_depth = cart_idx + depth_idx
    #                 if cart_idx_depth in seg_array_depth:
    #                     mean_t2 = np.nanmean(t2_array[seg_array_depth == cart_idx_depth])
    #                     std_t2 = np.nanstd(t2_array[seg_array_depth == cart_idx_depth])
    #                     median_t2 = np.nanmedian(t2_array[seg_array_depth == cart_idx_depth])
    #                     dict_results[f'{cart_region}_{depth_name}_t2_ms_mean'] = mean_t2
    #                     dict_results[f'{cart_region}_{depth_name}_t2_ms_std'] = std_t2
    #                     dict_results[f'{cart_region}_{depth_name}_t2_ms_median'] = median_t2
                  
    #     else:
    #         warnings.warn(
    #             'GL and TG tags not present. Skipping T2 computation. '+
    #             'NOTE: These are private tags and may have been removed ' +
    #             'in the DICOM anonymization process.')
    # else:
    #     print('Not a qdess image. Skipping T2 computation.')

    # print('Saving Results...')
    # # SAVE THICKNESS & T2 METRICS
    # # save as csv
    # df = pd.DataFrame([dict_results])
    # df.to_csv(os.path.join(path_save, f'{filename_save}_results.csv'), index=False)

    # # save as json
    # with open(os.path.join(path_save, f'{filename_save}_results.json'), 'w') as f:
    #     json.dump(dict_results, f, indent=4)
        
    #####################################################################################################################
    # Project T2 on Femur
    #####################################################################################################################
    # get the femur mesh and the femoral cartilage mesh
    # mesh_femur = dict_bones['femur']['mesh']
    # mesh_fc = mesh_femur.list_cartilage_meshes[0]

    # # get the T2 map
    # map_path_nrrd= os.path.join(path_save, f'{filename_save}_t2map.nrrd')
    # nrrd_t2 = read_nrrd(map_path_nrrd, set_origin_zero=True).GetOutput()

    # # apply inverse transform to the mesh (so its also at the origin)
    # sitk_image = sitk.ReadImage(map_path_nrrd)
    # nrrd_transformer = SitkVtkTransformer(sitk_image)
    # mesh_fc.apply_transform_to_mesh(transform=nrrd_transformer.get_inverse_transform())
    # mesh_femur.apply_transform_to_mesh(transform=nrrd_transformer.get_inverse_transform())

    # # setup the probe that we are using to get data from the T2 file 
    # line_resolution = 10000   # number of points along the line that the T2 data is sampled at
    # filler = 0              # if no data is found, what value to fill the data with
    # ray_length= 20.0          # how far to extend the ray from the surface (using negative to go inwards/towards the other side)
    # percent_ray_length_opposite_direction = 0.25  # extend the other way a % of the line to make sure get both edges. 1.0 = 100%|

    # data_probe = ProbeVtkImageDataAlongLine(
    #     line_resolution,
    #     nrrd_t2,
    #     save_mean=True,         # if we want mean. 
    #     save_max=True,          # if we want max
    #     save_std=True,          # if we want to see variation in the data along the line. 
    #     save_most_common=False, # for segmentations - to show the regions on the surface. 
    #     filler=filler
    # )

    # # get the points and normals from the mesh - this is what we'll iterate over to apply the probe to. 
    # points = mesh_femur.mesh.GetPoints()
    # normals = get_surface_normals(mesh_femur.mesh)
    # point_normals = normals.GetOutput().GetPointData().GetNormals()

    # # create an bounding box that we can query for intersections.
    # obb_cartilage = get_obb_surface(mesh_fc.mesh)

    # # iterate over the points & their normals. 
    # for idx in range(points.GetNumberOfPoints()):
    #     # for each point get its x,y,z and normal
    #     point = points.GetPoint(idx)
    #     normal = point_normals.GetTuple(idx) 

    #     # get the start/end of the ray that we are going to use to probe the data.
    #     # this is based on the ray length info defind above. 
    #     end_point_ray = n2l(l2n(point) + ray_length*l2n(normal))
    #     start_point_ray = n2l(l2n(point) + ray_length*percent_ray_length_opposite_direction*(-l2n(normal)))

    #     # get the number of intersections and the cell ids that intersect.
    #     points_intersect, cell_ids_intersect = get_intersect(obb_cartilage, start_point_ray, end_point_ray)

    #     # if 2 intersections (the inside/outside of the cartilage) then probe along the line between these
    #     # intersections. Otherwise, fill the data with the filler value.
    #     if len(points_intersect) == 2:
    #         # use the intersections, not the ray length info
    #         # this makes sure we only get values inside of the surface. 
    #         start = np.asarray(points_intersect[0])
    #         end = np.asarray(points_intersect[1])

    #         # start = start + (start-end)
    #         # end = end + (end-start) 
    #         data_probe.save_data_along_line(start_pt=start,
    #                                         end_pt=end)
    #     else:
    #         data_probe.append_filler()
            
    # # undo the transforms from above so that the mesh is put back to its original position.
    # mesh_femur.reverse_all_transforms()

    # mesh_femur.set_scalar('T2_max', data_probe.max_data)
    # mesh_femur.set_scalar('T2_mean', data_probe.mean_data)
    # mesh_femur.set_scalar('T2_std', data_probe.std_data)
    
    # save_path = os.path.join(path_save, f'femur_mesh.vtk')
    # mesh_femur.save_mesh(save_path)
    
    # femur_mesh_org= dict_bones['femur']['mesh']
    # femur_mesh_org.mesh = mesh_femur
    #####################################################################################################################
        
    #####################################################################################################################
    # Project T2C on Femur
    #####################################################################################################################
    if os.path.exists(path_save_t2c):
        
        print("Projecting T2C on Femur") 
        mesh_femur = BoneMesh(os.path.join(path_save, f'femur_mesh.vtk')) 
        mesh_fc = CartilageMesh(os.path.join(path_save, f'femur_cart_0_mesh.vtk'))
        mesh_femur.list_cartilage_meshes = mesh_fc
          
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
    
        t2c_map_path = os.path.join(path_save_t2c, 't2_difference_map_size_threshold.nii.gz')
        map_sitk = realign_seg_masks_SITK_format(t2c_map_path, seg_path)
        t2c_map_path_nrrd= t2c_map_path.replace('.nii.gz', '.nrrd')
        sitk.WriteImage(map_sitk, t2c_map_path_nrrd)

        # get the T2 map
        nrrd_t2c = read_nrrd(t2c_map_path_nrrd, set_origin_zero=True).GetOutput()

        # apply inverse transform to the mesh (so its also at the origin)
        sitk_image = sitk.ReadImage(t2c_map_path_nrrd)
        nrrd_transformer = SitkVtkTransformer(sitk_image)
        mesh_fc.apply_transform_to_mesh(transform=nrrd_transformer.get_inverse_transform())
        mesh_femur.apply_transform_to_mesh(transform=nrrd_transformer.get_inverse_transform())

        # setup the probe that we are using to get data from the T2 file 
        line_resolution = 100000   # number of points along the line that the T2 data is sampled at
        filler = 0              # if no data is found, what value to fill the data with
        ray_length= 20.0          # how far to extend the ray from the surface (using negative to go inwards/towards the other side)
        percent_ray_length_opposite_direction = 0.5  # extend the other way a % of the line to make sure get both edges. 1.0 = 100%|

        data_probe = ProbeVtkImageDataAlongLine(
            line_resolution,
            nrrd_t2c,
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
                
                start = start + (start-end) * 0.1
                end = end + (end-start) * 0.1
                
                # start = start 
                # end = end 
                data_probe.save_data_along_line(start_pt=start,
                                                end_pt=end)
            else:
                data_probe.append_filler()
                
        # undo the transforms from above so that the mesh is put back to its original position.
        mesh_femur.reverse_all_transforms()

        mesh_femur.set_scalar('T2C_max', data_probe.max_data)
        mesh_femur.set_scalar('T2C_mean', data_probe.mean_data)
        mesh_femur.set_scalar('T2C_std', data_probe.std_data)
        
        save_path = os.path.join(path_save, f'femur_mesh.vtk')
        mesh_femur.save_mesh(save_path)
        # femur_mesh_org= dict_bones['femur']['mesh']
        # femur_mesh_org.mesh = mesh_femur
        
        dict_bones['femur']['mesh'] = mesh_femur.mesh
        
        #####################################################################################################################
    # Project T2C on Femur
    #####################################################################################################################
    if os.path.exists(path_save_t2c):
        
        print("Projecting T2C on Femur") 
        mesh_femur = BoneMesh(os.path.join(path_save, f'femur_mesh.vtk')) 
        mesh_fc = CartilageMesh(os.path.join(path_save, f'femur_cart_0_mesh.vtk'))
        mesh_femur.list_cartilage_meshes = mesh_fc
          
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
    
        t2c_map_path = os.path.join(path_save_t2c, 't2_difference_map_size_threshold.nii.gz')
        map_sitk = realign_seg_masks_SITK_format(t2c_map_path, seg_path)
        t2c_map_path_nrrd= t2c_map_path.replace('.nii.gz', '.nrrd')
        sitk.WriteImage(map_sitk, t2c_map_path_nrrd)

        # get the T2 map
        nrrd_t2c = read_nrrd(t2c_map_path_nrrd, set_origin_zero=True).GetOutput()

        # apply inverse transform to the mesh (so its also at the origin)
        sitk_image = sitk.ReadImage(t2c_map_path_nrrd)
        nrrd_transformer = SitkVtkTransformer(sitk_image)
        mesh_fc.apply_transform_to_mesh(transform=nrrd_transformer.get_inverse_transform())
        mesh_femur.apply_transform_to_mesh(transform=nrrd_transformer.get_inverse_transform())

        # setup the probe that we are using to get data from the T2 file 
        line_resolution = 100000   # number of points along the line that the T2 data is sampled at
        filler = 0              # if no data is found, what value to fill the data with
        ray_length= -10.0          # how far to extend the ray from the surface (using negative to go inwards/towards the other side)
        percent_ray_length_opposite_direction = 0.5  # extend the other way a % of the line to make sure get both edges. 1.0 = 100%|

        data_probe = ProbeVtkImageDataAlongLine(
            line_resolution,
            nrrd_t2c,
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
        
        thickness_data = []
        
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
                
                start = start + (start-end) * 0.1
                end = end + (end-start) * 0.1
                
                thickness_data.append(
                        np.sqrt(
                            np.sum(np.square(l2n(points_intersect[0]) - l2n(points_intersect[1])))
                        )
                    )
                
                # start = start 
                # end = end 
                data_probe.save_data_along_line(start_pt=start,
                                                end_pt=end)
            else:
                data_probe.append_filler()
                
        # undo the transforms from above so that the mesh is put back to its original position.
        mesh_femur.reverse_all_transforms()

        mesh_femur.set_scalar('T2C_max', data_probe.max_data)
        mesh_femur.set_scalar('T2C_mean', data_probe.mean_data)
        mesh_femur.set_scalar('T2C_std', data_probe.std_data)
        
        save_path = os.path.join(path_save, f'femur_mesh.vtk')
        mesh_femur.save_mesh(save_path)
        # femur_mesh_org= dict_bones['femur']['mesh']
        # femur_mesh_org.mesh = mesh_femur
        
        dict_bones['femur']['mesh'] = mesh_femur.mesh
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

        # femur_mesh_org= dict_bones['femur']['mesh']
        # femur_mesh_org.mesh = clipped_mesh
        # dict_bones['femur']['mesh'] = femur_mesh_org
        clipped_mesh = BoneMesh(file_path)
        # clipped_mesh.list_cartilage_meshes[0] = mesh_fc
        dict_bones['femur']['mesh'] = clipped_mesh.mesh
    #########################################################################################################################
    
        print('Saving Meshes for NSM Fitting...')
        
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
                # fem_cart = femur.list_cartilage_meshes[0].copy()
                fem_cart = mesh_fc
                fem_cart.point_coords = fem_cart.point_coords * [-1, 1, 1]
                fem_cart.point_coords = fem_cart.point_coords + [2*center, 0, 0]
            else:
                femur = dict_bones['femur']['mesh']
                # fem_cart = femur.list_cartilage_meshes[0]
                fem_cart = mesh_fc

            # save the femur and cartilage meshes - this is in "NSM" format
            femur.save_mesh(os.path.join(path_save, 'femur_mesh_NSM_orig.vtk'))
            fem_cart.save_mesh(os.path.join(path_save, 'fem_cart_mesh_NSM_orig.vtk'))

    # Memory cleanup
    print('Cleaning up memory...')
    if model:
        del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    time.sleep(5)

if __name__ == "__main__":
    # Read command line arguments
    path_image = sys.argv[1]
    path_save = sys.argv[2]
    model_name = sys.argv[4] if len(sys.argv) > 4 else 'acl_qdess_bone_july_2024'
    
    # set path to config as config.json in current directory
    path_config = os.path.join(os.path.dirname(__file__), 'config.json')
    
    main(path_image, path_save,path_config, model_name)
