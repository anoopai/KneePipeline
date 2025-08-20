from re import sub
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
import subprocess
import warnings
from pathlib import Path
import traceback

# Make sure knees_to_use is defined
knees_to_use = ['aclr']
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# dir_path = '/dataNAS/people/anoopai/DESS_ACL_study'
dir_path = '/dataNAS/people/anoopai/KneePipeline'
data_path = os.path.join(dir_path, 'data')
log_path = '/dataNAS/people/anoopai/KneePipeline/logs'
output_file = os.path.join(log_path , f'T2projectionFemur_{knees_to_use[0]}.txt')

# Clear the output file if it exists or create a new one
with open(output_file, 'w') as f:
    f.write('')

# List to keep track of the dictionaries
analysis_complete = []

for knee_to_use in knees_to_use:
    for subject in os.listdir(data_path):
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

                            if os.path.isdir(knee_path) and knee == knee_to_use:

                                path_image = os.path.join(knee_path, 'scans/qdess')
                                path_save = os.path.join(knee_path, 'results_nsm')
                                
                                # Ensure path_save directory exists
                                if not os.path.exists(path_save):
                                    os.makedirs(path_save)

                                # Split the path into components
                                sub_component = Path(knee_path).parts[6]  # '11-P'
                                visit_component = Path(knee_path).parts[7]  # 'VISIT-1'
                                knee_component = Path(knee_path).parts[8]  # 'clat'
                                
                                if os.path.exists(path_image) and not os.listdir(path_save):
                         
                                    # Collect the extracted information in a dictionary
                                    analysis_info = {
                                        'sub': sub_component,
                                        'visit': visit_component,
                                        'knee': knee_component
                                        }
                                
                                    print(f"Performing T2C analysis for {analysis_info}")
                    
                                    # Append the dictionary to the list
                                    analysis_complete.append(analysis_info)

                                    try:        
                                        print('Analyzing:', knee_path)
                                        
                                        # Get the model name... by default to most recent bone seg model. 
                                        model_name = 'acl_qdess_bone_july_2024'
                                            
                                        print('Path to image analyzing:', path_image)

                                        # Add path of current file to sys.path
                                        # sys.path.append(os.path.dirname(os.path.realpath(__file__)))


                                        ################################################ READ IN CONFIGURATION STUFF ############################################
                                        # path_config = '/bmrNAS/people/aagatti/projects/auto_seg_server/Segmentation/config.json'
                                        path_config= '/dataNAS/people/anoopai/KneePipeline/config.json'
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
                                        sitk.WriteImage(sitk_seg, os.path.join(path_save, filename_save + '_all-labels.nrrd'))

                                        #################################### MESH CREATION AND CARTILAGE THICKNESS COMPUTATION ####################################
                                        print('Creating Meshes and Computing Cartilage Thickness...')
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

                                        # create 3D surfaces w/ cartilage thickness & compute thickness metrics
                                        dict_results = {}
                                        for bone_name, dict_ in dict_bones.items():
                                            # create bone mesh and crop as appropriate
                                            bone_mesh = mskt.mesh.BoneMesh(
                                                seg_image=sitk_seg,
                                                label_idx=dict_['tissue_idx'],
                                                list_cartilage_labels=dict_['list_cart_labels'],
                                                bone=bone_name,
                                                crop_percent=dict_['crop_percent'],
                                            )
                                            bone_mesh.create_mesh(smooth_image_var=0.5)
                                            bone_mesh.resample_surface(clusters=dict_['n_points'])
                                             
                                            # compute cartilage thickness metrics - on surfaces
                                            bone_mesh.calc_cartilage_thickness(image_smooth_var_cart=0.3125)
                                            bone_mesh.seg_image = sitk_seg_subregions
                                            
                                            # get labels to compute thickness metrics
                                            if bone_name == 'femur':
                                                cart_labels = [11, 12, 13, 14, 15]
                                                bone_mesh.list_cartilage_labels=cart_labels
                                            else:
                                                cart_labels = dict_['list_cart_labels']
                                            # assign labels to bone surface
                                            bone_mesh.assign_cartilage_regions()
                                            
                                            # store this mesh in dict for later use
                                            dict_bones[bone_name]['mesh'] = bone_mesh
                                            
                                            # # get thickness and region for each bone vertex
                                            # thickness = np.array(bone_mesh.get_scalar('thickness (mm)'))
                                            # regions = np.array(bone_mesh.get_scalar('labels'))
                                            
                                            # # for each region, compute thicknes statics. 
                                            # for region in cart_labels:
                                            #     dict_results[f"{dict_regions['cart'][region]}_mm_mean"] = np.nanmean(thickness[regions == region])
                                            #     dict_results[f"{dict_regions['cart'][region]}_mm_std"] = np.nanstd(thickness[regions == region])
                                            #     dict_results[f"{dict_regions['cart'][region]}_mm_median"] = np.nanmedian(thickness[regions == region])
                                            
                                            # save the bone and cartilage meshes. 
                                            bone_mesh.save_mesh(os.path.join(path_save, f'{bone_name}_mesh.vtk'))
                                            # iterate over the cartilage meshes associated with the bone_mesh and save: 
                                            for cart_idx, cart_mesh in enumerate(bone_mesh.list_cartilage_meshes):
                                                cart_mesh.save_mesh(os.path.join(path_save, f'{bone_name}_cart_{cart_idx}_mesh.vtk'))

                                        ########################################## NSM (FEMUR) FITTING ############################################
                                        print('Fitting NSM Model & Saving Results...')

                                        # Assign the segmentation
                                        seg_array = sitk.GetArrayFromImage(sitk_seg_subregions)

                                        # figure out if right or left leg. Use the medial/lateral tibial cartilage to determine side
                                        loc_med_cart_x = np.mean(np.where(seg_array == 3), axis=1)[0]
                                        loc_lat_cart_x = np.mean(np.where(seg_array == 4), axis=1)[0]
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
                                            
                                        # Call NSM_analysis.py script with the paths of the saved
                                        # femur and cartilage meshes to fit the NSM to & the path 
                                        # to save the results.
                                        arguments = [
                                            'python',
                                            '/dataNAS/people/anoopai/KneePipeline/NSM_analysis.py', 
                                            os.path.join(path_save, 'femur_mesh_NSM_orig.vtk'), 
                                            os.path.join(path_save, 'fem_cart_mesh_NSM_orig.vtk'), 
                                            path_save]

                                        # submit the NSM analysis script
                                        result = subprocess.run(arguments, capture_output=True, text=True)

                                        # Access the output and return code from running the NSM analysis script
                                        output = result.stdout
                                        error = result.stderr
                                        exit_code = result.returncode

                                        # print the output, error, and exit code
                                        print("NSM Script Output:", output)
                                        if error != '':
                                            print("NSM Script Error:", error)
                                        if exit_code != 0:
                                            print("NSM Script Exit Code:", exit_code)

                                        ###################################### COMPUTE T2 MAPS & T2 METRICS ##################################

                                        # include_required_tags = (
                                        #     (qdess.get_metadata(qdess.__GL_AREA_TAG__, None) is not None)
                                        #     and (qdess.get_metadata(qdess.__TG_TAG__, None) is not None)
                                        # )
                                        # if (qdess is not None) and include_required_tags:
                                        if (qdess is not None):
                                            
                                            print('Computing T2 Maps and Metrics...')

                                            # See if gl and tg private tags are present, if not, skip T2 computation
                                            # create T2 map and clip values

                                            cart = FemoralCartilage()
                                            # t2map = qdess.generate_t2_map(cart, suppress_fat=True, suppress_fluid=True)
                                            t2map= qdess.generate_t2_map(cart, tr= 0.01766e3, te= 0.005924e3, tg= 0.001904e6, alpha= 20, gl_area=3132)

                                            # convert to sitk for mean T2 computation
                                            sitk_t2map = t2map.volumetric_map.to_sitk(image_orientation='sagittal')

                                            # save the t2 map
                                            sitk.WriteImage(sitk_t2map, os.path.join(path_save, f'{filename_save}_t2map.nii.gz'))
                                            sitk.WriteImage(sitk_t2map, os.path.join(path_save, f'{filename_save}_t2map.nrrd'))

                                            seg_array = sitk.GetArrayFromImage(sitk_seg_subregions)

                                            # get T2 as array and set values outside of expected/reasonable range to nan
                                            t2_array = sitk.GetArrayFromImage(sitk_t2map)
                                            t2_array[t2_array>=80] = np.nan
                                            t2_array[t2_array<=0] = np.nan
                                            
                                            ####################### ADDED by Anoosha:  #######################################
                                            ##################### redo bone meshing #####################                        
                                            print('Creating Meshes and Computing Cartilage Thickness...')
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

                                            # create 3D surfaces w/ cartilage thickness & compute thickness metrics
                                            dict_results = {}
                                            for bone_name, dict_ in dict_bones.items():
                                                # create bone mesh and crop as appropriate
                                                bone_mesh = mskt.mesh.BoneMesh(
                                                    seg_image=sitk_seg,
                                                    label_idx=dict_['tissue_idx'],
                                                    list_cartilage_labels=dict_['list_cart_labels'],
                                                    bone=bone_name,
                                                    crop_percent=dict_['crop_percent'],
                                                )
                                                bone_mesh.create_mesh(smooth_image_var=0.5)
                                                bone_mesh.resample_surface(clusters=dict_['n_points'])
                                                
                                                # compute cartilage thickness metrics - on surfaces
                                                bone_mesh.calc_cartilage_thickness(image_smooth_var_cart=0.3125)
                                                bone_mesh.seg_image = sitk_seg_subregions
                                                
                                                # get labels to compute thickness metrics
                                                if bone_name == 'femur':
                                                    cart_labels = [11, 12, 13, 14, 15]
                                                    bone_mesh.list_cartilage_labels=cart_labels
                                                else:
                                                    cart_labels = dict_['list_cart_labels']
                                                # assign labels to bone surface
                                                bone_mesh.assign_cartilage_regions()
                                                
                                                # store this mesh in dict for later use
                                                dict_bones[bone_name]['mesh'] = bone_mesh
                                                
                                                # get thickness and region for each bone vertex
                                                thickness = np.array(bone_mesh.get_scalar('thickness (mm)'))
                                                regions = np.array(bone_mesh.get_scalar('labels'))
                                                
                                                # for each region, compute thicknes statics. 
                                                for region in cart_labels:
                                                    dict_results[f"{dict_regions['cart'][region]}_mm_mean"] = np.nanmean(thickness[regions == region])
                                                    dict_results[f"{dict_regions['cart'][region]}_mm_std"] = np.nanstd(thickness[regions == region])
                                                    dict_results[f"{dict_regions['cart'][region]}_mm_median"] = np.nanmedian(thickness[regions == region])
                                                
                                                # save the bone and cartilage meshes. 
                                                bone_mesh.save_mesh(os.path.join(path_save, f'{bone_name}_mesh.vtk'))
                                                # iterate over the cartilage meshes associated with the bone_mesh and save: 
                                                for cart_idx, cart_mesh in enumerate(bone_mesh.list_cartilage_meshes):
                                                    cart_mesh.save_mesh(os.path.join(path_save, f'{bone_name}_cart_{cart_idx}_mesh.vtk'))
                                            ##################################################################################
                                            
                                            # compute T2 metrics for each region & store in results dictionary
                                            for cart_idx, cart_region in dict_regions['cart'].items():
                                                if cart_idx in seg_array:
                                                    mean_t2 = np.nanmean(t2_array[seg_array == cart_idx])
                                                    std_t2 = np.nanstd(t2_array[seg_array == cart_idx])
                                                    median_t2 = np.nanmedian(t2_array[seg_array == cart_idx])
                                                    dict_results[f'{cart_region}_t2_ms_mean'] = mean_t2
                                                    dict_results[f'{cart_region}_t2_ms_std'] = std_t2
                                                    dict_results[f'{cart_region}_t2_ms_median'] = median_t2
                                            
                                            # convert segmentation into simple depth dependent version of the segmentation.
                                            for bone_name, dict_ in dict_bones.items():
                                                bone_mesh = dict_['mesh']
                                                # update bone_mesh list_cartilage_labels to be the original ones
                                                # this is only really needed for the femur, but we do it for all bones... just in case. 
                                                bone_mesh.list_cartilage_labels = dict_['list_cart_labels']
                                                # assign the segmentation mask to be the original one.. 
                                                bone_mesh.seg_image = sitk_seg
                                                bone_new_seg, bone_rel_depth = bone_mesh.break_cartilage_into_superficial_deep(rel_depth_thresh=0.5, return_rel_depth=True, resample_cartilage_surface=10_000)
                                                dict_['bone_new_seg'] = bone_new_seg
                                                dict_['bone_rel_depth'] = bone_rel_depth
                                            new_seg_combined = mskt.image.cartilage_processing.combine_depth_region_segs(
                                                sitk_seg_subregions,
                                                [x['bone_new_seg'] for x in dict_bones.values()],
                                            )
                                            
                                            # compute T2 metrics for each region & store in results dictionary
                                            # store as superficial / deep T2 maps. 
                                            seg_array_depth = sitk.GetArrayFromImage(new_seg_combined)
                                            for cart_idx, cart_region in dict_regions['cart'].items():
                                                for depth_idx, depth_name in [(100, 'deep'), (200, 'superficial')]:
                                                    cart_idx_depth = cart_idx + depth_idx
                                                    if cart_idx_depth in seg_array_depth:
                                                        mean_t2 = np.nanmean(t2_array[seg_array_depth == cart_idx_depth])
                                                        std_t2 = np.nanstd(t2_array[seg_array_depth == cart_idx_depth])
                                                        median_t2 = np.nanmedian(t2_array[seg_array_depth == cart_idx_depth])
                                                        dict_results[f'{cart_region}_{depth_name}_t2_ms_mean'] = mean_t2
                                                        dict_results[f'{cart_region}_{depth_name}_t2_ms_std'] = std_t2
                                                        dict_results[f'{cart_region}_{depth_name}_t2_ms_median'] = median_t2
                                                    
                                        else: 
                                            print("Not computing T2 metrics.")

                                            
                                        print('Saving Results...')
                                        # SAVE THICKNESS & T2 METRICS
                                        # save as csv
                                        df = pd.DataFrame([dict_results])
                                        df.to_csv(os.path.join(path_save, f'{filename_save}_results.csv'), index=False)

                                        # save as json
                                        with open(os.path.join(path_save, f'{filename_save}_results.json'), 'w') as f:
                                            json.dump(dict_results, f, indent=4)
                                            
                                    except Exception as e:
                                        log_file_path = os.path.join(log_path , 'knee_pipeline_errors.txt')
                                        with open(log_file_path, 'a') as f:
                                            f.write(f'\n ###################################################################################### \n')
                                            f.write(f'\n ERROR!!! {knee_path}\n')
                                            f.write(traceback.format_exc())
                                            
                                    # # Save the dictionary to the output file
                                    with open(output_file, 'a') as f:
                                        f.write(str(analysis_info) + '\n')
                                                
                                else:
                                    # with open(output_file, 'a') as f:
                                    #     f.write(str(analysis_info) + '\n')
                                    log_file_path = os.path.join(log_path , 'knee_pipeline_errors.txt')
                                    with open(log_file_path, 'a') as f:
                                        f.write(f'\n, No qdess folder found or already analysed: {knee_path}')
                                                    
