import os
import vtk
import numpy as np
import pandas as pd
import seaborn as sbn
import matplotlib.pyplot as plt
from sympy import per
from utils.append_df_to_excel import append_df_to_excel
from splot.esda import plot_moran
from splot.esda import moran_scatterplot
from esda.moran import Moran_Local
from scipy.spatial.distance import pdist, squareform
from libpysal.weights import DistanceBand
from esda.moran import Moran
from pysal.lib import weights
from splot.esda import moran_scatterplot
from pymskt.mesh import Mesh, BoneMesh
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from pymskt.mesh.meshTools import smooth_scalars_from_second_mesh_onto_base, transfer_mesh_scalars_get_weighted_average_n_closest


visits = ['VISIT-2', 'VISIT-3', 'VISIT-4', 'VISIT-5']
knees = ['aclr', 'clat', 'ctrl']
scalar_names = ['Diff_thickness (mm)']
mdc_scalar_name = 'Diff_thickness (mm)_MDC'
sig_values = [0.05]
k_neighbors_list = [10, 16, 20, 30, 50]
permutations= [999]
save_meshes = False
analysis_all = pd.DataFrame()

dir_path = '/dataNAS/people/anoopai/DESS_ACL_study'
code_dir_path = '/dataNAS/people/anoopai/KneePipeline/'
data_path = os.path.join(dir_path, 'data')
log_path = '/dataNAS/people/anoopai/KneePipeline/logs'
log_file_path = os.path.join(log_path, f'pipeline_DESS_spatial_autocorrelation_withMDC_NSMrecon.txt')
mean_path = os.path.join(code_dir_path, 'mean_data')
save_path = os.path.join(mean_path, f'spatial_autocorrelation/NSMrecon')
mean_femur_path = os.path.join(mean_path, 'MeanFemur_mesh_B-only.vtk')
mdc_femur_path = os.path.join(save_path, 'MDC_Mean_NSMrecon_femur_mesh_ctrl_VISIT-2.vtk')
output_file = os.path.join(save_path, f'spatial_autocorrelation.xlsx')

mdc_femur = BoneMesh(mdc_femur_path)

if os.path.exists(output_file):
    os.remove(output_file)
############################ Interpretation of Moran's I, Z score, and significant level ##############################################

# The p-value is not statistically significant: You cannot reject the null hypothesis. 
# It is quite possible that the spatial distribution of feature values is the result of random spatial processes. 
# The observed spatial pattern of feature values could very well be one of many, 
# many possible versions of complete spatial randomness (CSR).

# The p-value is statistically significant, and the z-score is positive: You may reject the null hypothesis. 
# The spatial distribution of high values and/or low values in the dataset is more spatially clustered than 
# would be expected if underlying spatial processes were random.

# The p-value is statistically significant, and the z-score is negative: You may reject the null hypothesis. 
# The spatial distribution of high values and low values in the dataset is more spatially dispersed than 
# would be expected if underlying spatial processes were random. A dispersed spatial pattern often reflects 
# some type of competitive process—a feature with a high value repels other features with high values; 
# similarly, a feature with a low value repels other features with low values.

######################################################################################################################################

# Function to read scalar data from a VTK file
def get_mesh_points_and_scalars(file_name, scalar_name, mdc_file_name):
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(file_name)
    reader.Update()
    femur_mesh = BoneMesh(file_name)
    mdc_mesh = BoneMesh(mdc_file_name)
    
    # Verify the data was correctly read
    data = reader.GetOutput()
    if data is None:
        raise ValueError("Failed to read the VTK file.")

    # Check points in the dataset
    points = data.GetPoints()
    if points is None:
        raise ValueError("No point data found in VTK file.")

    # Check for scalar data
    point_data = data.GetPointData()
    if point_data is None:
        raise ValueError("No point data section found in VTK file.")
    
    # scalars = point_data.GetScalars()

    points_all = np.array([points.GetPoint(i) for i in range(points.GetNumberOfPoints())])
    # scalars_all = np.array([scalars.GetTuple(i)[0] for i in range(scalars.GetNumberOfTuples())])
    scalars_all = vtk_to_numpy(femur_mesh.GetPointData().GetArray(scalar_name))
    mdc_scalars_all = vtk_to_numpy(mdc_mesh.GetPointData().GetArray(scalar_name))
    
    mdc_mask = (scalars_all < -mdc_scalars_all) | (scalars_all > mdc_scalars_all)

    # Set those values to -200
    scalars_all[~mdc_mask] = -200

    # Filter out points where scalars are -100 or -200
    mask = (scalars_all != -100) & (scalars_all != -200)
    
    # Apply mask to filter points and scalars
    points_filt = points_all[mask]
    scalars_filt = scalars_all[mask]
    
    return points_all, scalars_all, points_filt, scalars_filt, mask

# Function to compute Moran's I using libpysal
def compute_morans_i(locations, values, k_neighbors=10, permutation=999):
    
    from libpysal.weights import KNN
    from esda.moran import Moran, Moran_Local
    
    # Create a KNN spatial weights matrix
    weights_knn = KNN.from_array(locations, k=k_neighbors)
    
    # Compute Moran's I and Local Moran's I
    moran = Moran(values, weights_knn, permutations=permutation)
    moran_loc = Moran_Local(values, weights_knn, permutations=permutation)
    
    return moran, moran_loc, weights_knn

# Main function to run the analysis
for k_neighbors in k_neighbors_list:
    for permutation in permutations:
        for knee in knees:
            for visit in visits:                
                vtk_file = os.path.join(save_path, f'Mean_NSMrecon_femur_mesh_{knee}_{visit}.vtk') # Replace with your VTK file path

                for scalar_name in scalar_names:
                    # Step 1: Read VTK scalars data
                    print(f'Processing {knee.upper()} knee at {visit} for scalar {scalar_name}')
                    points, scalars, points_filt, scalars_filt, mask = get_mesh_points_and_scalars(vtk_file, scalar_name, mdc_femur)

                    if points_filt.size > 0:
                        # Step 3: Compute and print Moran's I
                        moran, moran_loc, weights_moran = compute_morans_i(points_filt, scalars_filt, k_neighbors=k_neighbors, permutation=permutation)
                        
                        print(f'{scalar_name}, K_neighbours: {k_neighbors} permutation: {permutation}, knee: {knee.upper()}, visit: {visit}, Moran I: {moran.I:.4f}, P-value: {moran.p_sim:.4f}')
                        
                        analysis_info = pd.DataFrame({
                            'knee': [knee],
                            'visit': [visit],
                            'k_neighbors': [k_neighbors],
                            'permutation': [permutation],
                            'moran_I': [f'{moran.I:.4f}'],
                            'P-value sim': [f'{moran.p_sim:.4f}'],
                            'z-score sim': [f'{moran.z_sim:.4f}'],  
                            'P-value Z sim': [f'{moran.p_z_sim:.4f}'],
                        })
                        
                        analysis_all = pd.concat([analysis_all, analysis_info], ignore_index=True)
                        
                        append_df_to_excel(analysis_info, sheet=scalar_name, save_path=output_file)
                        
                        # # Step 4: Plot Moran's Global and Local 
                        # plot_moran(moran, zstandard=True, figsize=(10,4), aspect_equal=False)
                        # plt.text(-3.5, 1.5, f'Moran I: {moran.I:.4f}', fontsize=12)
                        # plt.text(-3.5, 1.0, f'p-value: {moran.p_sim:.4f}', fontsize=12)
                        # plt.title(f'{knee.upper()} knee at {visit}')
                        # plt.savefig(os.path.join(save_path, f'MoranGlobal_{knee}_{visit}_knn{k_neighbors}_perm{permutation}.png'))
                        # plt.close()

                        # fig = plt.figure(figsize=(5, 5))
                        # ax = fig.add_subplot(111)
                        # moran_scatterplot(moran_loc, p=0.05, aspect_equal=False, ax=ax)

                        # plt.text(0.9, 0.9, 'HH', fontsize=15)
                        # plt.text(-0.9, -0.9, 'LL', fontsize=15)
                        # plt.text(-0.9, 0.9, 'LH', fontsize=15)
                        # plt.text(0.9, -0.9, 'HL', fontsize=15)
                        # plt.savefig(os.path.join(save_path, f'MoranLocal_{knee}_{visit}_knn{k_neighbors}_perm{permutation}.png'))
                        # plt.close()
                        
                        if save_meshes:

                            for sig_value in sig_values:
                                j = 0 
                                # Step 5: Save the results to a new VTK file
                                femur_save_path = os.path.join(save_path, f'Moran_{knee}_{visit}_knn{k_neighbors}_perm{permutation}_sig{sig_value}.vtk') # Replace with your VTK file path
                                if os.path.exists(femur_save_path):
                                    femur_mesh= BoneMesh(femur_save_path)
                                else:
                                    femur_mesh = BoneMesh(vtk_file)
                                    
                                scalar = vtk_to_numpy(femur_mesh.GetPointData().GetArray(scalar_name)) 
                                # direction = vtk_to_numpy(femur_mesh.GetPointData().GetArray('direction'))
                                # spearman_r = vtk_to_numpy(femur_mesh.GetPointData().GetArray('spearman_r'))
                                # diff_t2_mean_all = vtk_to_numpy(femur_mesh.GetPointData().GetArray('Diff_T2_mean_filt_all'))
                                scalar_sig = np.zeros_like(scalar).astype(float)  # New array for modified scalar values
                                direction_sig = np.zeros_like(scalar).astype(float)  # New array for modified direction values
                                spearman_r_sig = np.zeros_like(scalar).astype(float)  # New array for modified spearman_r values
                                diff_t2_mean_all_sig = np.zeros_like(scalar).astype(float)  # New array for modified diff_t2_mean_all values
                                
                                # Initialize arrays for Moran's I and other statistics
                                moran_I = np.zeros_like(scalar).astype(float)
                                moran_z = np.zeros_like(scalar).astype(float)
                                moran_p = np.ones_like(scalar).astype(float)
                                moran_q = np.zeros_like(scalar).astype(float)  
                                moran_EI = np.zeros_like(scalar).astype(float)
                                moran_z_sim = np.zeros_like(scalar).astype(float)
                                moran_p_sim = np.ones_like(scalar).astype(float)
                                moran_I_sig = np.zeros_like(scalar).astype(float)
                                moran_q_sig = np.zeros_like(scalar).astype(float)
                                moran_p_z_sim = np.ones_like(scalar).astype(float)

                                # Assuming moran_loc and mask are already defined
                                for i, data in enumerate(mask):
                                    if mask[i]:
                                        moran_I[i] = moran_loc.Is[j]
                                        moran_z[i] = moran_loc.z[j]
                                        moran_p[i] = moran_loc.p_sim[j]
                                        moran_q[i] = moran_loc.q[j]
                                        moran_EI[i] = moran_loc.EI[j]
                                        moran_z_sim[i] = moran_loc.z_sim[j]
                                        moran_p_sim[i] = moran_loc.p_sim[j]
                                        moran_p_z_sim[i] = 2 * moran_loc.p_z_sim[j]  # Two-tailed test
                                        if (moran_p_sim[i] < sig_value) and (moran_z_sim[i] > 1.96): # pnly if the z-sim is greater than 0 and significant
                                            moran_I_sig[i] = moran_I[i]
                                            scalar_sig[i] = scalar[i]
                                            moran_q_sig[i] = moran_q[i]
                                            # spearman_r_sig[i] = spearman_r[i]
                                            # diff_t2_mean_all_sig[i] = diff_t2_mean_all[i]
                                            # direction_sig[i] = direction[i]
                                        else:
                                            moran_I_sig[i] = 0.00
                                            scalar_sig[i] = 0.00
                                            moran_q_sig[i]= 0.00
                                            # direction_sig[i] = 0.00
                                            # spearman_r_sig[i] = 0.00
                                            # diff_t2_mean_all_sig[i] = 0.00
                                        j += 1
                                    else:
                                        moran_I[i] = 0.00
                                        moran_z[i] = 0.00
                                        moran_p[i] = 0.00
                                        moran_q[i] = 0.00
                                        moran_EI[i] = 0.00
                                        moran_z_sim[i] = 0.00
                                        moran_p_sim[i] = 0.00
                                        moran_p_z_sim[i] = 0.00
                                        moran_I_sig[i] = 0.00
                                        moran_q_sig[i]= 0.00
                                        scalar_sig[i] = 0.00 # Retain original scalar values for unchanged indices
                                        # direction_sig[i] = 0.00
                                        
                                # Create dictionaries to store the values for the mesh
                                moran_I_dict = moran_I
                                moran_z_dict = moran_z
                                moran_p_dict = moran_p
                                moran_q_dict = moran_q
                                moran_EI_dict = moran_EI
                                moran_z_sim_dict = moran_z_sim
                                morna_p_sim_dict = moran_p_sim
                                moran_p_z_sim_dict = moran_p_z_sim
                                moran_I_sig_dict = moran_I_sig
                                moran_q_sig_dict = moran_q_sig
                                scalar_sig_dict = scalar_sig
                                # direction_sig_dict = direction_sig
                                # spearman_r_sig_dict = spearman_r_sig
                                # diff_t2_mean_all_sig_dict = diff_t2_mean_all_sig

                                # Assign the new values to the mesh
                                femur_mesh.point_data[f'{scalar_name}_I'] = moran_I_dict
                                # femur_mesh.point_data['moran_z'] = moran_z_dict
                                # femur_mesh.point_data['moran_p'] = moran_p_dict
                                femur_mesh.point_data[f'{scalar_name}_q'] = moran_q_dict
                                # femur_mesh.point_data['moran_EI'] = moran_EI_dict
                                femur_mesh.point_data[f'{scalar_name}_z_sim'] = moran_z_sim_dict
                                femur_mesh.point_data[f'{scalar_name}_p_sim'] = morna_p_sim_dict
                                femur_mesh.point_data[f'{scalar_name}_p_z_sim'] = moran_p_z_sim_dict
                                femur_mesh.point_data[f'{scalar_name}_I_sig'] = moran_I_sig_dict
                                femur_mesh.point_data[f'{scalar_name}_q_sig'] = moran_q_sig_dict
                                femur_mesh.point_data[f'{scalar_name}_sig'] = scalar_sig_dict
                                # femur_mesh.point_data[f'direction_sig'] = direction_sig_dict
                                # femur_mesh.point_data[f'spearman_r_sig'] = spearman_r_sig_dict
                                # femur_mesh.point_data[f'Diff_T2_mean_filt_all_sig'] = diff_t2_mean_all_sig_dict
                                
                                # femur_mesh.save_mesh(vtk_file)
                                femur_mesh.save(femur_save_path)
                            
                else:
                    
                    analysis_info = pd.DataFrame({
                            'knee': [knee],
                            'visit': [visit],
                            'k_neighbors': [k_neighbors],
                            'permutation': [permutation],
                            'moran_I': 0,
                            'P-value sim': 1,
                            'z-score sim': 0,  
                            'P-value Z sim': 1,
                        })
                        
                    analysis_all = pd.concat([analysis_all, analysis_info], ignore_index=True)
                    append_df_to_excel(analysis_info, sheet=scalar_name, save_path=output_file)
                    
                    if save_meshes:
                        for sig_value in sig_values:
                            j = 0 
                            # Step 5: Save the results to a new VTK file
                            femur_save_path = os.path.join(save_path, f'Moran_{knee}_{visit}_knn{k_neighbors}_perm{permutation}_sig{sig_value}.vtk') # Replace with your VTK file path
                            if os.path.exists(femur_save_path):
                                femur_mesh= BoneMesh(femur_save_path)
                            else:
                                femur_mesh = BoneMesh(vtk_file)
                                
                            scalar = vtk_to_numpy(femur_mesh.GetPointData().GetArray(scalar_name)) 
                            scalar_sig = np.zeros_like(scalar).astype(float)  # New array for modified scalar values
                            
                            # Initialize arrays for Moran's I and other statistics
                            moran_I = np.zeros_like(scalar).astype(float)
                            moran_z = np.zeros_like(scalar).astype(float)
                            moran_p = np.ones_like(scalar).astype(float)
                            moran_q = np.zeros_like(scalar).astype(float)  
                            moran_EI = np.zeros_like(scalar).astype(float)
                            moran_z_sim = np.zeros_like(scalar).astype(float)
                            moran_p_sim = np.ones_like(scalar).astype(float)
                            moran_I_sig = np.zeros_like(scalar).astype(float)
                            moran_q_sig = np.zeros_like(scalar).astype(float)
                            moran_p_z_sim = np.ones_like(scalar).astype(float)

                            # Create dictionaries to store the values for the mesh
                            moran_I_dict = moran_I
                            moran_z_dict = moran_z
                            moran_p_dict = moran_p
                            moran_q_dict = moran_q
                            moran_EI_dict = moran_EI
                            moran_z_sim_dict = moran_z_sim
                            morna_p_sim_dict = moran_p_sim
                            moran_p_z_sim_dict = moran_p_z_sim
                            moran_I_sig_dict = moran_I_sig
                            moran_q_sig_dict = moran_q_sig
                            scalar_sig_dict = scalar_sig

                            # Assign the new values to the mesh
                            femur_mesh.point_data[f'{scalar_name}_I'] = moran_I_dict
                            femur_mesh.point_data[f'{scalar_name}_q'] = moran_q_dict
                            femur_mesh.point_data[f'{scalar_name}_z_sim'] = moran_z_sim_dict
                            femur_mesh.point_data[f'{scalar_name}_p_sim'] = morna_p_sim_dict
                            femur_mesh.point_data[f'{scalar_name}_p_z_sim'] = moran_p_z_sim_dict
                            femur_mesh.point_data[f'{scalar_name}_I_sig'] = moran_I_sig_dict
                            femur_mesh.point_data[f'{scalar_name}_q_sig'] = moran_q_sig_dict
                            femur_mesh.point_data[f'{scalar_name}_sig'] = scalar_sig_dict
                        
