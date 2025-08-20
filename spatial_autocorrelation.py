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
knees = ['ctrl']
sig_values = [0.05]
distance_bands = [0.05, 0.1, 0.5]
weight_decays = [-1, -5, -10, -20]
permutations= [999]
analysis_all = pd.DataFrame()

dir_path = '/dataNAS/people/anoopai/DESS_ACL_study'
code_dir_path = '/dataNAS/people/anoopai/KneePipeline/'
data_path = os.path.join(dir_path, 'data')
log_path = '/dataNAS/people/anoopai/KneePipeline/logs'
log_file_path = os.path.join(log_path, f'pipeline_DESS_errors.txt')
mean_path = os.path.join(code_dir_path, 'mean_data')
save_path = os.path.join(mean_path, f't2filt_and_thickness_change')
mean_femur_path = os.path.join(mean_path, 'MeanFemur_mesh_B-only.vtk')
mdc_femur_path = os.path.join(code_dir_path, 'MDC_controls/MeanFemur_mesh_B-only_ctrl_VISIT-2_MDC.vtk')

mdc_femur = BoneMesh(mdc_femur_path)

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
def get_mesh_points_and_scalars(file_name):
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(file_name)
    reader.Update()
    femur_mesh = BoneMesh(file_name)

    data = reader.GetOutput()
    points = data.GetPoints()
    scalars = data.GetPointData().GetScalars()

    if scalars is None:
        raise ValueError("No scalar data found in VTK file")

    all_points = np.array([points.GetPoint(i) for i in range(points.GetNumberOfPoints())])
    # all_scalars = np.array([scalars.GetTuple(i)[0] for i in range(scalars.GetNumberOfTuples())])
    all_scalars = vtk_to_numpy(femur_mesh.GetPointData().GetArray('spearman_r'))
    
    # Filter out points where scalars are -100 or -200
    mask = (all_scalars != -100) & (all_scalars != -200)

    # Apply mask to filter points and scalars
    points_filt = all_points[mask]
    scalars_filt = all_scalars[mask]
    
    return all_points, all_scalars, points_filt, scalars_filt, mask

# Function to compute Moran's I using libpysal
def compute_morans_i(locations, values, distance_band, weight_decay, permutation):
    from splot.esda import moran_scatterplot
    from esda.moran import Moran_Local
    from scipy.spatial.distance import pdist, squareform
    from libpysal.weights import DistanceBand
    from esda.moran import Moran
    from pysal.lib import weights

    # calculate Moran_Local and plot
    weights_moran = DistanceBand(locations, threshold=distance_band, alpha=weight_decay, binary=False)
    
    # weights = weights.contiguity.Queen(locations)
    moran = Moran(values, weights_moran, permutations=permutation)
    moran_loc = Moran_Local(values, weights_moran, permutations=permutation)
    
    return moran, moran_loc, weights_moran

# Main function to run the analysis
for distance_band in distance_bands:
    for weight_decay in weight_decays:
        for permutation in permutations:
            for knee in knees:
                output_file = os.path.join(save_path, f'spatial_autocorrelation_sensitivity_analysis_{knee}.xlsx')
                for visit in visits:                
                    vtk_file = os.path.join(save_path, f'MeanFemur_mesh_B-only_{knee}_{visit}.vtk') # Replace with your VTK file path

                    # Step 1: Read VTK scalars data
                    points, scalars, points_filt, scalars_filt, mask = get_mesh_points_and_scalars(vtk_file)

                    # Step 3: Compute and print Moran's I
                    moran, moran_loc, weights_moran = compute_morans_i(points_filt, scalars_filt, distance_band, weight_decay, permutation)
                    print(f'Distance: {distance_band}, Weight decay: {weight_decay}, knee: {knee.upper()}, visit: {visit}, Moran I: {moran.I:.4f}, P-value: {moran.p_sim:.4f}')
                    
                    analysis_info = pd.DataFrame({
                        'knee': [knee],
                        'visit': [visit],
                        'distance_band': [distance_band],
                        'weight_decay': [weight_decay],
                        'permutation': [permutation],
                        'moran_I': [f'{moran.I:.4f}'],
                        'P-value sim': [f'{moran.p_sim:.4f}'],
                        'z-score sim': [f'{moran.z_sim:.4f}'],  
                        'E(I) sim': [f'{moran.EI_sim:.4f}'],
                        'VI sim': [f'{moran.VI_sim:.4f}'],
                        'P-value Z sim': [f'{moran.p_z_sim:.4f}'],
                    })
                    
                    analysis_all = pd.concat([analysis_all, analysis_info], ignore_index=True)
                    
                    append_df_to_excel(analysis_info, sheet=knee, save_path=output_file)
                    
                    # Step 4: Plot Moran's Global and Local 
                    plot_moran(moran, zstandard=True, figsize=(10,4), aspect_equal=False)
                    plt.text(-3.5, 1.5, f'Moran I: {moran.I:.4f}', fontsize=12)
                    plt.text(-3.5, 1.0, f'p-value: {moran.p_sim:.4f}', fontsize=12)
                    plt.title(f'{knee.upper()} knee at {visit}')
                    plt.savefig(os.path.join(save_path, f'Moran_global_{knee}_{visit}_dist{distance_band}_decay{weight_decay}_perm{permutation}.png'))
                    plt.close()

                    # fig = plt.figure(figsize=(5, 5))
                    # ax = fig.add_subplot(111)
                    # moran_scatterplot(moran_loc, p=0.05, aspect_equal=False, ax=ax)

                    # plt.text(0.9, 0.9, 'HH', fontsize=15)
                    # plt.text(-0.9, -0.9, 'LL', fontsize=15)
                    # plt.text(-0.9, 0.9, 'LH', fontsize=15)
                    # plt.text(0.9, -0.9, 'HL', fontsize=15)
                    # plt.savefig(os.path.join(save_path, f'Moran_local_{knee}_{visit}_dist{distance_band}_decay{weight_decay}.png'))
                    # plt.close()

                    # Step 5: Save the results to a new VTK file
                    femur_mesh= BoneMesh(vtk_file)
                    spearman_r = vtk_to_numpy(femur_mesh.GetPointData().GetArray('spearman_r'))
                    spearman_r_sig = np.zeros_like(spearman_r).astype(float)  # New array for modified spearman_r values

                    moran_I = np.zeros_like(spearman_r).astype(float)
                    moran_z = np.zeros_like(spearman_r).astype(float)
                    moran_p = np.ones_like(spearman_r).astype(float)
                    moran_q = np.zeros_like(spearman_r).astype(float)  
                    moran_EI = np.zeros_like(spearman_r).astype(float)
                    moran_z_sim = np.zeros_like(spearman_r).astype(float)
                    moran_p_sim = np.ones_like(spearman_r).astype(float)
                    moran_I_sig = np.zeros_like(spearman_r).astype(float)
                    moran_p_z_sim = np.ones_like(spearman_r).astype(float)

                    # Assuming moran_loc and mask are already defined
                    for sig_value in sig_values:
                        j = 0
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
                                    spearman_r_sig[i] = spearman_r[i]
                                else:
                                    moran_I_sig[i] = moran_I_sig[i]
                                    spearman_r_sig[i] = -200.00
                                j += 1
                            else:
                                moran_I[i] = moran_I[i]
                                moran_z[i] = moran_z[i]
                                moran_p[i] = moran_p[i]
                                moran_q[i] = moran_q[i]
                                moran_EI[i] = moran_EI[i]
                                moran_z_sim[i] = moran_z_sim[i]
                                moran_p_sim[i] = moran_p_sim[i]
                                moran_p_z_sim[i] = moran_p_z_sim[i]
                                moran_I_sig[i] = moran_I_sig[i]
                                spearman_r_sig[i] = spearman_r[i]  # Retain original spearman_r values for unchanged indices

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
                        spearman_r_sig_dict = spearman_r_sig

                        # Assign the new values to the mesh
                        femur_mesh.point_data['moran_I'] = moran_I_dict
                        femur_mesh.point_data['moran_z'] = moran_z_dict
                        femur_mesh.point_data['moran_p'] = moran_p_dict
                        femur_mesh.point_data['moran_q'] = moran_q_dict
                        femur_mesh.point_data['moran_EI'] = moran_EI_dict
                        femur_mesh.point_data['moran_z_sim'] = moran_z_sim_dict
                        femur_mesh.point_data['moran_p_sim'] = morna_p_sim_dict
                        femur_mesh.point_data['moran_p_z_sim'] = moran_p_z_sim_dict
                        femur_mesh.point_data['moran_I_sig'] = moran_I_sig_dict
                        femur_mesh.point_data['spearman_r_sig'] = spearman_r_sig_dict

                        # femur_mesh.save_mesh(os.path.join(save_path, f'{knee}_{visit}_dist{distance_band}_p{sig_value}.vtk'))
                        femur_mesh.save_mesh(os.path.join(save_path, f'{knee}_{visit}_dist{distance_band}_decay{weight_decay}_perm{permutation}.vtk'))

# output_file = os.path.join(save_path, f'spatial_autocorrelation_sensitivity_analysis{knee}.xlsx')
# # Step 6: Save the analysis results to an Excel file
# analysis_all.to_excel(output_file, index=False)