import os

from pymskt.mesh import get_icp_transform, cpd_register, non_rigidly_register
from pymskt import mesh
from pymskt.mesh.meshTools import smooth_scalars_from_second_mesh_onto_base, transfer_mesh_scalars_get_weighted_average_n_closest
from pymskt.mesh import Mesh, BoneMesh

dir_path = '/dataNAS/people/anoopai/KneePipeline/data'
target_path = '/dataNAS/people/anoopai/KneePipeline/MeanBone_mesh_B-only.vtk'
source_path = '/dataNAS/people/anoopai/KneePipeline/data/10-P/VISIT-1/clat/results_nsm/femur_mesh_NSM_orig.vtk'
save_path_reg= '/dataNAS/people/anoopai/KneePipeline/data/10-P/VISIT-1/clat/results_nsm/femur_mesh_NSM_orig_template_reg.vtk'

orig_mesh = mesh.Mesh(source_path)
mean_mesh= mesh.Mesh(target_path)

print("Registering")
mean_mesh_reg = orig_mesh.non_rigidly_register(
    other_mesh= mean_mesh,
    reg_method="cpd",
    return_transformed_mesh=True,
    transfer_scalars= True
)

mean_mesh_reg = BoneMesh(mean_mesh_reg)
mean_mesh_reg.save_mesh(save_path_reg)



