def project_T2_data(mesh_femur, mesh_fc, map_path_nrrd):
    
    import sys
    import os

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
        
    nrrd_t2 = read_nrrd(map_path_nrrd, set_origin_zero=True).GetOutput()

    # apply inverse transform to the mesh (so its also at the origin)
    sitk_image = sitk.ReadImage(map_path_nrrd)
    nrrd_transformer = SitkVtkTransformer(sitk_image)
    mesh_fc.apply_transform_to_mesh(transform=nrrd_transformer.get_inverse_transform())
    mesh_femur.apply_transform_to_mesh(transform=nrrd_transformer.get_inverse_transform())

    # setup the probe that we are using to get data from the T2 file 
    line_resolution = 10000   # number of points along the line that the T2 data is sampled at
    filler = 0.0             # if no data is found, what value to fill the data with
    ray_length= 25.0          # how far to extend the ray from the surface (using negative to go inwards/towards the other side)
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
            data_probe.save_data_along_line(start_pt=start, end_pt=end)
        else:
            data_probe.append_filler()
            
    # undo the transforms from above so that the mesh is put back to its original position.
    mesh_femur.reverse_all_transforms()
    mesh_fc.reverse_all_transforms()
    
    return data_probe