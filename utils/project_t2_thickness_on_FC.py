def project_t2_thickness_on_FC(map_path, seg_path, seg_path_subregions):
     
    import os
    from pymskt.mesh import Mesh
    import SimpleITK as sitk
    import numpy as np

    from pymskt.image import read_nrrd
    from pymskt.mesh.meshTransform import SitkVtkTransformer
    from pymskt.mesh.meshTools import ProbeVtkImageDataAlongLine
    from pymskt.mesh.meshTools import get_surface_normals, n2l, l2n
    from pymskt.mesh.utils import is_hit, get_intersect, get_surface_normals, get_obb_surface
    
    map_img = sitk.ReadImage(map_path)
    seg_img = sitk.ReadImage(seg_path)
    seg_img_labels = sitk.ReadImage(seg_path_subregions)
    seg_img_labels_array = sitk.GetArray()
    cart_subregion_labels = [11, 12, 13, 14, 15]
    
    
    mesh = Mesh(path_seg_image=seg_path, label_idx=2)
    mesh.create_mesh(smooth_image_var=0.3)

    # resample surface will reduce the number of points in the mesh and separate them equally over the surface. 
    mesh.resample_surface(clusters=10000)
    
    # read the t2 image data in vtk. - set origin to zero so that its at the origin
    # it doesnt account for rotations (now) and so easiest to align with the mesh
    # by undoing its translation, and then undoing rotation & translation for the mesh
    nrrd_t2 = read_nrrd(map_path, set_origin_zero=True).GetOutput()
    nrrd_labels = read_nrrd(seg_path, set_origin_zero=True).GetOutput()

    # apply inverse transform to the mesh (so its also at the origin)
    nrrd_transformer = SitkVtkTransformer(map_img)
    mesh.apply_transform_to_mesh(transform=nrrd_transformer.get_inverse_transform())

    # setup the probe that we are using to get data from the T2 file 
    line_resolution = 10000   # number of points along the line that the T2 data is sampled at
    filler = 0              # if no data is found, what value to fill the data with
    ray_length= -10          # how far to extend the ray from the surface (using negative to go inwards/towards the other side)
    percent_ray_length_opposite_direction = 1.0  # extend the other way a % of the line to make sure get both edges. 1.0 = 100%|

    thickness_data = []
    label_data = []
    
    data_probe = ProbeVtkImageDataAlongLine(
        line_resolution,
        nrrd_t2,
        save_mean=True,         # if we want mean. 
        save_max=True,          # if we want max
        save_std=False,         # if we want to see variation in the data along the line. 
        save_most_common=False, # for segmentations - to show the regions on the surface. 
        filler=filler
    )
    
    label_data_probe = ProbeVtkImageDataAlongLine(
    line_resolution,
    nrrd_labels,
    save_most_common=True,
    filler=0.0,
    data_categorical=True
    )

    # get the points and normals from the mesh - this is what we'll iterate over to apply the probe to. 
    points = mesh.mesh.GetPoints()
    normals = get_surface_normals(mesh.mesh)
    point_normals = normals.GetOutput().GetPointData().GetNormals()

    # create an bounding box that we can query for intersections.
    obb_cartilage = get_obb_surface(mesh.mesh)

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
            data_probe.save_data_along_line(start_pt=start,
                                            end_pt=end)
            thickness_data.append(np.sqrt(np.sum(np.square(l2n(points_intersect[0]) - l2n(points_intersect[1])))))
            label_data_probe.save_data_along_line(
                start_pt=points_intersect[0], end_pt=points_intersect[1])
        else:
            data_probe.append_filler()
            thickness_data.append(0.0)
            label_data.append(0.0)

    # undo the transforms from above so that the mesh is put back to its original position.
    mesh.reverse_all_transforms()

    # mesh.set_scalar('T2C_max', data_probe.max_data)
    mesh.set_scalar('T2_mean (ms)', data_probe.mean_data)
    mesh.set_scalar('T2_std (ms)', data_probe.std_data)
    mesh.set_scalar('T2_max (ms)', data_probe.max_data)
    mesh.set_scalar('Subregion label', label_data_probe.most_common_data)
    mesh.set_scalar('Thickness (mm)', thickness_data)
    
    save_path = map_path.replace('.nii.gz', '.vtk')
    mesh.save_mesh(save_path)