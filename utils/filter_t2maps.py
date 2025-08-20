def filter_t2maps(t2_map_path, fwhm, t2_map_save_path):

    '''
    Filter the quantitative maps with a Gaussian filter.
    :param qmap: quantitative map
    :param fwhm: full width at half maximum (FWHM) in mm
    :return: filtered quantitative map
    '''

    import nibabel as nib
    import numpy as np
    import SimpleITK as sitk
    from scipy.ndimage import gaussian_filter
    

    # Get the image data as a numpy array
    qmap= sitk.ReadImage(t2_map_path)
    data = sitk.GetArrayFromImage(qmap)
    
    data[data>=80] = 80
    data[data<=0] = 0

    # Define the voxel sizes in mm
    voxel_sizes = np.array(qmap.GetSpacing())

    # Calculate the sigma value for the Gaussian filter
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0))) / voxel_sizes

    # Apply the Gaussian filter to the image data
    filtered_data = gaussian_filter(data, sigma)

    # Create a new NIfTI image with the filtered data
    filtered_img = sitk.GetImageFromArray(filtered_data)
    filtered_img.CopyInformation(qmap)
    
    # Save the filtered image
    sitk.WriteImage(filtered_img, t2_map_save_path)
    t2_map_save_path_nii = t2_map_save_path.replace('.nrrd', '.nii.gz')
    sitk.WriteImage(filtered_img, t2_map_save_path_nii)
    
    return filtered_img