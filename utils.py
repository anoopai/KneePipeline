import numpy as np


def clip_femur_top(mesh_orig):
    """
    Clips the top portion of a femur mesh based on its dimensions.
    
    Args:
        mesh_orig: The original mesh to be clipped
        
    Returns:
        The clipped mesh in its original position
    """
    # Calculate the centroid of the mesh
    centroid = np.mean(mesh_orig.points, axis=0)

    # Translate the mesh to center it at the origin (0, 0, 0)
    mesh_orig.point_coords -= centroid

    # Get the bounds and calculate dimensions
    bounds = mesh_orig.bounds
    dimension_x = bounds[1] - bounds[0]
    dimension_y = bounds[3] - bounds[2]
    dimension_z = bounds[5] - bounds[4]

    # Determine clipping height based on dimensions
    if dimension_z > 0.7 * dimension_y:  # if S-I dimension > 70% of R-L dimension
        # if SI > 0.7 * ML then clip to 0.7 * ML
        z_max = bounds[4] + 0.7 * abs(bounds[1] - bounds[0])
    else:  # crop to 95% of current S-I height
        # if SI < 0.7 * ML, then... we don't have enough height to clip
        # so we clip to 95% of the current S-I height
        z_max = bounds[4] + 0.95 * abs(bounds[5] - bounds[4])

    # Clip along z-axis
    mesh_orig.clip("z", value=z_max, invert=True, inplace=True)

    # Translate the mesh back to its original position
    mesh_orig.point_coords += centroid
    return mesh_orig