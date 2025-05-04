import numpy as np
import os
import json

PATH_JSON = os.path.join(os.path.dirname(__file__), 'model.json')

def Bscore(latents, path_json=PATH_JSON):
    """
    Function to compute Bscore from latents
    
    Args:
        latents: np.array
            Latent representation of the data
        path_json: str
            Path to the json file containing the model
    
    Returns:
        bscore: np.array
            Bscore of the data
    """
    if isinstance(latents, (list, tuple)):
        latents = np.array(latents)
        
    with open(path_json, 'r') as f:
        model = json.load(f)
    
    bscore_vector = np.array(model['bscore_vector'])
    mean_healthy = np.array(model['mean_healthy'])
    std_healthy = np.array(model['std_healthy'])
    
    # project data onto coeffs
    projection = latents @ bscore_vector.T
    
    # standardize data
    bscore = (projection - mean_healthy) / std_healthy
    
    return bscore