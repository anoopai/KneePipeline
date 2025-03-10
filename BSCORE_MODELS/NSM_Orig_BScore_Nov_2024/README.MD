

# if getting a new Bscore using already saved results that include
# the old Bscore (Logistig regression based) and the saved latents

folder_new_bscore = '/dataNAS/people/aagatti/projects/nsm_femur/ShapeMedKnee_BScore/NSM_Orig_BScore_Nov_2024/'
# add to path
import sys
sys.path.append(folder_new_bscore)
from Bscore import Bscore

# folder to results
folder_results = 'PATH_TO_SUBJECT RESULTS

json_path = os.path.join(folder_results, 'NSM_recon_params.json')

with open(json_path, 'r') as f:
    dict_ = json.load(f)
    latent = dict_['latent']
    new_bscore = Bscore(latent)