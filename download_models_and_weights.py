# in python script: 
from huggingface_hub import snapshot_download
import os
os.chdir('./models_and_weights')
snapshot_download(repo_id="aagatti/ShapeMedKnee", local_dir='./NSM_models')

# in python script: 
from huggingface_hub import snapshot_download
import os
# os.chdir('./models_and_weights')
snapshot_download(repo_id="aagatti/dosma_bones", local_dir='dosma_weights')