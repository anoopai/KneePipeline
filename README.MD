### Environment Creation

- Dependencies are: 
    - DOSMA (bone_seg)
    - pymskt
    - NSM
```
# create environment
conda create -n knee_pipeline python=3.8
conda activate knee_pipeline

# install non-python dependcies
conda install anaconda::cmake
conda install conda-forge::lit

# Install NSM 
git clone https://github.com/gattia/nsm
cd nsm
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia
make requirements
pip install -e .

# to download NSM models, install huggingface API
pip install huggingface-hub


# Install DOSMA
git clone https://github.com/gattia/DOSMA
cd DOSMA
git checkout bone_seg
pip install '.[ai]'

# Install pymskt - this is superfluous becuase its already included in NSM
pip install mskt

# IF HAVING ISSUES WITH TF, INSTALL VERSION 2.11 (pip install tensorflow==2.11)
```

### Download models from huggingface

Download NSM HF Models using instructions on: https://github.com/gattia/ShapeMedKnee/
Also, instructions are below: 
```bash
huggingface-cli login
```

Once you run the above, you will then input your access token from huggingface.
You will need to have/create a hugginface account, and then can get the access
token by:
- login to huggingface
- click on user icon in top right
- click settings
- click "access tokens" on the left 
- click "Create new token" on left (topish)
- copy the token and input into the commandline prompt

```python
# in python script: 
from huggingface_hub import snapshot_download
snapshot_download(repo_id="aagatti/ShapeMedKnee", local_dir='./NSM_MODELS')
```
That will download the NSM models into the current directory in a folder
called: `NSM_MODELS`. This will contain both the neural shape model itself
and it will also contain a function and parameters needed to compute a BScore
from the fitted NSM latent vector. 

Download the DOSMA weights so you can perform the segmentation:

```python
# in python script: 
from huggingface_hub import snapshot_download
snapshot_download(repo_id="aagatti/dosma_bones", local_dir='./DOSMA_WEIGHTS')
```

### Update the config file to point to the above files

Update the `config_template.json` so that it includes the paths to your
downloaded files - RESAVE IT AS `config.json`:
    - models/acl_qdess_bone_july_2024: Update to path where `"Goyal_Bone_Cart_July_2024_best_model.h5` saved
    - nsm/path_model_config: update to path of the nsm `model_config.json`
    - nsm/path_model_state: update to path of the nsm model state (`2000.pth`)
    - bscore/path_model_folder: update to folder where bscore json and .py script saved 


### Run script
To run script: 
```bash
python /path/to/dosma_knee_seg.py <path_to_qdess_image> <path_to_save_results>
```

### Results

All results will be saved in <path_to_save_results>. This will include
- segmentations in .nii.gz and .nrrd format
- results.csv and results.json files with the thickness and T2 results by region
- meshes for the bones and cartilage: 
    - femur, tibia, patella, fem cart, med tib cart (0), lat tib cart (1), pat cart
- femur meshes pre-processed for the NSM:
    - femur_mesh_NSM_orig.vtk
    - fem_cart_mesh_NSM_orig.vtk
- Results from the NSM analysis:
    - reconstructed meshes using NSM:
        - NSM_recon_femur_mesh_NSM_orig.vtk
        - NSM_recon_fem_cart_mesh_NSM_orig.vtk
    - NSM fitted parameters in NSM_recon_params.json
        - NSM fitted latent vector (length 512)
        - BScore based on this latent vector, and pre-training on whole OAI
        - registration parameters used to align femur in NSM space
        - assd metrics for the error in the reconstructed bone/cart surfaces. 




If doing work in notebooks to test then follow below: 

Do work in notebook, need to start job on node and connect to node to run notebook. 
```
ON SERVER: 
salloc -c 4 --gres=gpu:2080ti:1 --mem=24gb --time=1-00

LOCALLY: 
ssh -L 1523:amalfi:22 aagatti@torino
ssh -L 1525:siena:22 aagatti@torino
ssh -L 1524:vigata:22 aagatti@torino

# connect new vscode remotely to the appropriate server (amalfi/vigata/siena)

```