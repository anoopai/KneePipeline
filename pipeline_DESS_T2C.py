from math import log
from pathlib import Path
import os
import subprocess
import json
import sys
import time
import shutil

# Constants and configurations
knee_to_use = 'aclr'
# dir_path = '/dataNAS/people/anoopai/DESS_ACL_study'
dir_path = '/dataNAS/people/anoopai/KneePipeline/'
data_path = os.path.join(dir_path, 'data')
log_path = '/dataNAS/people/anoopai/KneePipeline/logs'
output_file = os.path.join(log_path, f'NSM_T2C_{knee_to_use}.txt')
log_file_path = os.path.join(log_path, f'pipeline_DESS_errors.txt')

if os.path.exists(output_file):
    os.remove(output_file)
    
if os.path.exists(log_file_path):
    os.remove(log_file_path)

def call_pipeline_script(script, path_image, path_save, path_save_t2c, log_file_path):
    command = [
        'python',
        script,  
        path_image,
        path_save,
        path_save_t2c,
        log_file_path
    ]
    
    # Start the process with Popen
    process = subprocess.run(command,text=True)

# Clear the output file if it exists or create a new one
with open(output_file, 'w') as f:
    f.write('')

# List to keep track of the dictionaries
analysis_complete = []

# Log function for both console and log file
def log_message(message):
    print(message)
    with open(output_file, 'a') as logfile:
        logfile.write(message + '\n')

for subject in os.listdir(data_path):
    subject_path = os.path.join(data_path, subject)

    if os.path.isdir(subject_path):
        # Iterate through each visit in the subject path
        for visit in os.listdir(subject_path):
            if visit not in ['VISIT-1', 'VISIT-6']:  # Exclude VISIT-6
                visit_path = os.path.join(subject_path, visit)

                if os.path.isdir(visit_path):
                    # Iterate through each knee type in the visit path
                    for knee in os.listdir(visit_path):
                        knee_path = os.path.join(visit_path, knee)

                        if os.path.isdir(knee_path) and knee == knee_to_use:
                            path_image = os.path.join(knee_path, 'scans/qdess')
                            path_save = os.path.join(knee_path, f'results_nsm')
                            path_save_t2c = os.path.join(knee_path, f'results_t2c_new')
                            
                            # Ensure path_save directory exists
                            if not os.path.exists(path_save):
                                os.makedirs(path_save)
                            
                            # Split the path into components
                            sub_component = Path(knee_path).parts[6]  # '11-P'
                            visit_component = Path(knee_path).parts[7]  # 'VISIT-1'
                            knee_component = Path(knee_path).parts[8]  # 'clat'

                            # if os.path.exists(path_image) and not os.listdir(path_save):
                            if os.path.exists(path_image):
                                analysis_info = {}
                                analysis_info = {
                                    'sub': sub_component,
                                    'visit': visit_component,
                                    'knee': knee_component
                                }

                                print(f"Performing analysis on {analysis_info}")
                                
                                # Start tracking time
                                start_time = time.time()

                                try: 
                                    # Call the pipeline script
                                    call_pipeline_script(
                                        script='dosma_knee_seg_Dess.py',
                                        path_image=path_image,
                                        path_save=path_save,
                                        path_save_t2c = path_save_t2c,
                                        log_file_path = log_file_path
                                    )
                                
                                    # Time tracking
                                    end_time = time.time()
                                    total_time = end_time - start_time
                                    minutes = int(total_time // 60)
                                    seconds = total_time % 60
                                    print(f'Total time taken: {minutes} minutes and {seconds:.2f} seconds')
                                    
                                    analysis_info['time'] = f'{minutes} mins and {seconds:.2f} secs'
                                    
                                    with open(output_file, 'a') as f:
                                        f.write('\n Completed' + str(analysis_info))
                                    
                                except Exception as e:
                                    with open(log_file_path, 'a') as f:
                                        f.write(f"Error processing {analysis_info}: {e}\n")