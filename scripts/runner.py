import subprocess
import os
from utils import logger

training_logger= logger.get_logger('training_logger')

# Get the path to the directory containing the script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Change the working directory to the outermost folder
os.chdir(os.path.join(script_dir, '..'))

script_folders = ['openings', 'projects', 'posts' ]
script_files = ['recommendation.py', 'similar.py']

training_logger.info("-------Running Training Scripts-------")

for folder in script_folders:
    for file in script_files:
        try:
            training_logger.info(f"Executing file {file} in {folder}")
            script_path = os.path.join('scripts', folder, file)
            subprocess.run(['python3', script_path])
        except Exception as e:
            training_logger.error(f"Error executing file {file} in {folder}: {e}")
