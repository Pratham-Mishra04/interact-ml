import subprocess
import os
from utils import logger

training_logger= logger.get_logger('training_logger')

# Get the path to the directory containing the script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Change the working directory to the outermost folder
os.chdir(os.path.join(script_dir, '..'))

script_files = ['openings.py', 'projects.py', 'posts.py' ]

training_logger.info("-------Running Connector Scripts-------")

for file in script_files:
    try:
        training_logger.info(f"Executing file {file}")
        script_path = os.path.join('connectors', file)
        subprocess.run(['python3', script_path])
    except Exception as e:
        training_logger.error(f"Error executing file {file}: {e}")
