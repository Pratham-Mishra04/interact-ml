import subprocess
import os
import logging

logging.basicConfig(filename='cron.log', level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filemode='a')

script_folders = ['connectors', 'scripts']
script_files = ['runner.py']

logging.info(f"-------------TRAINING INITIATED-------------")

for folder in script_folders:
    for file in script_files:
        try:
            logging.info(f"Executing file {file} in {folder}")
            script_path = os.path.join(folder, file)
            logging.info(script_path)
            subprocess.run(['python3', script_path])
            logging.info(f"Finished Executing file {file} in {folder}")
        except Exception as e:
            logging.error(f"Error Executing file {file} in {folder}: {e}")
