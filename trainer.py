import subprocess
import os

script_folders = ['connectors', 'scripts']
script_files = ['runner.py']

def logger(level , title, description, path):
    subprocess.run(['python3', 'api_logger.py', level, title, description, path], cwd='utils')

logger("info","Training Initiated", "","trainer.py")

for folder in script_folders:
    for file in script_files:
        try:
            script_path = os.path.join(folder, file)
            subprocess.run(['python3', script_path])
        except Exception as e:
            logger("error",f"Error Executing file {file} in {folder}", e, "trainer.py")
