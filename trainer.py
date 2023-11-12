import subprocess
import os

script_folders = ['connectors', 'scripts']
script_files = ['runner.py']

for folder in script_folders:
    for file in script_files:
        try:
            print(f"Executing file {file} in {folder}")
            script_path = os.path.join(folder, file)
            subprocess.run(['python3', script_path])
        except Exception as e:
            print(f"Error executing file {file} in {folder}: {e}")
