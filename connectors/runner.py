import subprocess
import os

# Get the path to the directory containing the script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Change the working directory to the outermost folder
os.chdir(os.path.join(script_dir, '..'))

script_files = ['openings.py', 'projects.py', 'posts.py' ]

for file in script_files:
    script_path = os.path.join('connectors', file)
    subprocess.run(['python3', script_path])
