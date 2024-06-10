import subprocess
import os

# Get the path to the directory containing the script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Change the working directory to the outermost folder
os.chdir(os.path.join(script_dir, '..'))

script_files = ['openings/similar.py','posts/recommendation.py',
                'projects/similar.py','projects/recommendation.py', 
                'topics.py'
            ]

for file in script_files:
    script_path = os.path.join('scripts', file)
    subprocess.run(['python3', script_path])
