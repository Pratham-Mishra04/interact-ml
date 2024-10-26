import subprocess
import os
from concurrent.futures import ProcessPoolExecutor

# Get the path to the directory containing the script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Change the working directory to the outermost folder
os.chdir(os.path.join(script_dir, '..'))

script_files = ['openings/similar.py',
                'posts/recommendation.py',
                'projects/similar.py',
                'projects/recommendation.py', 
                'topics.py']

def run_script(script_path):
    result = subprocess.run(['python3', script_path], capture_output=True, text=True)
    return result

with ProcessPoolExecutor() as executor:
    script_paths = [os.path.join('scripts', file) for file in script_files]

    futures = {executor.submit(run_script, path): path for path in script_paths}

    for future in futures:
        script_path = futures[future]
        try:
            result = future.result()
            print(f"Script {script_path} completed with return code {result.returncode}")
            if result.stdout:
                print(f"Output:\n{result.stdout}")
            if result.stderr:
                print(f"Error:\n{result.stderr}")
        except Exception as e:
            print(f"Script {script_path} failed with error: {e}")