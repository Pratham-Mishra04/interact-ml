import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

# Get the path to the directory containing the script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Change the working directory to the outermost folder
os.chdir(os.path.join(script_dir, ".."))

script_files = [
    "openings/similar.py",
    "posts/recommendation.py",
    "projects/similar.py",
    "projects/recommendation.py",
    "topics.py",
]


def run_script(script):
    script_path = os.path.join("scripts", script)
    try:
        subprocess.run(["python3", script_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running {script}: {e}")


# Execute scripts in parallel
with ThreadPoolExecutor() as executor:
    executor.map(run_script, script_files)
