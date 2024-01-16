import subprocess
import sys
from pathlib import Path

# DVC Pull
try:
    subprocess.run(['dvc', 'pull'], check=True)
except subprocess.CalledProcessError:
    sys.exit('Failed to pull data from DVC remote')
