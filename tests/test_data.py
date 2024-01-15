import sys
from pathlib import Path
import torch
import os

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import DVC commands to fetch data files
import dvc.api

# Define DVC remote and data file paths
dvc_remote = "storage"  # Change to your DVC remote name
train_data_path = "data/processed/train_dataset.pt"
test_data_path = "data/processed/test_dataset.pt"

def test_data_loading():
    # depending on dataset
    N_train = 12_362
    N_test = 3_091

    # Fetch data files using DVC
    with dvc.api.open(train_data_path, remote=dvc_remote) as train_dataset, \
         dvc.api.open(test_data_path, remote=dvc_remote) as test_dataset:
         
        assert len(train_dataset) == N_train, "Incorrect number of samples in training set"
        assert len(test_dataset) == N_test, "Incorrect number of samples in test set"
        # Add more assertions as needed
