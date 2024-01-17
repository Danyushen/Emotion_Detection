import sys
from pathlib import Path
import torch
import os
import tempfile

# Import DVC commands to fetch data files
import dvc.api

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


# Define DVC remote and data file paths
dvc_remote = "storage"  # Change to your DVC remote name
train_data_path = "data/processed/train_dataset.pt"
test_data_path = "data/processed/test_dataset.pt"


def test_data_loading():
    # depending on dataset
    N_train = 12_362
    N_test = 3_091

    # Create a temporary directory to store the downloaded data
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Fetch data files using DVC and save them to the temporary directory
        with dvc.api.open(train_data_path, remote=dvc_remote, mode="rb") as train_dataset:
            with open(os.path.join(tmp_dir, "train_dataset.bin"), "wb") as tmp_train_file:
                tmp_train_file.write(train_dataset.read())

        with dvc.api.open(test_data_path, remote=dvc_remote, mode="rb") as test_dataset:
            with open(os.path.join(tmp_dir, "test_dataset.bin"), "wb") as tmp_test_file:
                tmp_test_file.write(test_dataset.read())

        # Load the datasets from the temporary directory
        loaded_train_dataset = torch.load(os.path.join(tmp_dir, "train_dataset.bin"))
        loaded_test_dataset = torch.load(os.path.join(tmp_dir, "test_dataset.bin"))

        # Perform assertions on the loaded datasets
        assert len(loaded_train_dataset) == N_train, "Incorrect number of samples in training set"
        assert len(loaded_test_dataset) == N_test, "Incorrect number of samples in test set"
