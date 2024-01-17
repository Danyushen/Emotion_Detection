import sys
from pathlib import Path
import torch
import os

# tests/test_data.py
from tempfile import TemporaryDirectory

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


def test_data_loading():
    # depending on dataset
    N_train = 12_362
    N_test = 3_091

    with TemporaryDirectory():
        train_dataset = torch.load(os.path.join(project_root, "data/processed/train_dataset.pt"))
        test_dataset = torch.load(os.path.join(project_root, "data/processed/test_dataset.pt"))

        assert len(train_dataset) == N_train, "Incorrect number of samples in training set"
        assert len(test_dataset) == N_test, "Incorrect number of samples in test set"
        # Add more assertions as needed
