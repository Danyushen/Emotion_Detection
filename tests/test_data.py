import sys
from pathlib import Path
import torch
import os

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# tests/test_data.py
import pytest
from tempfile import TemporaryDirectory

def test_data_loading():
    # depending on dataset
    N_train = 12_362
    N_test = 3_091
    with TemporaryDirectory() as tmp_dir:
        train_dataset = torch.load(os.path.join(project_root, "data/processed/train_dataset.pt"))
        test_dataset = torch.load(os.path.join(project_root, "data/processed/test_dataset.pt"))

        assert len(train_dataset) == N_train, "Incorrect number of samples in training set"
        assert len(test_dataset) == N_test, "Incorrect number of samples in test set"
        # Add more assertions as needed
