import sys
from pathlib import Path
import pytest
from torch.utils.data import TensorDataset, DataLoader
import torch
from omegaconf import OmegaConf
from unittest.mock import patch

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.train_model import main as train_main

def test_train_initialization(mocker):
    """
    Test the initialization of the training components.
    """
    # Mock Wandb and internal function calls
    mocker.patch('src.train_model.wandb.init')
    mocker.patch('pytorch_lightning.Trainer')  # Mock PyTorch Lightning Trainer

    # Mocking model and DataLoader with more realistic behavior
    mocker.patch('src.train_model.EfficientNetV2Model')  # Mock your model
    mock_dataset = TensorDataset(torch.randn(10, 3, 224, 224), torch.randn(10))  # Mock dataset with 10 samples
    mocker.patch('torch.load', return_value=mock_dataset)  # Mock torch.load to return the mock dataset


    # Create a mock configuration using OmegaConf
    mock_config = OmegaConf.create({
        'base_settings': {'seed': 42},
        'wandb': {'project': 'test_project'},
        'hyperparameters': {'learning_rate': 0.001, 'batch_size': 32, 'num_classes': 10},
        'paths': {'train_dataset': 'data/processed/train_dataset.pt', 'test_dataset': 'data/processed/test_dataset.pt'},
        'trainer': {'devices': 1, 'max_epochs': 1},
    })

    train_main(mock_config)  # Test initialization with the mock configuration
