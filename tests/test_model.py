import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import pytest
import torch
import os

from src.models.model import EfficientNetV2Model

def test_model_input_output_shape():
    """
    Test if the model accepts the correct input shape and produces the expected output shape.
    """
    # Create a dummy input tensor of the correct shape
    # Assuming the input is a 3-channel image of size 224x224
    dummy_input = torch.randn(1, 3, 224, 224)  # Batch size of 1

    # Instantiate the model
    model = EfficientNetV2Model()

    # Forward pass through the model
    output = model(dummy_input)

    # Check if the output shape matches the expected shape
    assert output.shape == (1, 6)  # Adjust the shape based on your actual output shape