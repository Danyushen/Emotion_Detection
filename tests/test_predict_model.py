import torch
from torch.utils.data import DataLoader, TensorDataset

import sys
from pathlib import Path
# Add the project root to the Python path
print(sys.path)
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
print(sys.path)

from src.models.model import EfficientNetV2Model

# Function to correct the module path
def custom_map_location(storage, loc):
    return storage

def test_model_prediction():
    # instantiate your model
    model = EfficientNetV2Model()
    model.eval()  # evaluation mode

    # create dummy input data
    dummy_input = torch.randn(1, 3, 224, 224)  # Example for an image input

    # perform a prediction
    with torch.no_grad():
        predictions = model(dummy_input)

    # check prediction output
    assert predictions is not None, "Model did not return predictions."
    assert isinstance(predictions, torch.Tensor), "Predictions should be a torch.Tensor."

def test_model_with_varied_input_sizes():
    # instantiate your model
    model = EfficientNetV2Model() 
    model.eval()  # evaluation mode

    # test with different batch sizes
    for batch_size in [1, 5, 10]:  # xxample batch sizes
        # create dummy input data with varying batch sizes
        dummy_input = torch.randn(batch_size, 3, 224, 224) 

        # perform a prediction
        with torch.no_grad():
            predictions = model(dummy_input)

        # check prediction output
        assert predictions is not None, "Model did not return predictions for batch size {}".format(batch_size)
        assert isinstance(predictions, torch.Tensor), "Predictions should be a torch.Tensor for batch size {}".format(batch_size)
        assert predictions.shape[0] == batch_size, "Output batch"

def test_model_with_real_data():
    # load model
    def map_location(storage, loc):
        if hasattr(storage, '_set_from_file'):
            return storage
        return custom_map_location(storage, loc)
    
    model_path = 'model.pt'
    model =  torch.load(model_path, map_location=map_location)
    model.eval()

    # load the test dataset
    dataset_path = 'data/processed/test_dataset.pt'
    test_dataset = torch.load(dataset_path)

    testloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # iterate over the test data and perform predictions
    for images, labels in testloader:
        with torch.no_grad():
            predictions = model(images)

        # check prediction output
        assert predictions is not None, "Model did not return predictions."
        assert isinstance(predictions, torch.Tensor), "Predictions should be a torch.Tensor."

        break  # break after one batch to reduce test duration

# import torch
# from src.models.model import EfficientNetV2Model
# from torch.utils.data import DataLoader

# def test_model_with_real_data():
#     model_path = 'path/to/model_state_dict.pt'  # Update the path as needed

#     # Create a new model instance
#     model = EfficientNetV2Model(num_classes=6)  # Assuming 6 classes
#     model.load_state_dict(torch.load(model_path))
#     model.eval()

#     # Load your test dataset
#     test_dataset = torch.load('path/to/test_dataset.pt')
#     testloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

#     # Perform a prediction on test data
#     with torch.no_grad():
#         for images, labels in testloader:
#             predictions = model(images)
#             assert predictions is not None, "Model did not return predictions."
#             assert isinstance(predictions, torch.Tensor), "Predictions should be a torch.Tensor."
#             break  # Example: stop after one batch
