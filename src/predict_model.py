import torch
from torch.utils.data import DataLoader
from pathlib import Path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# this globals should be in config file later
base_dir = Path(__file__).parent.parent

model = torch.load(base_dir /"model_all_data_epoch_10_lr_0.001.pt")
test_dataset = torch.load(base_dir /"data/processed/test_dataset.pt")
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


def predict(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader
) -> None:
    """Run prediction for a given model and dataloader.
    
    Args:
        model: model to use for prediction
        dataloader: dataloader with batches
    
    Returns
        Tensor of shape [N, d] where N is the number of samples and d is the output dimension of the model

    """
    model = model.eval()
    with torch.no_grad():
        total_correct = 0
        total_samples = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            _, predicted = torch.max(y_pred, 1)
            total_samples += y.size(0)
            total_correct += (predicted == y).sum().item()
            #print(f"predicted: {predicted} actual: {y}")
        accuracy = total_correct / total_samples
    
        print(f"Validation Accuracy: {accuracy * 100:.2f}%")
        

    predictions = torch.cat([model(x) for x, _ in dataloader], 0)
    

    return predictions

predictions = predict(model, test_loader)
print(predictions.shape)

   

    