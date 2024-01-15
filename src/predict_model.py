import torch
from torch.utils.data import DataLoader
from pathlib import Path
import hydra
import logging

@hydra.main(config_path="..", config_name="config.yml")
def main(cfg):
    # Load base settings and hyperparameters from config file into variables
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = Path(__file__).parent.parent
    test_dataset_dir = cfg.paths.test_dataset
    model_checkpoint_dir = cfg.paths.model_checkpoint

    batch_size = cfg.hyperparameters.batch_size

    # Create Logger
    log = logging.getLogger(__name__)

    # Load model checkpoint
    model = torch.load(base_dir / model_checkpoint_dir)
    log.info(f'Model checkpoint loaded: {model_checkpoint_dir}')

    # Load test set
    test_dataset = torch.load(base_dir / test_dataset_dir)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Run prediction
    predict(model, test_loader, device, log)

def predict(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device,
    log
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
            # print(f"predicted: {predicted} actual: {y}")
        accuracy = total_correct / total_samples

        log.info(f"Validation Accuracy: {accuracy * 100:.2f}%")

    predictions = torch.cat([model(x) for x, _ in dataloader], 0)
    log.info(f'Number of predictions: {predictions.shape[0]}, Number of labels: {predictions.shape[1]}')

    return predictions

if __name__ == '__main__':
    main()