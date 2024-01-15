from models.model import EfficientNetV2Model
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from pathlib import Path
import hydra
import time
import logging

@hydra.main(config_path="..", config_name="config.yaml")
def main(cfg):
    # Load base settings and hyperparameters from config file into variables
    base_dir = Path(__file__).parent.parent
    train_dataset_dir = cfg.paths.train_dataset
    seed = cfg.base_settings.seed

    num_classes = cfg.hyperparameters.num_classes
    lr = cfg.hyperparameters.learning_rate
    batch_size = cfg.hyperparameters.batch_size
    num_epochs = cfg.hyperparameters.num_epochs

    # Load the processed dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    train_dataset = torch.load(base_dir / train_dataset_dir)

    # Create Logger
    log = logging.getLogger(__name__)

    # Run the functions
    class_weights = get_class_weights(dataset=train_dataset, num_classes=num_classes)
    # train_subset = random_subset(train_dataset, subset_size)
    train(train_dataset, batch_size, num_classes, lr, class_weights, num_epochs, device, base_dir, log)

def get_class_weights(dataset, num_classes):
    """ Calculate class weights based on label distribution in the dataset"""
    class_counts = torch.zeros(num_classes)
    for _, label in dataset:
        class_counts[label] += 1

    class_weights = 1 / class_counts

    # Normalize the weights
    class_weights /= class_weights.sum()

    return class_weights


def random_subset(dataset, subset_size):
    """ Create a random subset of the dataset"""
    indices = torch.randperm(len(dataset))[:subset_size]
    subset = Subset(dataset, indices)

    return subset


def normalize(tensor):
    """ Normalize a tensor"""
    return (tensor - tensor.mean()) / tensor.std()


def train(dataset, batch_size, num_classes, lr, class_weights, num_epochs, device, base_dir, log):
    """ Train the model"""
    log.info('Begin training..')
    train_loader = DataLoader(dataset, batch_size=batch_size)

    model = EfficientNetV2Model(num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    for epoch in range(num_epochs):
        log.info(f'Starting epoch {epoch}/{num_epochs}')
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
        log.info(f"Epoch: {epoch}, Loss: {loss}")

    log.info("Finished training, saving model..")

    time_str = time.strftime('%Y-%m-%d_%H%M%S')
    model_dir = f'{base_dir}/src/models/checkpoints'
    model_file = f'model_all_data_epoch_{num_epochs}_lr_{lr}_{time_str}.pt'

    torch.save(model, f'{model_dir}/{model_file}')
    log.info(f'Model has been saved as: {model_file}')

if __name__ == '__main__':
    main()