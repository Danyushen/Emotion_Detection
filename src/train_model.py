import torch
import hydra
import time
import logging
import torch.nn as nn
import pytorch_lightning as pl

from pathlib import Path
from torch.utils.data import DataLoader, Subset
from models.model import EfficientNetV2Model
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

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
    model = EfficientNetV2Model(num_classes=num_classes, lr=lr, class_weights=class_weights)

    # checkpoint_callback = ModelCheckpoint(dirpath="./models/checkpoints", monitor="train_loss", mode="min")
    # early_stopping_callback = EarlyStopping(monitor="train_loss", patience=3, verbose=True, mode="min")

    trainer = pl.Trainer(
        devices=1,
        accelerator="cpu",
        max_epochs=num_epochs,
        # callbacks=[checkpoint_callback, early_stopping_callback],
    )
    trainer.fit(model, train_loader)

if __name__ == '__main__':
    main()