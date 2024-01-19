import hydra
import logging
import torch
import matplotlib
import os
import wandb
import requests
import sys
from pathlib import Path
# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers.wandb import WandbLogger
from src.models.model import EfficientNetV2Model


# Set the API key
os.environ["WANDB_API_KEY"] = "3a8227d16fffba40e5a4f21fbe96329c602fac69"

# Now initialize wandb
wandb.init(project="emotion_detection")

def running_in_google_cloud():
    try:
        metadata_url = "http://metadata.google.internal"
        headers = {"Metadata-Flavor": "Google"}
        response = requests.get(metadata_url, headers=headers, timeout=2)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

# Check if running in Google Cloud and set environment variable
if running_in_google_cloud():
    os.environ['RUNNING_IN_CLOUD'] = '1'


log = logging.getLogger(__name__)
base_dir = Path(__file__).parent.parent

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
matplotlib.use("Agg")  # no UI backend

@hydra.main(config_path="../conf", config_name="config.yaml", version_base="1.3.2")
def main(config):

    torch.manual_seed(config.base_settings.seed)

    # initialize wandb
    wandb.init(
        project=config.wandb.project,
        config={
            "learning_rate": config.hyperparameters.learning_rate,
            "batch_size": config.hyperparameters.batch_size,
            "architecture": "EfficientNetV2",
            "epochs": config.trainer.max_epochs,
            "seed": config.base_settings.seed,
        },
    )

    # Check environment and set data paths
    data_path = '/gcs/data_tensors/data/processed/' if os.getenv('RUNNING_IN_CLOUD') else 'data/processed/'


    # Load datasets
    train_dataset = torch.load(os.path.join(data_path, 'train_dataset.pt'))
    test_dataset = torch.load(os.path.join(data_path, 'test_dataset.pt'))

    # Create dataloaders
    batch_size = config.hyperparameters.batch_size
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # initialize model
    model = EfficientNetV2Model(num_classes=config.hyperparameters.num_classes, lr=config.hyperparameters.learning_rate)
    
    # Checkpoint directory
    checkpoint_dir = "gs://data_tensors" if os.getenv('RUNNING_IN_CLOUD') else os.getcwd()

    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_dir, monitor="val_loss", mode="min")
    early_stopping_callback = EarlyStopping(monitor="val_loss", patience=3, verbose=True, mode="min")

    # initialize trainer
    trainer = pl.Trainer(
        devices=config.trainer.devices,
        max_epochs=config.trainer.max_epochs,
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=WandbLogger(),
    )

    trainer.fit(model, train_dataloader, test_dataloader)

    wandb.finish()


if __name__ == "__main__":
    main()