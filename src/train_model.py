import hydra
import logging
import torch
import matplotlib
import os
import wandb

# Set the API key
os.environ["WANDB_API_KEY"] = "3a8227d16fffba40e5a4f21fbe96329c602fac69"

# Now initialize wandb
wandb.init(project="emotion_detection")

import pytorch_lightning as pl

from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers.wandb import WandbLogger

import sys
from pathlib import Path
# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.models.model import EfficientNetV2Model
#from models.model import EfficientNetV2Model

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
    if os.environ.get('RUNNING_IN_CLOUD'):
        data_path = '/gcs/data_tensors/data/processed/'
    else:
        data_path = 'data/processed/'

    # Load datasets
    train_dataset = torch.load(os.path.join(data_path, 'train_dataset.pt'))
    test_dataset = torch.load(os.path.join(data_path, 'test_dataset.pt'))

    # Create dataloaders
    batch_size = config.hyperparameters.batch_size
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # initialize model
    model = EfficientNetV2Model(num_classes=config.hyperparameters.num_classes, lr=config.hyperparameters.learning_rate)
    
    # initialize callbacks
    if os.environ.get('RUNNING_IN_CLOUD'):
        checkpoint_dir = "gs://data_tensors"
    else:
        checkpoint_dir = os.getcwd()

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

    # Manually save the model's state dictionary
    model_path = os.path.join(checkpoint_dir, 'model.pt')
    torch.save(model.state_dict(), model_path)

    wandb.finish()


if __name__ == "__main__":
    main()