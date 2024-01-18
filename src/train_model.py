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
#print(os.getcwd())

#hej = torch.zeros(10, 10)
#torch.save(hej, "hej.pt")

from src.models.model import EfficientNetV2Model
#from models.model import EfficientNetV2Model

log = logging.getLogger(__name__)
base_dir = Path(__file__).parent.parent

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
matplotlib.use("Agg")  # no UI backend


@hydra.main(config_path="..", config_name="config.yaml", version_base="1.3.2")
def main(config):
    print(os.getcwd())
    hej2 = torch.zeros(10, 10)
    torch.save(hej2, "hej2.pt") 

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

    # load data
    train_dataset = torch.load(base_dir / config.paths.train_dataset)
    test_dataset = torch.load(base_dir / config.paths.test_dataset)

    # create dataloaders
    batch_size = config.hyperparameters.batch_size
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # initialize model
    model = EfficientNetV2Model(num_classes=config.hyperparameters.num_classes, lr=config.hyperparameters.learning_rate)

    # initialize callbacks
    checkpoint_callback = ModelCheckpoint(dirpath="src/models/checkpoints", monitor="val_loss", mode="min")
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
