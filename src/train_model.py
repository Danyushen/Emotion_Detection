import hydra
import logging
import torch
import matplotlib
import wandb
import matplotlib.pyplot as plt
import pytorch_lightning as pl


from pathlib import Path
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers.wandb import WandbLogger

from models.model import EfficientNetV2Model

log = logging.getLogger(__name__)
base_dir = Path(__file__).parent.parent

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
matplotlib.use("Agg")  # no UI backend

# comment for now
# wandb.init(project="efficientnetv2", entity="efficientnetv2")

@hydra.main(config_path="..", config_name="config.yaml", version_base="1.3.2")
def main(config):

    torch.manual_seed(config.base_settings.seed)

    # load data
    train_dataset = torch.load(base_dir / config.paths.train_dataset)
    test_dataset = torch.load(base_dir / config.paths.test_dataset)

    # create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=config.hyperparameters.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config.hyperparameters.batch_size, shuffle=False)

    # initialize model
    model = EfficientNetV2Model(num_classes=config.hyperparameters.num_classes)

    # initialize callbacks
    checkpoint_callback = ModelCheckpoint(dirpath="./models", monitor="val_loss", mode="min")
    early_stopping_callback = EarlyStopping(monitor="val_loss", patience=3, verbose=True, mode="min")

    # initialize trainer
    trainer = pl.Trainer(
        devices=config.trainer.devices,
        max_epochs=config.trainer.max_epochs,
        callbacks=[checkpoint_callback, early_stopping_callback],
        # logger=WandbLogger(),
    )

    trainer.fit(model, train_dataloader, test_dataloader)

if __name__ == '__main__':
    main()