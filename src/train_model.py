import hydra
import logging
import torch
import matplotlib
import wandb
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


@hydra.main(config_path="../conf", config_name="config.yaml", version_base="1.3.2")
def main(config):
    trainer_conf, model_conf = (
        config.trainer.trainer_cpu,
        config.model.default_model
    )

    torch.manual_seed(model_conf.seed)

    # initialize wandb
    wandb.init(
        project=model_conf.wandb_project,
        config={
            "learning_rate": model_conf.learning_rate,
            "batch_size": model_conf.batch_size,
            "architecture": "EfficientNetV2",
            "epochs": trainer_conf.max_epochs,
            "seed": model_conf.seed,
        },
    )

    # load data
    train_dataset = torch.load(base_dir / model_conf.paths.train_dataset)
    test_dataset = torch.load(base_dir / model_conf.paths.test_dataset)

    # create dataloaders
    batch_size = model_conf.batch_size
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # initialize model
    model = EfficientNetV2Model(num_classes=model_conf.num_classes, lr=model_conf.learning_rate)

    # initialize callbacks
    checkpoint_callback = ModelCheckpoint(dirpath="./models/checkpoints", monitor="val_loss", mode="min")
    early_stopping_callback = EarlyStopping(monitor="val_loss", patience=3, verbose=True, mode="min")

    # initialize trainer
    trainer = pl.Trainer(
        devices=trainer_conf.devices,
        max_epochs=trainer_conf.max_epochs,
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=WandbLogger(),
    )

    trainer.fit(model, train_dataloader, test_dataloader)

    wandb.finish()


if __name__ == "__main__":
    main()
