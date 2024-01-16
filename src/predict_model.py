import torch
import hydra
import logging
import click
import pytorch_lightning as pl

from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

log = logging.getLogger(__name__)
base_dir = Path(__file__).parent.parent

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@click.group()
def cli():
    pass

@click.command()
@click.option('--model', type=str, help='Path to model checkpoint')
@click.option('--dataset', type=str, help='Path to dataset which will be used for prediction')
def predict(model: torch.nn.Module, dataset: TensorDataset) -> None:
    """ Predict on test dataset """
    return torch.cat([model(batch) for batch in DataLoader(dataset)], 0)