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
def predict(model: str, dataset: str) -> None:
    """ Predict on test dataset """

    model, dataset = torch.load(model).to(DEVICE), torch.load(dataset)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    return torch.cat([model(x) for x, y in loader], 0)

cli.add_command(predict)

if __name__ == '__main__':
    cli()