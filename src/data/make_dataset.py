
import torch
import csv
import torchvision.transforms.functional as FT
import albumentations as A
import numpy as np

from torch.utils.data import random_split
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset
from PIL import Image
from pathlib import Path
import subprocess

# this globals should be in config file later
base_dir = Path(__file__).parent.parent.parent

# define paths
DATASET_META = base_dir / "data/raw/data.csv"
DATASET_DIR = base_dir / "data/raw/dataset"

PROCESSED_TRAIN_DATASET = base_dir / "data/processed/train_dataset.pt"
PROCESSED_TEST_DATASET = base_dir / "data/processed/test_dataset.pt"

# define class labels
CLASSES = ['ahegao', 'angry', 'happy', 'neutral', 'sad', 'surprise']
LABELS = { label : i for label, i in zip(CLASSES, range(len(CLASSES))) }

TRAIN_SPLIT = 0.8


TANSFORM_PIPELINE = A.Compose([
    A.Resize(width=90, height=90)
])

class ImgTransformer:
    """ Custom image transformer class """
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def __call__(self, img: Image):
        img = self.pipeline(image=np.array(img))['image']
        transformed_img = Image.fromarray(img)

        return FT.to_tensor(transformed_img)

class EmotionDataset(Dataset):
    """ EmotionDataset custom wrapper class """
    def __init__(self, dir=DATASET_DIR, transformer=None):
        self.dir = dir
        self.transformer = transformer
        self.images = {}

        with open(DATASET_META, newline='') as f:
            reader = csv.reader(f, delimiter=',')
            next(reader) # skip first row with column labels

            # read each row and parse into dict {idx : { path, label }}
            for idx, path, label in reader:
                self.images[int(idx)] = { "path" : str(path), "label" : str(label).lower() }

            self.len = len(self.images)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        img, label = Image.open(f"{self.dir}/{self.images[idx]['path']}"), self.images[idx]['label']

        # use transformer if added
        if self.transformer:
            return self.transformer(img), torch.tensor([LABELS[label]])

        return img, LABELS[label]

def get_split_ratio(n_data, split=0.8):
    """ Get n_data split ratio """
    n_train = int(n_data * split)
    n_test = n_data - n_train

    # sanity check
    assert n_train + n_test == n_data

    return n_train, n_test


def save_tensor_dataset(dataset: Dataset, path: str):
    """ Save given dataset as TensorDataset by given file path """

    path.parent.mkdir(parents=True, exist_ok=True)
    data, labels = [], []

    for i in range(len(dataset)):
        x, y = dataset[i]

        data.append(x.unsqueeze(0))
        labels.append(y)

    torch.save(TensorDataset(
        torch.cat(data, dim=0),
        torch.cat(labels, dim=0)
        ), path)

def pull_data_from_dvc():
    """ Pull data from DVC remote if it doesn't exist locally """
    if not DATASET_META.exists() or not DATASET_DIR.exists():
        # Run DVC pull command
        subprocess.run(["dvc", "pull", str(DATASET_META), str(DATASET_DIR)], check=True)

def save_and_add_to_dvc(dataset: Dataset, path: Path):
    """ Save given dataset as TensorDataset and add to DVC """
    save_tensor_dataset(dataset, path)
    
    # Add to DVC and push to remote
    subprocess.run(["dvc", "add", str(path)], check=True)
    subprocess.run(["dvc", "push"], check=True)

if __name__ == '__main__':

    # Pull data from DVC if not available locally
    pull_data_from_dvc()

    transformer = ImgTransformer(pipeline=TANSFORM_PIPELINE)
    dataset = EmotionDataset(dir=DATASET_DIR, transformer=transformer)

    # split dataset to train, test
    train_set, test_test = random_split(dataset, get_split_ratio(len(dataset), TRAIN_SPLIT))

    # Save both datasets into processed folder and add to DVC
    save_and_add_to_dvc(train_set, PROCESSED_TRAIN_DATASET)
    save_and_add_to_dvc(test_test, PROCESSED_TEST_DATASET)
