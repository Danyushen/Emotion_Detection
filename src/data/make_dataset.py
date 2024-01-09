
import torch
import csv
import torchvision.transforms.functional as FT

from torch.utils.data import random_split
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset
from PIL import Image
from pathlib import Path

# this globals should be in config file later
base_dir = Path(__file__).parent.parent.parent

# define paths
DATASET_META = base_dir / "data/data.csv"
DATASET_DIR = base_dir / "data/raw/dataset"

PROCESSED_TRAIN_DATASET = base_dir / "data/processed/train_dataset.pt"
PROCESSED_TEST_DATASET = base_dir / "data/processed/test_dataset.pt"

# define class labels
CLASSES = ['ahegao', 'angry', 'happy', 'neutral', 'sad', 'surprise']
LABELS = { label : i for label, i in zip(CLASSES, range(len(CLASSES))) }

TRAIN_SPLIT = 0.8
IMG_RES = (90, 90)

class ImgTransformer:
    """ Custom image transformer class 

    Can be later integrated with Albumentation    
    
    """
    def __init__(self):
        pass

    def __call__(self, img: Image):
        return FT.to_tensor(img.resize(IMG_RES, Image.BICUBIC))
    

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
    data, labels = [], []

    for i in range(len(dataset)):
        x, y = dataset[i]

        data.append(x.unsqueeze(0))
        labels.append(y)

    torch.save(TensorDataset(
        torch.cat(data, dim=0), 
        torch.cat(labels, dim=0)
        ), path)


if __name__ == '__main__':
    transformer = ImgTransformer()
    dataset = EmotionDataset(dir=DATASET_DIR, transformer=transformer)

    # split dataset to train, test
    train_set, test_test = random_split(dataset, get_split_ratio(len(dataset), TRAIN_SPLIT))

    # save both datasets into processed folder
    save_tensor_dataset(train_set, PROCESSED_TRAIN_DATASET)
    save_tensor_dataset(test_test, PROCESSED_TEST_DATASET)