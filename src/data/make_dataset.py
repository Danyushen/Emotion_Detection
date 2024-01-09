
from typing import Any
import torch
import csv
import torchvision.transforms.functional as FT

from torch.utils.data import random_split
from torch.utils.data import Dataset
from PIL import Image

# this globals should be in config file later
DATA_DIR = "data/raw"

LABELS = {
    'ahegao' : 0,
    'angry' : 1,
    'happy' : 2,
    'neutral' : 3,
    'sad' : 4,
    'suprise' : 5
}

SPLIT = 0.2

class ImgTransformer:
    def __init__(self):
        pass

    def __call__(self, img):
        return FT.to_tensor(img)

class EmotionDataset(Dataset):
    """ EmotionDataset custom wrapper class """
    def __init__(self, dir="data/raw", transformer=None):
        self.dir = dir
        self.transformer = transformer

        # read csv metadata, skip first row and lower label
        with open(f"{self.dir}/data.csv", newline='') as f:
            reader = csv.reader(f, delimiter=',')
            next(reader)

            self.images = {int(idx): {"path" : str(path), "label" : str(label).lower()} for (idx, path, label) in reader}
            self.len = len(self.images)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        img, label = Image.open(f"{self.dir}/{self.images[idx]['path']}"), self.images[idx]['label']

        if self.transformer:
            return self.transformer(img), LABELS[label]
        
        return img, LABELS[label]


if __name__ == '__main__':

    transformer = ImgTransformer()
    dataset = EmotionDataset(dir=DATA_DIR, transformer=transformer)


    pass    