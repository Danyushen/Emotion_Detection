from models.model import EfficientNetV2Model
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from pathlib import Path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# this globals should be in config file later
base_dir = Path(__file__).parent.parent

# Define hyperparameters
num_classes = 6
lr = 0.0001
batch_size = 64
seed = 42
torch.manual_seed(seed)
#subset_size = 3000
num_epochs = 5

# Load the processed dataset
train_dataset = torch.load( base_dir /"data/processed/train_dataset.pt")

def get_class_weights(dataset):
    """ Calculate class weights based on label distribution in the dataset"""
    class_counts = torch.zeros(num_classes)
    for _, label in dataset:
        class_counts[label] += 1

    class_weights = 1 / class_counts

    # Normalize the weights
    class_weights /= class_weights.sum()
    
    return class_weights

def random_subset(dataset, subset_size):
    """ Create a random subset of the dataset"""
    indices = torch.randperm(len(dataset))[:subset_size]
    subset = Subset(dataset, indices)
    
    return subset

def normalize(tensor):
    """ Normalize a tensor"""
    return (tensor - tensor.mean()) / tensor.std()

def train(dataset, batch_size, num_classes, lr, class_weights, num_epochs):
    """ Train the model"""
    train_loader = DataLoader(dataset, batch_size=batch_size)

    model = EfficientNetV2Model(num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch} Loss {loss}")

    torch.save(model, f"model_all_data_epoch_{num_epochs}_lr_{lr}.pt")

if __name__ == '__main__':
    class_weights = get_class_weights(train_dataset)
    print(class_weights)
    #train_subset = random_subset(train_dataset, subset_size)
    train(train_dataset, batch_size, num_classes, lr, class_weights, num_epochs)