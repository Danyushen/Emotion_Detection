import torch
import torchmetrics

import torchvision.models as models
from pytorch_lightning import LightningModule
from torch import nn
from torch.optim import Adam

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define a PyTorch model using EfficientNetV2 backbone
class EfficientNetV2Model(LightningModule):
    def __init__(self, num_classes=6, lr=1e-3, class_weights=None):
        super(EfficientNetV2Model, self).__init__()

        self.base_model = models.efficientnet_v2_m(num_classes=1000, weights='DEFAULT')
        # Remove the classification head, leave only the backbone
        self.base_model.classifier = nn.Identity()

        # Freeze the weights of the backbone
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Add custom layers for classification
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(1280, num_classes)  # Assuming EfficientNetV2 backbone with 1280 output channels

        self.lr = lr
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)


    def forward(self, x: torch.Tensor)-> torch.Tensor:
        x = self.base_model(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x
    
    def training_step(self, batch, batch_idx: int):
        x, y = batch
        x, y = x.to(device), y.to(device)
        y_pred = self.forward(x)

        loss = self.criterion(y_pred, y)
        acc = self.accuracy(x, y)

        self.log_dict({"train_loss": loss, "train_acc": acc}, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def predict_step(self, batch, batch_idx: int , dataloader_idx: int = None):
        return self.forward(batch)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)