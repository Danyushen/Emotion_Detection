import timm
import torch
import torch.nn as nn
import pytorch_lightning as pl

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader as Dataloader

class EfficientNetV2Model(pl.LightningModule):

    def __init__(self, num_classes=6, lr=1e-3):
        super().__init__()

        # define metrics and lr
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()

        # load pretrained resolution net model from timm as a backbone
        self.base_model = timm.create_model('tf_efficientnet_b0.aa_in1k', pretrained=True)
        num_features = self.base_model.classifier.in_features
        
        # modify the last layer to fit the number of classes
        self.base_model.classifier = nn.Linear(num_features, num_features)

    def forward(self, x: torch.Tensor)-> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f'Expected input to a 4D tensor but got {x.ndim}D tensor instead.')

        return self.base_model(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def common_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        return self.criterion(y_hat, y)

    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log_dict({'train_loss': loss}, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log_dict({'val_loss': loss}, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def test_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log_dict({'test_loss': loss}, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, _ = batch
        y_hat = self(x)
        return y_hat       

if __name__ == '__main__':

    dataset = TensorDataset(torch.randn(10, 3, 90, 90), torch.randint(0, 6, (10,)))
    train_dataloader = Dataloader(dataset, batch_size=1, shuffle=True)
    test_dataloader = Dataloader(dataset, batch_size=1, shuffle=False)

    # test the model
    model = EfficientNetV2Model()

    # train model
    trainer = pl.Trainer(devices=1, accelerator='cpu', max_epochs=10)
    trainer.fit(model, train_dataloader, test_dataloader)

    # example image
    x = torch.randn(1, 3, 90, 90)

    # predict class
    y_hat = model(x)
    print(y_hat)
    print(y_hat.shape)