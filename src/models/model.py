import timm
import torch
import torch.nn as nn


# Define a PyTorch model using EfficientNetV2 backbone
class EfficientNetV2Model(nn.Module):

    def __init__(self, num_classes=6):
        super().__init__()

        # load pretrained resolution net model from timm as a backbone
        self.base_model = timm.create_model('tf_efficientnet_b0.aa_in1k', pretrained=True)
        num_features = self.base_model.classifier.in_features
        
        # modify the last layer to fit the number of classes
        self.base_model.classifier = nn.Linear(num_features, num_features)

    def forward(self, x: torch.Tensor)-> torch.Tensor:
        return self.base_model(x)


if __name__ == '__main__':

    criterion = nn.CrossEntropyLoss()

    # test the model
    model = EfficientNetV2Model()

    # load a sample image and label batch
    x = torch.randn(10, 3, 90, 90)
    y = torch.randint(0, 6, (10,))
    
    # forward pass
    preds = model(x)

    # calculate loss
    loss = criterion(preds, y)
    print(loss)


