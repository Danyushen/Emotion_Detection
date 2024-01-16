import torch
import torch.nn as nn
import torchvision.models as models


# Define a PyTorch model using EfficientNetV2 backbone
class EfficientNetV2Model(nn.Module):
    def __init__(self, num_classes=6):
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

    def forward(self, x: torch.Tensor)-> torch.Tensor:
        x = self.base_model(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# Create an instance of the PyTorch model
#model = EfficientNetV2Model()

# Print the summary
#print(model)
