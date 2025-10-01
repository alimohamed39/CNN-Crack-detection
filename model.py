import torch
import torch.nn as nn
import torchvision.models as models

########################################################
# Importing the pre-trained ResNet-18 model
def get_model(num_classes=2):
    model = models.resnet18(pretrained=True)

    # Get the number of features in the last layer
    in_features = model.fc.in_features

    # Replace the final fully connected layer
    model.fc = nn.Linear(in_features, num_classes)

    return model  # Modified model with correct output size
