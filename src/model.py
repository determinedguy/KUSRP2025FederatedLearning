from torch import nn
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models
import torch
from options import args_parser
args = args_parser()


class CNN(nn.Module):
    def __init__(self, args):
        super(CNN, self).__init__()
        self.args = args
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()


        # Dummy forward pass to calculate the flattened size
        self._to_linear = None
        self._init_linear_layer()

        self.fc1 = nn.Linear(self._to_linear, 50)
        self.fc2 = nn.Linear(50, args.num_classes)
    
    def _init_linear_layer(self):
        # Dummy forward pass to calculate the flattened size
        x = torch.randn(1, self.args.num_channels, 224, 224)
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2_drop(self.conv2(x))), 2)
        self._to_linear = x.view(1, -1).shape[1]


    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    
def get_model(model_name, num_classes):
    """
    Returns a model instance based on the model name.
    Supports fine-tuning for pretrained models.

    Args:
        model_name (str): One of ['cnn', 'resnet18', 'resnet50', 'mobilenet_v2']
        num_classes (int): Number of output classes

    Returns:
        nn.Module: the model
    """
    if model_name == 'cnn':
        return CNN(args)  # mevcut basit CNN modelin
    elif model_name == 'resnet18':
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    elif model_name == 'shufflenet_v2':
        model = models.shufflenet_v2_x0_5(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    else:
        raise NotImplementedError(f"Model '{model_name}' is not supported.")