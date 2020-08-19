import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import alexnet

class alex_mdf_model(nn.Module):

    def __init__(self):
        super(alex_mdf_model, self).__init__()
        self.arch = alexnet(pretrained=True)
        self.arch.classifier[-1] = nn.Linear(4096,1)

        for i,layer in enumerate(self.arch.features.children()):
            if "MaxPool" in str(layer):
                self.arch.features[i] = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

    def forward(self,x):
        x = x.permute(3,1,2,0)
        x = x.repeat(3,1,1,1)
        x = x.permute(3,0,1,2)
        x = self.arch(x)
        return torch.sigmoid(x)