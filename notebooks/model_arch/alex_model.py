import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import alexnet

class AlexNet2(nn.Module):

    def __init__(self):
        super(AlexNet2, self).__init__()
        self.layers = alexnet(pretrained=True)
        self.layers.classifier[-1] = nn.Linear(4096,1)
        for i,layers in enumerate(self.layers.features):
            if "MaxPool2d" in str(layers):
                self.layers.features[i] = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

    def forward(self,x):
        x = x.permute(3,1,2,0)
        x = x.repeat(3,1,1,1)
        x = x.permute(3,0,1,2)
        for feat in self.layers.features:
            x = feat(x)
        x = self.layers.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.layers.classifier(x)
        return F.sigmoid(x)
