import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import alexnet

class AlexNet2(nn.Module):

    def __init__(self):
        super(AlexNet2, self).__init__()
        features = list(alexnet(pretrained=True).features)
        self.features = nn.ModuleList(features).eval()
        self.features.classifier[-1] = nn.Linear(4096,1)

    def forward(self,x):
        x = x.permute(3,1,2,0)
        x = x.repeat(3,1,1,1)
        x = x.permute(3,1,2,0)
        x = x.permute(0,3,1,2)

 

def classic_alex():
    model = torchvision.models.alexnet(pretrained=True)
    model.classifier[6] = nn.Linear(4096,1) # change the last layer to have only two classes
    return model