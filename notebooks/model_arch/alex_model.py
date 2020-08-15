import torch
import torch.nn as nn
import torch.nn.functional as F

def classic_alex():
    model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)
    model.classifier[6] = nn.Linear(4096,2) # change the last layer to have only two classes
    return model