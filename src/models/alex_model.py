import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import alexnet

class alex_mdf_model(nn.Module):

    def __init__(self):
        super(alex_mdf_model, self).__init__()
        self.arch = alexnet(pretrained=False)
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



class alex_mdf_s_model(nn.Module):

    def __init__(self):
        super(alex_mdf_s_model, self).__init__()
        self.features = nn.Sequential(
                            nn.Conv2d(1,32,kernel_size=(7,7),stride=(2,2), padding=(2,2)),
                            nn.ReLU(inplace=True),
                            nn.MaxPool2d(kernel_size=3,stride=2,padding=1,dilation=1,ceil_mode=False),
                            nn.Conv2d(32,128,kernel_size=(5,5), stride=(1,1),padding=(2,2)),
                            nn.ReLU(inplace=True),
                            nn.MaxPool2d(kernel_size=3,stride=2,padding=1,dilation=1,ceil_mode=False),
                            nn.Conv2d(128,256,kernel_size=(3,3),stride=(2,2), padding=(2,2)),
                            nn.ReLU(inplace=True),
                            nn.MaxPool2d(kernel_size=3,stride=2,padding=1,dilation=1,ceil_mode=False),
                            nn.Conv2d(256,128,kernel_size=(3,3),stride=(2,2), padding=(2,2)),
                            nn.ReLU(inplace=True),
                            nn.MaxPool2d(kernel_size=3,stride=2,padding=1,dilation=1,ceil_mode=False),
                            nn.Conv2d(128,128,kernel_size=(3,3),stride=(2,2), padding=(2,2)),
                            nn.ReLU(inplace=True),
                            nn.MaxPool2d(kernel_size=3,stride=2,padding=1,dilation=1,ceil_mode=False),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(6,6))
        self.classifier = nn.Sequential(
                            nn.Dropout(p=0.5, inplace=False),
                            nn.Linear(in_features=4608, out_features=4096, bias=True),
                            nn.ReLU(inplace=True),
                            nn.Dropout(p=0.5, inplace=False),
                            nn.Linear(in_features=4096, out_features=4096, bias=True),
                            nn.ReLU(inplace=True),
                            nn.Linear(in_features=4096, out_features=1, bias=True)   
        )

    def forward(self,x):
        x = x.permute(0,3,1,2)
        x = self.features(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.classifier(x)
        return torch.sigmoid(x)



