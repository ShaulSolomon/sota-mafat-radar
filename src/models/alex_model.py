import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import alexnet


'''
alex_mdf_model(
  (arch): AlexNet(
    (features): Sequential(
      (0): Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
      (1): ReLU(inplace=True)
      (2): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
      (3): Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
      (4): ReLU(inplace=True)
      (5): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
      (6): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (7): ReLU(inplace=True)
      (8): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (9): ReLU(inplace=True)
      (10): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (11): ReLU(inplace=True)
      (12): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    )
    (avgpool): AdaptiveAvgPool2d(output_size=(6, 6))
    (classifier): Sequential(
      (0): Dropout(p=0.5, inplace=False)
      (1): Linear(in_features=9216, out_features=4096, bias=True)
      (2): ReLU(inplace=True)
      (3): Dropout(p=0.5, inplace=False)
      (4): Linear(in_features=4096, out_features=4096, bias=True)
      (5): ReLU(inplace=True)
      (6): Linear(in_features=4096, out_features=1, bias=True)
    )
  )
)
'''

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
        #x = x.repeat(3,1,1,1)
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



