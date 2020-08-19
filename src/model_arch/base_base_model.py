import torch
import torch.nn as nn
import torch.nn.functional as F

class base_base_model(nn.Module):
    '''
    Architecture built identical to base cnn model given to us.
    After some googling - it seems that for:
        [x] - kernel_initializer - as its set to init its the same as the default
        [ ] - bias_regularizer - that it set in the optimizer
    '''
    def __init__(self):
        super(base_base_model, self).__init__()
        self.cn1 = nn.Conv2d(1,16,3)
        self.cn2 = nn.Conv2d(16,32,3)
        self.fc1 = nn.Linear(5760,128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32,1)

        self.maxpool = nn.MaxPool2d((2,2))

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.permute(0,3,1,2)
        x = self.maxpool(F.relu(self.cn1(x)))
        x = self.maxpool(F.relu(self.cn2(x)))
        x = torch.flatten(x).reshape(batch_size,-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))