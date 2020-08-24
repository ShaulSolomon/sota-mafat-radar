

import torch
import torch.nn as nn
import torch.nn.functional as F


class SNR_base_model(nn.Module):
    '''
    Architecture with concatenation of external data in the form of SNR type value
    In this model we have 2 inputs to the forward pass

    '''
    def __init__(self):
        super(SNR_base_model, self).__init__()
        self.cn1 = nn.Conv2d(1,16,3)
        self.cn2 = nn.Conv2d(16,32,3)
        self.fc1 = nn.Linear(5761,128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32,1)

        self.maxpool = nn.MaxPool2d((2,2))

    def forward(self, x1, x2):
        batch_size = x1.shape[0]
        x = x1.permute(0,3,1,2)
        x = self.maxpool(F.relu(self.cn1(x)))
        x = self.maxpool(F.relu(self.cn2(x)))
        x= torch.flatten(x).reshape(batch_size,-1)
        x = torch.cat((x),(x2),0)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))


def init_weights(m):
    if (type(m) == nn.Linear) | (type(m) == nn.Conv2d):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)