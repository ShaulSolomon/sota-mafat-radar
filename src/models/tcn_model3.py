import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F


class Chomp2d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp2d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding = -1, dropout=0.2):
        super(TemporalBlock, self).__init__()

        padding = (kernel_size-1) * dilation if padding == -1 else padding
        n_outputs = n_outputs[0] if type(n_outputs) == list else n_outputs

        self.conv1 = weight_norm(nn.Conv2d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=(kernel_size//2,padding), dilation=(1,dilation)))
        self.chomp1 = Chomp2d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv2d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=(kernel_size//2,padding), dilation=(1,dilation)))
        self.chomp2 = Chomp2d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)

        self.downsample = nn.Conv2d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.singlechannel = nn.Conv2d(num_channels[-1],1,1)
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.Linear(126*32, 1)

    def forward(self, x):
        x = x.permute(0,3,1,2)
        x = self.network(x)
        x = self.dropout(self.singlechannel(x)).flatten(start_dim=1)
        return F.sigmoid(self.decoder(x))



# class TestTCN(nn.Module):
#     def __init__(self, num_inputs, num_channels, kernel_size):
#         super(TestTCN, self).__init__()
#         n_inputs = num_inputs
#         n_outputs = num_channels[0]
#         dilation_size1 = 2 ** 2
#         self.padding = (kernel_size-1) * dilation_size1
#         self.conv1 = nn.Conv2d(n_inputs, n_outputs, kernel_size,
#                                     stride=1, padding=(kernel_size//2,self.padding), dilation=(1,dilation_size1))
#         self.chomp1 = Chomp2d(self.padding)
#         self.relu1 = nn.ReLU()

#         n_inputs2 = n_outputs
#         self.conv2 = nn.Conv2d(n_inputs2, n_outputs, kernel_size,
#                             stride=1, padding=(kernel_size//2,self.padding), dilation=(1,dilation_size1))
#         self.chomp2 = Chomp2d(self.padding)
#         self.relu2 = nn.ReLU()

#         self.downsample = nn.Conv2d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None


        
#     def forward(self, x):
#         print(x.shape)
#         out = self.conv1(x)
#         print(out.shape)
#         out = self.chomp1(out)
#         print(out.shape)
#         out = self.relu1(out)
#         # out = self.relu2(self.chomp2(self.conv2(out)))
#         res = x if self.downsample is None else self.downsample(x)
#         return (out + res)