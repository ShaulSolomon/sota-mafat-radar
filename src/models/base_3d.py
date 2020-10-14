import torch
import torch.nn as nn
import torch.nn.functional as F


class base3d(nn.Module):
    def __init__(self):
        super(base3d, self).__init__()

        self.conv_layer1 = self._conv_layer_set(1, 16)
        self.conv_layer2 = self._conv_layer_set(16, 32)
        self.fc1 = nn.Linear(2 ** 3 * 32, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 1)

        self.relu = nn.LeakyReLU()
        self.batch = nn.BatchNorm1d(128)
        self.drop = nn.Dropout(p=0.15)

    def _conv_layer_set(self, in_c, out_c):
        conv_layer = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 3), padding=0),
            nn.LeakyReLU(),
            nn.MaxPool3d((2, 2, 2)),
        )
        return conv_layer

    def forward(self, x):
        # Set 1
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = out.view(out.size(0), -1)
        out = self.drop(self.batch(self.relu(self.fc1(out))))
        out = self.drop(self.batch(self.relu(self.fc2(out))))
        return torch.sigmoid(self.fc3(out))


class alex_3d(nn.Module):
    def __init__(self):
        super(alex_3d, self).__init__()

        self.conv_layer1 = self._conv_layer_set(1, 32, (3, 3, 3), (3, 3, 3), pad=2)
        self.conv_layer2 = self._conv_layer_set(32, 128, (3, 3, 1), (2, 2, 1), pad=(2, 2, 2))
        self.conv_layer3 = self._conv_layer_set(128, 256, pool=(2, 2, 1), pad=(1, 1, 0))
        self.conv_layer4 = self._conv_layer_set(256, 128, pool=(2, 2, 1), pad=(1, 1, 0))
        self.conv_layer5 = self._conv_layer_set(128, 128, pool=(1, 1, 1), pad=(1, 1, 0))

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=640, out_features=1024, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=1024, out_features=256, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=256, out_features=1, bias=True)
        )

    def _conv_layer_set(self, in_c, out_c, kernel=(3, 3, 3), pool=(2, 2, 2), pad=0):
        conv_layer = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=kernel, padding=pad),
            nn.LeakyReLU(),
            nn.MaxPool3d(pool),
        )
        return conv_layer

    def forward(self, x):
        # Set 1
        #x = x.permute(0,1, 3, 2, 4) # permutation from expected input dimensions of (batch_size, channels, short_time, long_time, scale)
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = self.conv_layer3(out)
        out = self.conv_layer4(out)
        out = self.conv_layer5(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return torch.sigmoid(out)
