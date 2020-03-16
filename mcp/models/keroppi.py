# Reference: https://www.kaggle.com/c/bengaliai-cv19/discussion/123757

import torch
import torch.nn as nn
import torch.nn.functional as F


class KeroSEResNeXt(nn.Module):
    def __init__(self, in_channels=1, out_dim=10):
        super(KeroSEResNeXt, self).__init__()

        self.block0 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.block1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, padding=0, stride=2, ceil_mode=True),
            SENextBottleneckBlock(64, 128, 256, stride=1, is_shortcut=True, excite_size=64),
            * [SENextBottleneckBlock(256, 128, 256, stride=1, is_shortcut=False, excite_size=64) for i in range(1, 3)],
        )

        self.block2 = nn.Sequential(
            SENextBottleneckBlock(256, 256, 512, stride=2, is_shortcut=True, excite_size=32),
            * [SENextBottleneckBlock(512, 256, 512, stride=1, is_shortcut=False, excite_size=32) for i in range(1, 4)],
        )

        self.block3 = nn.Sequential(
            SENextBottleneckBlock(512, 512, 1024, stride=2, is_shortcut=True, excite_size=16),
            * [SENextBottleneckBlock(1024, 512, 1024, stride=1, is_shortcut=False, excite_size=16) for i in range(1, 6)],
        )

        self.block4 = nn.Sequential(
            SENextBottleneckBlock(1024, 1024, 2048, stride=2, is_shortcut=True, excite_size=8),
            * [SENextBottleneckBlock(2048, 1024, 2048, stride=1, is_shortcut=False, excite_size=8) for i in range(1, 3)],
        )

        self.dropout = nn.Dropout(p=0.2)
        self.logit = nn.Linear(2048, out_dim)

    def forward(self, x):
        batch_size = len(x)

        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch_size, -1)
        x = self.dropout(x)
        logit = self.logit(x)
        return logit


class SENextBottleneckBlock(nn.Module):
    def __init__(self, in_channel, channel, out_channel, stride=1, group=32, reduction=16, excite_size=-1, is_shortcut=False):
        super(SENextBottleneckBlock, self).__init__()
        self.is_shortcut = is_shortcut

        self.conv_bn1 = ConvBn2d(in_channel, channel, kernel_size=1, padding=0, stride=1)
        self.conv_bn2 = ConvBn2d(channel, channel, kernel_size=3, padding=1, stride=stride, groups=group)
        self.conv_bn3 = ConvBn2d(channel, out_channel, kernel_size=1, padding=0, stride=1)
        self.scale = SqueezeExcite(out_channel, reduction, excite_size)

        if is_shortcut:
            self.shortcut = ConvBn2d(in_channel, out_channel, kernel_size=1, padding=0, stride=stride)

    def forward(self, x):
        z = F.relu(self.conv_bn1(x), inplace=True)
        z = F.relu(self.conv_bn2(z), inplace=True)
        z = self.scale(self.conv_bn3(z))
        z += self.shortcut(x) if self.is_shortcut else x
        z = F.relu(z, inplace=True)
        return z


class ConvBn2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1, stride=1, groups=1, is_bn=True):
        super(ConvBn2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-5)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class SqueezeExcite(nn.Module):
    def __init__(self, in_channel, reduction=4, excite_size=-1):
        super(SqueezeExcite, self).__init__()
        self.excite_size = excite_size
        self.fc1 = nn.Conv2d(in_channel, in_channel // reduction, kernel_size=1, padding=0)
        self.fc2 = nn.Conv2d(in_channel // reduction, in_channel, kernel_size=1, padding=0)

    def forward(self, x):
        s = F.adaptive_avg_pool2d(x, 1)
        s = self.fc1(s)
        s = F.relu(s, inplace=True)
        s = self.fc2(s)
        x = x * torch.sigmoid(s)
        return x
