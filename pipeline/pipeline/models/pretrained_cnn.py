import pretrainedmodels
import torch
from torch import nn
import torch.nn.functional as F

from .linear_block import LinearBlock


class PretrainedCNN(nn.Module):
    def __init__(self, model_name='se_resnext101_32x4d',
                 in_channels=3, out_dim=10, hdim=512, activation=F.leaky_relu,
                 use_bn=True, pretrained='imagenet', kernel_size=3, stride=1, padding=1):
        super(PretrainedCNN, self).__init__()
        self.conv0 = nn.Conv2d(in_channels, 3,
                               kernel_size=kernel_size, stride=stride, padding=padding,
                               bias=True)
        self.base_model = pretrainedmodels.__dict__[model_name](pretrained=pretrained)
        inch = self.base_model.last_linear.in_features
        self.lin1 = LinearBlock(inch, hdim, use_bn=use_bn, activation=activation, residual=False)
        self.lin2 = LinearBlock(hdim, out_dim, use_bn=use_bn, activation=None, residual=False)

    def forward(self, x):
        h = self.conv0(x)
        h = self.base_model.features(h)
        h = torch.sum(h, dim=(-1, -2))
        h = self.lin1(h)
        h = self.lin2(h)
        return h
