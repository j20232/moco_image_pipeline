from torch import nn
import torch.nn.functional as F
import os
import sys

DIR_NAME = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(DIR_NAME + "/local_cnn_finetune/"))

import cnn_finetune
from cnn_finetune.local_pretrainedmodels.pretrainedmodels.models.senet import SEResNeXtBottleneck, SEModule


def initialize_weights(module):
    if type(module) == SEModule or type(module) == SEResNeXtBottleneck or type(module) == nn.Sequential:
        for child in module.children():
            initialize_weights(child)
    elif "reset_parameters" in dir(module):
        module.reset_parameters()


class FreezedSEResNeXt(nn.Module):
    def __init__(self, model_name,
                 in_channels=3, out_dim=10, hdim=512, activation=F.leaky_relu,
                 use_bn=True, pretrained=False, kernel_size=3, stride=1, padding=1):
        super(FreezedSEResNeXt, self).__init__()
        print("Architecture: ", model_name)
        is_remote = os.getcwd() != "/kaggle/working"
        self.base_model = cnn_finetune.make_model("se_resnext50_32x4d",
                                                  num_classes=out_dim,
                                                  pretrained=is_remote and pretrained)
        self.conv0 = nn.Conv2d(in_channels, 3,
                               kernel_size=kernel_size, stride=stride, padding=padding,
                               bias=True)
        for child in self.base_model.children():
            for cnt, c in enumerate(child.children()):
                if cnt == 0:
                    initialize_weights(c)
                else:
                    for p in c.parameters():
                        p.requires_grad = False

    def forward(self, x):
        return self.base_model(self.conv0(x))
