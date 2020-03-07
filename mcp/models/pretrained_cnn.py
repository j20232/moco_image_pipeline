from torch import nn
import torch.nn.functional as F

import os
import sys

DIR_NAME = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(DIR_NAME + "/local_cnn_finetune/"))
sys.path.append(os.path.join(DIR_NAME + "/local_timm/"))

import cnn_finetune
import timm


def get_official_names():
    return [
        # torchvision
        "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
        "resnext50_32x4d", "resnext101_32x8d",
        "densenet121", "densenet169", "densenet201", "densenet161",
        "mobilenet_v2",
        "shufflenet_v2_x0_5", "shufflenet_v2_x1_0",
    ]


def get_pretrained_names():
    return [
        "resnext101_64x4d",
        "nasnetalarge",
        "nasnetamobile",
        "dpn68", "dpn68b", "dpn92", "dpn98", "dpn131", "dpn107",
        "xception",
        "senet154", "se_resnet50", "se_resnet101", "se_resnet152", "se_resnext50_32x4d", "se_resnext101_32x4d",
        "pnasnet5large",
    ]


def get_timm_names():
    return [
        "mnasnet_050", "mnasnet_075", "mnasnet_100", "mnasnet_140",
        "semnasnet_050", "semnasnet_075", "semnasnet_100", "semnasnet_140",
        "mnasnet_small",
        "mobilenetv2_100",
        "fbnetc_100",
        "spnasnet_100",
        "efficientnet_b0",
        "efficientnet_b1",
        "efficientnet_b2", "efficientnet_b2a",
        "efficientnet_b3", "efficientnet_b3a",
        "efficientnet_b4", "efficientnet_b5", "efficientnet_b6", "efficientnet_b7", "efficientnet_b8",
        "efficientnet_es", "efficientnet_em", "efficientnet_el",
        "efficientnet_cc_b0_4e", "efficientnet_cc_b0_8e", "efficientnet_cc_b1_8e",
        "mixnet_s", "mixnet_m", "mixnet_l", "mixnet_xl", "mixnet_xxl"
    ]


class PretrainedCNN(nn.Module):
    def __init__(self, model_name,
                 in_channels=3, out_dim=10, hdim=512, activation=F.leaky_relu,
                 use_bn=True, pretrained=False, kernel_size=3, stride=1, padding=1):
        super(PretrainedCNN, self).__init__()
        print("Architecture: ", model_name)
        is_remote = os.getcwd() != "/kaggle/working"
        if model_name in get_official_names() or model_name in get_pretrained_names():
            self.base_model = cnn_finetune.make_model(model_name, num_classes=out_dim,
                                                      pretrained=is_remote and pretrained)
        elif model_name in get_timm_names():
            self.base_model = timm.create_model(model_name, num_classes=out_dim,
                                                pretrained=is_remote and pretrained)
        else:
            print("Not supported architecture")
            assert False

        self.conv0 = nn.Conv2d(in_channels, 3,
                               kernel_size=kernel_size, stride=stride, padding=padding,
                               bias=True)

    def forward(self, x):
        return self.base_model(self.conv0(x))
