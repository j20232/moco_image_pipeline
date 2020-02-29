import importlib
from torch import nn
import torch.nn.functional as F
from .local_cnn_finetune import cnn_finetune as local_cnn_finetune
from .local_timm import timm as local_timm


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
        "resnext101_32x4d", "resnext101_64x4d",
        "nasnetalarge",
        "nasnetamobile",
        "inceptionresnetv2",
        "dpn68", "dpn68b", "dpn92", "dpn98", "dpn131", "dpn107",
        "inception_v4",
        "xception",
        "senet154", "se_resnet50", "se_resnet101", "se_resnet152", "se_resnext50_32x4d", "se_resnext101_32x4d",
        "pnasnet5large",
        "polynet"
    ]


class PretrainedCNN(nn.Module):
    def __init__(self, model_name='se_resnext101_32x4d', is_local=False,
                 in_channels=3, out_dim=10, hdim=512, activation=F.leaky_relu,
                 use_bn=True, pretrained=True, kernel_size=3, stride=1, padding=1):
        super(PretrainedCNN, self).__init__()
        print("Architecture: ", model_name)
        module = local_cnn_finetune if is_local else importlib.import_module("cnn_finetune")
        if model_name in get_official_names() or model_name in get_pretrained_names():
            self.base_model = module.make_model(model_name, num_classes=out_dim,
                                                pretrained=not is_local and pretrained)
        else:
            # timm
            module = local_timm if is_local else importlib.import_module("timm")
            self.base_model = module.create_model(model_name, num_classes=out_dim,
                                                  pretrained=not is_local and pretrained)
        self.conv0 = nn.Conv2d(in_channels, 3,
                               kernel_size=kernel_size, stride=stride, padding=padding,
                               bias=True)

    def forward(self, x):
        return self.base_model(self.conv0(x))
