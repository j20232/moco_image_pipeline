import importlib
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as official_models
from .local_pretrained_models import pretrainedmodels as local_pretrained_models
from .local_timm import timm as local_timm

from .linear_block import LinearBlock


class PretrainedCNN(nn.Module):
    def __init__(self, model_name='se_resnext101_32x4d', is_local=False,
                 in_channels=3, out_dim=10, hdim=512, activation=F.leaky_relu,
                 use_bn=True, pretrained=True, kernel_size=3, stride=1, padding=1):
        super(PretrainedCNN, self).__init__()
        print("Architecture: ", model_name)
        module = local_pretrained_models if is_local else importlib.import_module("pretrainedmodels")
        if model_name in dir(official_models):
            # official
            self.base_model = getattr(official_models, model_name)(pretrained=pretrained and not is_local)
            feature_layers = list(self.base_model.children())[:-1]
            final_layer = list(self.base_model.children())[-1]
            self.base_model = nn.Sequential(*feature_layers, nn.Linear(final_layer.in_features, out_dim))
        elif model_name in dir(module):
            # pretrainedmodels
            pre = "imagenet" if not is_local and pretrained else None
            self.base_models = module.__dict__[model_name](pretrained=pre)
            inch = self.base_models.last_linear.in_features
            base_model = self.base_models.features
            lin1 = LinearBlock(inch, hdim, use_bn=use_bn, activation=activation, residual=False)
            lin2 = LinearBlock(hdim, out_dim, use_bn=use_bn, activation=None, residual=False)
            layers = [base_model, lin1, lin2]
            self.base_model = nn.Sequential(*layers)
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
