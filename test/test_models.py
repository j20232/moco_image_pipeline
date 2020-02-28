import unittest

import torch
from mcp.models import PretrainedCNN


class ModelTest(unittest.TestCase):

    def test_official_implementation(self):
        t = torch.ones([1, 1, 32, 32])
        names = [
            "resnet18",
            "densenet121", "densenet161", "densenet169", "densenet201",
            "mnasnet0_5",
            "mobilenet_v2",
        ]
        for name in names:
            print(name)
            model = PretrainedCNN(name, in_channels=1, out_dim=10,
                                  pretrained=False, is_local=True)
            model.eval()
            out = model(t)
            self.assertEqual(out.shape[1], 10)
