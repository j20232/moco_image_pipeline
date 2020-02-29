import unittest

import torch
from mcp.models import PretrainedCNN, get_official_names, get_pretrained_names


class ModelTest(unittest.TestCase):

    """
    def test_official_cnn(self):
        t = torch.ones([1, 1, 64, 64])
        names = get_official_names()
        for name in names:
            model = PretrainedCNN(name, in_channels=1, out_dim=10,
                                  pretrained=False, is_local=True)
            model.eval()
            out = model(t)
            self.assertEqual(out.shape[1], 10)
    """

    def test_pretrained_cnn(self):
        t = torch.ones([1, 1, 64, 64])
        names = get_pretrained_names()
        for name in names:
            model = PretrainedCNN(name, in_channels=1, out_dim=10,
                                  pretrained=False, is_local=True)
            model.eval()
            out = model(t)
            self.assertEqual(out.shape[1], 10)
