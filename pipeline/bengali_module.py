import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl

from pipeline.models import PretrainedCNN
from config.bengali_config import get_cfg


class BengaliModule(pl.LightningModule):

    def __init__(self, index):
        super(BengaliModule, self).__init__()
        self.cfg = get_cfg()
        n_total = self.cfg.N_GRAPHEME + self.cfg.N_VOWEL + self.cfg.N_CONSONANT
        print('n_total', n_total)

        model_name = self.cfg.MODEL

        # Set pretrained='imagenet' to download imagenet pretrained model...
        self.predictor = PretrainedCNN(in_channels=1, out_dim=n_total,
                                       model_name=model_name, pretrained=None)

    def forward(step, x):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass
        
    def validation_end(self, outputs):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def test_end(self, outputs):
        pass

    def configure_optimizers(self):
        pass

    @pl.data_loader
    def train_dataloader(self):
        pass

    @pl.data_loader
    def val_dataloader(self):
        pass

    @pl.data_loader
    def test_dataloader(self):
        pass

