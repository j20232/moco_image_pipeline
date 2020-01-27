import os

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms

import pytorch_lightning as pl
from models import PretrainedCNN


class MocoModule(pl.LightningModule):

    def __init__(self):
        super(MocoModule, self).__init__()
        n_grapheme = 168
        n_vowel = 11
        n_consonant = 7
        n_total = n_grapheme + n_vowel + n_consonant
        print('n_total', n_total)

        model_name='se_resnext50_32x4d'

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

