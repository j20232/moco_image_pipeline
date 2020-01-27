import os
from pathlib import Path
from yacs.config import CfgNode as CN

import torch
import torch.optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch_lightning as pl

from pipeline.models import PretrainedCNN
from pipeline.functions import accuracy


# default settings
_C = CN()
_C.N_GRAPHEME = 160
_C.N_VOWEL = 11
_C.N_CONSONANT = 7
_C.MODEL = "se_resnext50_32x4d"
_C.PRETRAINED = "null"
_C.NN = CN()
_C.NN.OPTIM_NAME = "Adam"
_C.NN.OPTIM_PARAMS = []
_C.NN.SCHEDULER_NAME = "ReduceLROnPlateau"
_C.NN.SCHEDULER_PARAMS = []


def get_cfg():
    return _C.clone()

class Bengali(pl.LightningModule):

    def __init__(self, cfg_dir, index):
        super(Bengali, self).__init__()

        # read cfg
        self.cfg = get_cfg()
        yaml_path = Path(".").resolve() / "config" / cfg_dir / f"{index}.yaml"
        self.cfg.merge_from_file(yaml_path)
        self.cfg.freeze()

        self.n_total_class = self.cfg.N_GRAPHEME + self.cfg.N_VOWEL + self.cfg.N_CONSONANT
        print('n_total', self.n_total_class)

        # Set pretrained='imagenet' to download imagenet pretrained model...
        pretrained = self.cfg.PRETRAINED
        if pretrained == "null":
            pretrained = None
        self.model = PretrainedCNN(in_channels=1, out_dim=self.n_total_class,
                                   model_name=self.cfg.MODEL,
                                   pretrained=pretrained)
        # TODO: Implement the function to read trained model here

    def forward(self, x):
        self.model(x)

    def calc_loss(self, batch, prefix="train"):
        x, y = batch
        pred = self.forward(x)
        if isinstance(preds, tuple):
            assert len(preds) == 3
            preds = pred
        else:
            assert pred.shape[1] == self.n_total_class
            preds = torch.split(pred,
                                [self.cfg.N_GRAPHEME, self.cfg.N_VOWEL, self.cfg.N_CONSONANT],
                                dim=1)
        loss_grapheme = F.cross_entropy(preds[0], y[:, 0])
        loss_vowel = F.cross_entropy(preds[1], y[:, 1])
        loss_consonant = F.cross_entropy(preds[2], y[:, 2])
        loss = loss_grapheme + loss_vowel + loss_consonant
        logger_logs = {
            "{}_loss".format(prefix): loss.item(),
            "{}_loss_grapheme".format(prefix): loss_grapheme.item(),
            "{}_loss_vowel".format(prefix): loss_vowel.item(),
            "{}_loss_consonant".format(prefix): loss_consonant.item(),
            "{}_acc_grapheme".format(prefix): accuracy(preds[0], y[:, 0]),
            "{}_acc_vowel".format(prefix): accuracy(preds[1], y[:, 1]),
            "{}_acc_consonant".format(prefix): accuracy(preds[2], y[:, 2])
        }
        return loss, logger_logs


    def training_step(self, batch, batch_idx):
        loss, logger_logs = self.calc_loss(batch)
        return {"loss": loss, "progress_bar": {"training_loss": loss}, "log": logger_logs}

    def validation_step(self, batch, batch_idx):
        _, logger_logs = self.calc_loss(batch, "val")
        return logger_logs
        
    def validation_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_loss_grapheme = torch.stack([x["val_loss_grapeme"] for x in outputs]).mean()
        avg_loss_vowel = torch.stack([x["val_loss_vowel"] for x in outputs]).mean()
        avg_loss_consonant = torch.stack([x["val_loss_consonant"] for x in outputs]).mean()
        avg_acc_grapheme = torch.stack([x["val_acc_grapheme"] for x in outputs]).mean()
        avg_acc_vowel = torch.stack([x["val_acc_vowel"] for x in outputs]).mean()
        avg_acc_consonant = torch.stack([x["val_acc_consonant"] for x in outputs]).mean()
        logs = {
            "val_loss": avg_loss,
            "val_loss_grapheme": avg_loss_grapheme,
            "val_loss_vowel": avg_loss_vowel,
            "val_loss_consonant": avg_loss_consonant,
            "val_acc_grapheme": avg_acc_grapheme,
            "val_acc_vowel": avg_acc_vowel,
            "val_acc_consonant": avg_acc_consonant
        }
        return {"avg_val_loss": avg_loss, "log": logs}

    def test_step(self, batch, batch_idx):
        # TODO Implement here
        x, y = batch
        y_hat = self.forward(x)

    def test_end(self, outputs):
        # TODO Implement here
        pass

    def configure_optimizers(self):
        tmp = self.cfg.NN.OPTIM_PARAMS[0]
        for d in self.cfg.NN.OPTIM_PARAMS:
            tmp.update(**d)
        optimizer = getattr(torch.optim, self.cfg.NN.OPTIM_NAME)(self.model.parameters(), **tmp)

        tmp = self.cfg.NN.SCHEDULER_PARAMS[0]
        for d in self.cfg.NN.SCHEDULER_PARAMS:
            tmp.update(**d)
        scheduler = getattr(torch.optim.lr_scheduler, self.cfg.NN.SCHEDULER_NAME)(self.optimizer, **tmp)
        return optimizer, scheduler

    @pl.data_loader
    def train_dataloader(self):
        # TODO: Implement here
        pass

    @pl.data_loader
    def val_dataloader(self):
        # TODO: Implement here
        pass

    @pl.data_loader
    def test_dataloader(self):
        # TODO: Implement here
        pass

