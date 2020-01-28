import pandas as pd
from pathlib import Path
from yacs.config import CfgNode as CN

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from pipeline.datasets import SimpleDataset
from pipeline.functions import accuracy
from pipeline.models import PretrainedCNN
from pipeline.utils.seed import seed_everything


# Default settings
_C = CN()
_C.SEED = 1116
_C.BS = 32
_C.MODEL = "se_resnext50_32x4d"
_C.PRETRAINED = "null"
_C.OPT_N = "Adam"
_C.OPT_P = []
_C.SCH_N = "ReduceLROnPlateau"
_C.SCH_P = []

# Const
GRAPH = 160
VOWEL=11
CONSO = 7
ROOT_PATH = Path(".").resolve()
CONFIG_PATH = ROOT_PATH / "config"
TRAIN_CSV_PATH = ROOT_PATH / "input" / "train.csv"
TRAIN_IMG_PATH = ROOT_PATH / "input" / "train_images"
TEST_IMG_PATH = ROOT_PATH / "input" / "test_images"
SUB_CSV_PATH = ROOT_PATH / "input" / "sample_submission.csv"


def get_cfg():
    return _C.clone()

class Bengali(pl.LightningModule):

    def __init__(self, cfg_dir, index):
        super(Bengali, self).__init__()
        # read cfg
        self.cfg = get_cfg()
        yaml_path = CONFIG_PATH / cfg_dir / f"{index}.yaml"
        self.cfg.merge_from_file(yaml_path)
        self.cfg.freeze()
        seed_everything(self.cfg.SEED)

        # Set pretrained='imagenet' to download imagenet pretrained model...
        pretrained = self.cfg.PRETRAINED
        if pretrained == "null":
            pretrained = None
        self.n_total_class = GRAPH + VOWEL + CONSO
        self.model = PretrainedCNN(in_channels=1, out_dim=self.n_total_class,
                                   model_name=self.cfg.MODEL,
                                   pretrained=pretrained)
        # TODO: Implement the function to read trained model here

        # TODO: define how to split data
        train_df = pd.read_csv(TRAIN_CSV_PATH)
        self.train_df = train_df
        self.valid_df = train_df

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
            preds = torch.split(pred, [GRAPH, VOWEL, CONSO], dim=1)
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
        print("test_step")
        x, y = batch
        y_hat = self.forward(x)

    def test_end(self, outputs):
        print("test_end")

    def configure_optimizers(self):
        optimizer = getattr(optim, self.cfg.OPT_N)(self.model.parameters(), **self.cfg.OPT_P[0])
        scheduler = getattr(lr_scheduler, self.cfg.SCH_N)(optimizer, **self.cfg.SCH_P[0])
        return optimizer, scheduler

    @pl.data_loader
    def train_dataloader(self):
        paths = [Path(TRAIN_IMG_PATH / f"{x}.png") for x in self.train_df["image_id"].values]
        labels = self.train_df[["grapheme_root", "vowel_diacritic", "consonant_diacritic"]].values
        return DataLoader(SimpleDataset(paths, labels, transform=None), batch_size=self.cfg.BS)

    @pl.data_loader
    def val_dataloader(self):
        paths = [Path(TRAIN_IMG_PATH / f"{x}.png") for x in self.valid_df["image_id"].values]
        labels = self.valid_df[["grapheme_root", "vowel_diacritic", "consonant_diacritic"]].values
        return DataLoader(SimpleDataset(paths, labels, transform=None), batch_size=self.cfg.BS)

    @pl.data_loader
    def test_dataloader(self):
        paths = [x for x in TEST_IMG_PATH.glob("*.png")]
        return DataLoader(SimpleDataset(paths, transform=None), batch_size=self.cfg.BS)
