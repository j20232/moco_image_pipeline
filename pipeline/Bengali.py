import copy
import pandas as pd
import shutil
from pathlib import Path
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from pipeline.datasets import SimpleDataset
from pipeline.functions.metrics import accuracy
from pipeline.models import PretrainedCNN
from pipeline.utils.train_utils import show_logs

GRAPH = 168
VOWEL = 11
CONSO = 7
ROOT_PATH = Path(".").resolve()
CONFIG_PATH = ROOT_PATH / "config"
LOG_PATH = ROOT_PATH / "logs"
TRAIN_CSV_PATH = ROOT_PATH / "input" / "train.csv"
TRAIN_IMG_PATH = ROOT_PATH / "input" / "train_images"
TEST_IMG_PATH = ROOT_PATH / "input" / "test_images"
SUB_CSV_PATH = ROOT_PATH / "input" / "sample_submission.csv"


class Bengali():

    def __init__(self, name, index, cfg):
        super(Bengali, self).__init__()
        self.competition_name = name
        self.index = index
        self.cfg = cfg
        self.n_total_class = GRAPH + VOWEL + CONSO
        self.model = PretrainedCNN(in_channels=3, out_dim=self.n_total_class,
                                   model_name=self.cfg["model"]["name"],
                                   pretrained=self.cfg["model"]["pretrained"])
        self.optimizer = getattr(optim, self.cfg["optim"]["name"])(
            self.model.parameters(), **self.cfg["optim"]["params"][0])
        self.scheduler = getattr(lr_scheduler, self.cfg["scheduler"]["name"])(
            self.optimizer, **self.cfg["scheduler"]["params"][0])
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # TODO: Implement the function to read trained model here

        # TODO: define how to split data
        train_df = pd.read_csv(TRAIN_CSV_PATH)
        df = train_df.head(10)
        self.train_loader = self.get_train_dataloader(df, True)
        df = train_df.head(10)
        self.valid_loader = self.get_train_dataloader(df, False)

    def fit(self):
        self.model = self.model.to(self.device)
        best_model_weight = copy.deepcopy(self.model.state_dict())
        best_results = {"loss": 10000000}
        early_stopping_count = 0
        final_epoch = 0
        results_train = {"loss": 0, "loss_grapheme": 0, "loss_vowel": 0, "loss_consonant": 0,
                         "acc_grapheme": 0, "acc_vowel": 0, "acc_consonant": 0}
        results_valid = results_train
        Path(LOG_PATH).mkdir(parents=True, exist_ok=True)
        save_path = LOG_PATH / self.competition_name / self.index
        if save_path.exists():
            shutil.rmtree(str(save_path))
        writer = SummaryWriter(log_dir=str(save_path))
        for ep in tqdm(range(self.cfg["params"]["epochs"])):
            final_epoch += 1
            self.scheduler.step()
            results_train = self.train_one_epoch(results_train)
            results_valid = self.valid_one_epoch(results_valid)
            show_logs(self.cfg, ep, results_train, results_valid)

            # early stopping
            if results_valid["loss"] < best_results["loss"]:
                best_results["loss"] = results_valid["loss"]
                best_results["loss_grapheme"] = results_valid["loss_grapheme"]
                best_results["loss_vowel"] = results_valid["loss_vowel"]
                best_results["loss_consonant"] = results_valid["loss_consonant"]
                best_results["acc"] = results_valid["acc"]
                best_results["acc_grapheme"] = results_valid["acc_grapheme"]
                best_results["acc_vowel"] = results_valid["acc_vowel"]
                best_results["acc_consonant"] = results_valid["acc_consonant"]
                best_model_weight = copy.deepcopy(self.model.state_dict())
                early_stopping_count = 0
            else:
                early_stopping_count += 1
            if early_stopping_count > self.cfg["params"]["es_rounds"]:
                print("Early stopping at round {}".format(ep))
                break

            # logging
            writer.add_scalars("data/loss",
                               {"train": results_train["loss"],
                                "valid": results_valid["loss"]}, ep)
            writer.add_scalars("data/loss_grapheme",
                               {"train": results_train["loss_grapheme"],
                                "valid": results_valid["loss_grapheme"]}, ep)
            writer.add_scalars("data/loss_vowel",
                               {"train": results_train["loss_vowel"],
                                "valid": results_valid["loss_vowel"]}, ep)
            writer.add_scalars("data/loss_consonant",
                               {"train": results_train["loss_consonant"],
                                "valid": results_valid["loss_consonant"]}, ep)
            writer.add_scalars("data/acc",
                               {"train": results_train["acc"],
                                "valid": results_valid["acc"],
                                "train_grapheme": results_train["acc_grapheme"],
                                "valid_grapheme": results_valid["acc_grapheme"],
                                "train_vowel": results_train["acc_vowel"],
                                "valid_vowel": results_valid["acc_vowel"],
                                "train_consonant": results_train["acc_consonant"],
                                "valid_consonant": results_valid["acc_consonant"]}, ep)
        self.model.load_state_dict(best_model_weight)
        self.model = self.model.to("cpu")
        writer.close()
        return self.model, best_results, final_epoch

    def calc_loss(self, preds, labels, log=None, loader_length=1):
        loss_grapheme = F.cross_entropy(preds[0], labels[:, 0])
        loss_vowel = F.cross_entropy(preds[1], labels[:, 1])
        loss_consonant = F.cross_entropy(preds[2], labels[:, 2])
        loss = loss_grapheme + loss_vowel + loss_consonant
        acc_grapheme = accuracy(preds[0], labels[:, 0])
        acc_vowel = accuracy(preds[1], labels[:, 1])
        acc_consonant = accuracy(preds[2], labels[:, 2])
        log["loss"] += (loss / loader_length).cpu().detach().numpy()
        log["loss_grapheme"] += (loss_grapheme / loader_length).cpu().detach().numpy()
        log["loss_vowel"] += (loss_vowel / loader_length).cpu().detach().numpy()
        log["loss_consonant"] += (loss_consonant / loader_length).cpu().detach().numpy()
        log["acc_grapheme"] += (acc_grapheme / loader_length).cpu().detach().numpy()
        log["acc_vowel"] += (acc_vowel / loader_length).cpu().detach().numpy()
        log["acc_consonant"] += (acc_consonant / loader_length).cpu().detach().numpy()
        return loss, log

    def train_one_epoch(self, log):
        self.model.train()
        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            preds = self.model(inputs)
            if isinstance(preds, tuple) is False:
                preds = torch.split(preds, [GRAPH, VOWEL, CONSO], dim=1)
            loss, log = self.calc_loss(preds, labels, log, len(self.train_loader))
            loss.backward()
            self.optimizer.step()
        log["acc"] = (log["acc_grapheme"] + log["acc_vowel"] + log["acc_consonant"]) / 3
        return log

    def valid_one_epoch(self, log):
        self.model.eval()
        with torch.no_grad():
            for inputs, labels in self.valid_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                preds = self.model(inputs)
                if isinstance(preds, tuple) is False:
                    preds = torch.split(preds, [GRAPH, VOWEL, CONSO], dim=1)
                loss, log = self.calc_loss(preds, labels, log, len(self.valid_loader))
        log["acc"] = (log["acc_grapheme"] + log["acc_vowel"] + log["acc_consonant"]) / 3
        return log

    def get_train_dataloader(self, df, is_train):
        # Train if is_train else Valid
        paths = [Path(TRAIN_IMG_PATH / f"{x}.png") for x in df["image_id"].values]
        labels = df[["grapheme_root", "vowel_diacritic", "consonant_diacritic"]].values
        # TODO: Implement augmentation methods
        return DataLoader(SimpleDataset(paths, labels, transform=transforms.ToTensor()),
                          batch_size=self.cfg["params"]["batch_size"], shuffle=is_train)

    def get_test_dataloader(self):
        # TODO: Fix here
        paths = [x for x in TEST_IMG_PATH.glob("*.png")]
        return DataLoader(SimpleDataset(paths, transform=transforms.ToTensor()),
                          batch_size=self.cfg["params"]["batch_size"])
