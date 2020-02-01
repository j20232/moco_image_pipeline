import copy
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from pipeline.datasets import SimpleDataset
from pipeline.functions.metrics import accuracy
from pipeline.models import PretrainedCNN

GRAPH = 168
VOWEL = 11
CONSO = 7
ROOT_PATH = Path(".").resolve()
CONFIG_PATH = ROOT_PATH / "config"
TRAIN_CSV_PATH = ROOT_PATH / "input" / "train.csv"
TRAIN_IMG_PATH = ROOT_PATH / "input" / "train_images"
TEST_IMG_PATH = ROOT_PATH / "input" / "test_images"
SUB_CSV_PATH = ROOT_PATH / "input" / "sample_submission.csv"


class Bengali():

    def __init__(self, cfg):
        super(Bengali, self).__init__()
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
        df = train_df.head(64)
        self.train_loader = self.get_train_dataloader(df, True)
        df = train_df.head(64)
        self.valid_loader = self.get_train_dataloader(df, False)

    def fit(self):
        self.model = self.model.to(self.device)
        best_model_weight = copy.deepcopy(self.model.state_dict())
        best_params = {"loss": 10000000}
        early_stopping_count = 0
        for ep in tqdm(range(self.cfg["params"]["epochs"])):
            self.scheduler.step()
            results_train = self.train_one_epoch()
            results_valid = self.valid_one_epoch()
            self.show_logs(ep, results_train, results_valid)
            if results_valid["loss"] > best_params["loss"]:
                best_params["loss"] = results_valid["loss"]
                best_params["loss_grapheme"] = results_valid["loss_grapheme"]
                best_params["loss_vowel"] = results_valid["loss_vowel"]
                best_params["loss_consonant"] = results_valid["loss_consonant"]
                best_params["acc"] = results_valid["acc"]
                best_params["acc_grapheme"] = results_valid["acc_grapheme"]
                best_params["acc_vowel"] = results_valid["acc_vowel"]
                best_params["acc_consonant"] = results_valid["acc_consonant"]
                best_model_weight = copy.deepcopy(self.model.state_dict())
                early_stopping_count = 0
            else:
                early_stopping_count += 1
            if early_stopping_count > self.cfg["params"]["es_rounds"]:
                print("Early stopping at round {}".format(ep))
                break
        self.model.load_state_dict(best_model_weight)
        self.model = self.model.to("cpu")
        return self.model

    def train_one_epoch(self):
        self.model.train()
        log = {
            "loss": 0, "loss_grapheme": 0, "loss_vowel": 0, "loss_consonant": 0,
            "acc_grapheme": 0, "acc_vowel": 0, "acc_consonant": 0
        }

        loader_length = len(self.train_loader)
        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            preds = self.model(inputs)
            if isinstance(preds, tuple) is False:
                preds = torch.split(preds, [GRAPH, VOWEL, CONSO], dim=1)

            loss_grapheme = F.cross_entropy(preds[0], labels[:, 0])
            loss_vowel = F.cross_entropy(preds[1], labels[:, 1])
            loss_consonant = F.cross_entropy(preds[2], labels[:, 2])
            loss = loss_grapheme + loss_vowel + loss_consonant
            loss.backward()
            self.optimizer.step()

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
        log["acc"] = (log["acc_grapheme"] + log["acc_vowel"] + log["acc_consonant"]) / 3
        return log

    def valid_one_epoch(self):
        self.model.eval()
        log = {
            "loss": 0, "loss_grapheme": 0, "loss_vowel": 0, "loss_consonant": 0,
            "acc_grapheme": 0, "acc_vowel": 0, "acc_consonant": 0
        }

        loader_length = len(self.valid_loader)
        with torch.no_grad():
            for inputs, labels in self.valid_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                preds = self.model(inputs)
                if isinstance(preds, tuple) is False:
                    preds = torch.split(preds, [GRAPH, VOWEL, CONSO], dim=1)

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
        log["acc"] = (log["acc_grapheme"] + log["acc_vowel"] + log["acc_consonant"]) / 3
        return log

    def test():
        pass

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

    def show_logs(self, epoch, results_train, results_valid):
        if self.cfg["params"]["verbose"] == -1 or epoch + 1 % self.cfg["params"]["verbose"] != 0:
            return
        header = "| train / valid | epoch "
        train = "| train | {} ".format(epoch + 1)
        valid = "| valid | {} ".format(epoch + 1)
        for key in results_train.keys():
            header += "| {} ".format(key)
            train += "| {} ".format(results_train[key])
            valid += "| {} ".format(results_valid[key])
        header += "|"
        train += "|"
        valid += "|"
        print(header)
        print(train)
        print(valid)
        print("--------------------")
