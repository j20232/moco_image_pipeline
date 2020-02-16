import copy
import pandas as pd
import shutil
from tqdm import tqdm
import numpy as np
from pathlib import Path

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

import pipeline.augmentation as aug
from pipeline.datasets import SimpleDataset
from pipeline.functions.metrics import accuracy
from pipeline.models import PretrainedCNN
from pipeline.utils import show_logs

GRAPH = 168
VOWEL = 11
CONSO = 7
ROOT_PATH = Path(".").resolve()
CONFIG_PATH = ROOT_PATH / "config"
LOG_PATH = ROOT_PATH / "logs"
MODEL_PATH = ROOT_PATH / "models"
INPUT_PATH = ROOT_PATH / "input"
TRAIN_IMG_PATH = INPUT_PATH / "train_images"
SUB_CSV_PATH = INPUT_PATH / "sample_submission.csv"


# dirty
class Normalizer():
    def __call__(self, img):
        return (img.astype(np.float32) - 0.0692) / 0.2051


class Bengali():

    def __init__(self, name, index, cfg, is_train=True):
        super(Bengali, self).__init__()
        self.competition_name = name
        self.index = index
        self.cfg = cfg
        self.n_total_class = GRAPH + VOWEL + CONSO
        self.model = PretrainedCNN(in_channels=3, out_dim=self.n_total_class,
                                   model_name=self.cfg["model"]["name"],
                                   pretrained=self.cfg["model"]["pretrained"])

        self.competition_model_path = MODEL_PATH / self.competition_name
        self.competition_model_path.mkdir(parents=True, exist_ok=True)
        self.check_point_weight_path = self.competition_model_path / f"check_point_{self.index}.pth"
        if self.check_point_weight_path.exists():
            print("Loaded check_point ({})".format(self.check_point_weight_path))
            self.model.load_state_dict(torch.load(str(self.check_point_weight_path)))

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if is_train:
            self.__set_training()

    def __set_training(self):
        self.optimizer = getattr(optim, self.cfg["optim"]["name"])(
            self.model.parameters(), **self.cfg["optim"]["params"][0])
        self.scheduler = getattr(lr_scheduler, self.cfg["scheduler"]["name"])(
            self.optimizer, **self.cfg["scheduler"]["params"][0])
        self.__create_validation_set()

    def __create_validation_set(self):
        train_df = pd.read_csv(INPUT_PATH / self.cfg["dataset"]["train"])
        self.train_loader = self.__get_train_dataloader(train_df, True)
        valid_df = pd.read_csv(INPUT_PATH / self.cfg["dataset"]["valid"])
        self.valid_loader = self.__get_train_dataloader(valid_df, False)
        print("Loaded train and validation dataset!")

    def __get_train_dataloader(self, df, is_train):
        # Train if is_train else Valid
        paths = [Path(TRAIN_IMG_PATH / f"{x}.png") for x in df["image_id"].values]
        labels = df[["grapheme_root", "vowel_diacritic", "consonant_diacritic"]].values
        tfms = []
        for tfm_dict in self.cfg["transform"]:
            name, params = tfm_dict["name"], tfm_dict["params"]
            lib = aug if name in aug.modules else transforms
            tfms.append(getattr(lib, name)(**params))
        tfms.append(Normalizer())
        tfms.append(transforms.ToTensor())
        return DataLoader(SimpleDataset(paths, labels, transform=transforms.Compose(tfms)),
                          batch_size=self.cfg["params"]["batch_size"], shuffle=is_train,
                          num_workers=self.cfg["params"]["num_workers"])

    def fit(self):
        self.__initialize_fitting()
        for ep in tqdm(range(self.cfg["params"]["epochs"])):
            train_log = {"loss": 0,
                         "loss_grapheme": 0, "loss_vowel": 0, "loss_consonant": 0,
                         "acc_grapheme": 0, "acc_vowel": 0, "acc_consonant": 0}
            valid_log = copy.deepcopy(train_log)
            results_train = self.__train_one_epoch(train_log)
            results_valid = self.__valid_one_epoch(valid_log)
            show_logs(self.cfg, ep, results_train, results_valid)
            self.__add_tensorboard(results_train, results_valid, ep)
            if self.__check_early_stopping(results_valid):
                print("Early stopping at round {}".format(ep))
                break
            self.model.load_state_dict(self.best_model_weight)
        self.__close_fitting()
        return self.best_results

    def __train_one_epoch(self, log):
        self.scheduler.step()
        self.model.train()
        for inputs, labels in tqdm(self.train_loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            preds = self.model(inputs)
            if isinstance(preds, tuple) is False:
                preds = torch.split(preds, [GRAPH, VOWEL, CONSO], dim=1)
            loss, log = self.__calc_loss(preds, labels, log, len(self.train_loader))
            loss.backward()
            self.optimizer.step()
        log["acc"] = (log["acc_grapheme"] + log["acc_vowel"] + log["acc_consonant"]) / 3
        return log

    def __valid_one_epoch(self, log):
        self.model.eval()
        with torch.no_grad():
            for inputs, labels in tqdm(self.valid_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                preds = self.model(inputs)
                if isinstance(preds, tuple) is False:
                    preds = torch.split(preds, [GRAPH, VOWEL, CONSO], dim=1)
                loss, log = self.__calc_loss(preds, labels, log, len(self.valid_loader))
        log["acc"] = (log["acc_grapheme"] + log["acc_vowel"] + log["acc_consonant"]) / 3
        return log

    def __calc_loss(self, preds, labels, log=None, loader_length=1):
        loss_grapheme = F.cross_entropy(preds[0], labels[:, 0])
        loss_vowel = F.cross_entropy(preds[1], labels[:, 1])
        loss_consonant = F.cross_entropy(preds[2], labels[:, 2])
        loss = loss_grapheme + loss_vowel + loss_consonant

        log["loss_grapheme"] += (loss_grapheme / loader_length).cpu().detach().numpy()
        log["loss_vowel"] += (loss_vowel / loader_length).cpu().detach().numpy()
        log["loss_consonant"] += (loss_consonant / loader_length).cpu().detach().numpy()
        log["loss"] += (loss / loader_length).cpu().detach().numpy()

        acc_grapheme = accuracy(preds[0], labels[:, 0])
        acc_vowel = accuracy(preds[1], labels[:, 1])
        acc_consonant = accuracy(preds[2], labels[:, 2])

        log["acc_grapheme"] += (acc_grapheme / loader_length).cpu().detach().numpy()
        log["acc_vowel"] += (acc_vowel / loader_length).cpu().detach().numpy()
        log["acc_consonant"] += (acc_consonant / loader_length).cpu().detach().numpy()
        return loss, log

    def __check_early_stopping(self, results_valid):
        if results_valid["loss"] < self.best_results["loss"]:
            self.best_results["loss_grapheme"] = results_valid["loss_grapheme"]
            self.best_results["loss_vowel"] = results_valid["loss_vowel"]
            self.best_results["loss_consonant"] = results_valid["loss_consonant"]
            self.best_results["loss"] = results_valid["loss"]

            self.best_results["acc_grapheme"] = results_valid["acc_grapheme"]
            self.best_results["acc_vowel"] = results_valid["acc_vowel"]
            self.best_results["acc_consonant"] = results_valid["acc_consonant"]
            self.best_results["acc"] = results_valid["acc"]

            self.best_model_weight = copy.deepcopy(self.model.state_dict())
            torch.save(self.best_model_weight, str(self.check_point_weight_path))
            self.early_stopping_count = 0
        else:
            self.early_stopping_count += 1
        return self.early_stopping_count > self.cfg["params"]["es_rounds"]

    def __initialize_fitting(self):
        self.model = self.model.to(self.device)
        self.best_model_weight = copy.deepcopy(self.model.state_dict())
        self.best_results = {"loss": 10000000}
        self.early_stopping_count = 0
        Path(LOG_PATH).mkdir(parents=True, exist_ok=True)
        save_path = LOG_PATH / self.competition_name / self.index
        shutil.rmtree(str(save_path), ignore_errors=True)
        self.writer = SummaryWriter(log_dir=str(save_path))
        print("Initialized the setting of training!")

    def __close_fitting(self):
        torch.save(self.best_model_weight, str(self.competition_model_path / f"{self.index}.pth"))
        self.writer.close()
        print("Finished training!")

    def __add_tensorboard(self, results_train, results_valid, ep):
        self.writer.add_scalars("data/loss",
                                {"train": results_train["loss"],
                                 "valid": results_valid["loss"]}, ep)
        self.writer.add_scalars("data/loss_grapheme",
                                {"train": results_train["loss_grapheme"],
                                 "valid": results_valid["loss_grapheme"]}, ep)
        self.writer.add_scalars("data/loss_vowel",
                                {"train": results_train["loss_vowel"],
                                 "valid": results_valid["loss_vowel"]}, ep)
        self.writer.add_scalars("data/loss_consonant",
                                {"train": results_train["loss_consonant"],
                                 "valid": results_valid["loss_consonant"]}, ep)
        self.writer.add_scalars("data/acc",
                                {"train": results_train["acc"],
                                 "valid": results_valid["acc"],
                                 "train_grapheme": results_train["acc_grapheme"],
                                 "valid_grapheme": results_valid["acc_grapheme"],
                                 "train_vowel": results_train["acc_vowel"],
                                 "valid_vowel": results_valid["acc_vowel"],
                                 "train_consonant": results_train["acc_consonant"],
                                 "valid_consonant": results_valid["acc_consonant"]}, ep)
