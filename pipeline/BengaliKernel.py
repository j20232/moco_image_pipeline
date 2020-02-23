import gc
import zipfile
from tqdm import tqdm
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from .pipeline.datasets import SimpleDataset
from .pipeline.models import PretrainedCNN
from .pipeline.utils.convert import crop_and_resize_img

GRAPH = 168
VOWEL = 11
CONSO = 7
FILES = [
    "test_image_data_0.parquet.zip",
    "test_image_data_1.parquet.zip",
    "test_image_data_2.parquet.zip",
    "test_image_data_3.parquet.zip"
]
SIZE = 128
WIDTH = 236
HEIGHT = 137


# dirty
class Normalizer():
    def __call__(self, img):
        return (img.astype(np.float32) - 0.0692) / 0.2051


class BengaliKernel():

    def __init__(self, competition, cfg, input_path, model_weight_path, output_path):
        self.cfg = cfg
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = PretrainedCNN(in_channels=3, out_dim=GRAPH + VOWEL + CONSO,
                                   is_local=True, **self.cfg["model"])
        self.model.load_state_dict(torch.load(str(model_weight_path), map_location=self.device))
        self.model = self.model.to(self.device)
        print("Loaded pretrained model: {}".format(model_weight_path))
        self.input_path = input_path / competition
        self.output_path = output_path
        self.cache_dir = output_path / "cache"
        self.read_input_parquets()

    def read_input_parquets(self):
        if self.cache_dir.exists():
            print("You've already generated cache dir at {}".format(self.cache_dir))
            return
        gc.enable()
        print("Reading input parquet files...")
        test_files = [self.input_path / "test_image_data_0.parquet",
                      self.input_path / "test_image_data_1.parquet",
                      self.input_path / "test_image_data_2.parquet",
                      self.input_path / "test_image_data_3.parquet"]
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        for f in test_files:
            img_df = pd.read_parquet(f, engine="pyarrow")
            for idx in tqdm(range(len(img_df))):
                img0 = 255 - img_df.iloc[idx, 1:].values.reshape(HEIGHT, WIDTH).astype(np.uint8)
                img = (img0 * (255.0 / img0.max())).astype(np.uint8)
                img = crop_and_resize_img(img, SIZE, WIDTH, HEIGHT)
                name = img_df.iloc[idx, 0]
                cv2.imwrite(str(self.cache_dir / f"{name}.png"), img)
            del img_df
            gc.collect()
        print("Save parquet files as png at {}".format(self.cache_dir))

    def define_dataloader(self):
        self.img_paths = list(self.cache_dir.glob("*.png"))
        tfms = [Normalizer(), transforms.ToTensor()]
        return DataLoader(SimpleDataset(self.img_paths, transform=transforms.Compose(tfms)),
                          batch_size=self.cfg["params"]["test_batch_size"], shuffle=False,
                          num_workers=self.cfg["params"]["num_workers"])

    def predict_for_ensemble(self):
        test_dataloader = self.define_dataloader()
        self.model.eval()
        names = []
        graph = None
        vowel = None
        conso = None
        with torch.no_grad():
            for imgs, paths in tqdm(test_dataloader):
                imgs = imgs.to(self.device)
                preds = self.model(imgs)
                if isinstance(preds, tuple) is False:
                    preds = torch.split(preds, [GRAPH, VOWEL, CONSO], dim=1)
                names.extend([n.split("/")[-1].split(".")[0] for n in list(paths)])
                prob_graph = F.softmax(preds[0], dim=1).cpu().detach().numpy()
                prob_vowel = F.softmax(preds[1], dim=1).cpu().detach().numpy()
                prob_conso = F.softmax(preds[2], dim=1).cpu().detach().numpy()
                graph = prob_graph if graph is None else np.append(graph, prob_graph, axis=0)
                vowel = prob_vowel if vowel is None else np.append(vowel, prob_vowel, axis=0)
                conso = prob_conso if conso is None else np.append(conso, prob_conso, axis=0)
        grapheme_df = pd.DataFrame(graph)
        vowel_df = pd.DataFrame(vowel)
        conso_df = pd.DataFrame(conso)
        grapheme_df["image_id"] = names
        grapheme_df = grapheme_df.set_index(["image_id"])
        vowel_df["image_id"] = names
        vowel_df = vowel_df.set_index(["image_id"])
        conso_df["image_id"] = names
        conso_df = conso_df.set_index(["image_id"])
        return grapheme_df, vowel_df, conso_df

    def predict(self):
        grapheme_df, vowel_df, conso_df = self.predict_for_ensemble()
        row_id = []
        target = []
        for i in tqdm(range(len(self.img_paths))):
            row_id += [f'Test_{i}_grapheme_root', f'Test_{i}_vowel_diacritic',
                       f'Test_{i}_consonant_diacritic']
            g = np.argmax(grapheme_df.query("image_id=='Test_{}'".format(i)).values[0])
            v = np.argmax(vowel_df.query("image_id=='Test_{}'".format(i)).values[0])
            c = np.argmax(conso_df.query("image_id=='Test_{}'".format(i)).values[0])
            target += [g, v, c]
        submission_df = pd.DataFrame({'row_id': row_id, 'target': target})
        submission_df.to_csv(self.output_path / 'submission.csv', index=False)
        print("Done")
