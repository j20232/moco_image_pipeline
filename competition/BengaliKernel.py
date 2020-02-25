import os
import sys

import gc
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.append(os.path.join(".."))
from mcp.datasets import SimpleDataset, SimpleDatasetNoCache
from mcp.models import PretrainedCNN
from mcp.utils.convert import crop_and_resize_img

GRAPH = 168
VOWEL = 11
CONSO = 7
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
        self.model = PretrainedCNN(in_channels=1, out_dim=GRAPH + VOWEL + CONSO,
                                   is_local=True, **self.cfg["model"])
        self.model.load_state_dict(torch.load(str(model_weight_path), map_location=self.device))
        self.model = self.model.to(self.device)
        print("Loaded pretrained model: {}".format(model_weight_path))
        self.input_path = input_path / competition
        self.output_path = output_path
        self.cache_dir = output_path / "cache"

    def predict(self):
        gc.enable()
        print("Reading input parquet files...")
        test_files = [self.input_path / "test_image_data_0.parquet",
                      self.input_path / "test_image_data_1.parquet",
                      self.input_path / "test_image_data_2.parquet",
                      self.input_path / "test_image_data_3.parquet"]
        row_id = None
        target = None
        for f in test_files:
            img_df = pq.read_pandas(f).to_pandas()
            imgs = []
            paths = []
            for idx in range(len(img_df)):
                img0 = 255 - img_df.iloc[idx, 1:].values.reshape(HEIGHT, WIDTH).astype(np.uint8)
                img = (img0 * (255.0 / img0.max())).astype(np.uint8)
                img = crop_and_resize_img(img, SIZE, WIDTH, HEIGHT)
                name = img_df.iloc[idx, 0]
                imgs.append(img)
                paths.append(name)
            tfms = [Normalizer(), transforms.ToTensor()]
            loader = DataLoader(SimpleDatasetNoCache(imgs, paths, transform=transforms.Compose(tfms)),
                                batch_size=self.cfg["params"]["test_batch_size"], shuffle=False,
                                num_workers=self.cfg["params"]["num_workers"])
            names, graph, vowel, conso = self.predict_for_ensemble(loader)
            g_ids = [f"{s}_grapheme_root" for s in names]
            v_ids = [f"{s}_vowel_diacritic" for s in names]
            c_ids = [f"{s}_consonant_diacritic" for s in names]
            r = np.stack([g_ids, v_ids, c_ids], 1)
            row_id = np.append(row_id, r.flatten()) if row_id is not None else r.flatten()
            g = np.argmax(graph, axis=1)
            v = np.argmax(vowel, axis=1)
            c = np.argmax(conso, axis=1)
            t = np.stack([g, v, c], 1)
            target = np.append(target, t.flatten()) if target is not None else t.flatten()
            del graph, vowel, conso
            del g_ids, v_ids, c_ids, r
            del g, v, c, t
            del imgs, paths
            gc.collect()

        submission_df = pd.DataFrame({'row_id': row_id, 'target': target})
        submission_df.to_csv(self.output_path / 'submission.csv', index=False)
        print(submission_df.head(10))
        print("Submission length: ", len(submission_df))
        print("Done")

    # ------------------------------ Prediction with cache ------------------------------

    def predict_with_cache(self):
        self.read_input_parquets()
        self.img_paths = list(self.cache_dir.glob("*.png"))
        tfms = [Normalizer(), transforms.ToTensor()]
        test_dataloader = DataLoader(SimpleDataset(self.img_paths, transform=transforms.Compose(tfms)),
                                     batch_size=self.cfg["params"]["test_batch_size"], shuffle=False,
                                     num_workers=self.cfg["params"]["num_workers"])
        """
        grapheme_df, vowel_df, conso_df = self.predict_for_ensemble(test_dataloader)
        ids = grapheme_df.index.values
        g_ids = [f"{s}_grapheme_root" for s in ids]
        v_ids = [f"{s}_vowel_diacritic" for s in ids]
        c_ids = [f"{s}_consonant_diacritic" for s in ids]
        row_id = np.stack([g_ids, v_ids, c_ids], 1)
        row_id = row_id.flatten()
        g = np.argmax(grapheme_df.values, axis=1)
        v = np.argmax(vowel_df.values, axis=1)
        c = np.argmax(conso_df.values, axis=1)
        target = np.stack([g, v, c], 1)
        target = target.flatten()
        submission_df = pd.DataFrame({'row_id': row_id, 'target': target})
        submission_df.to_csv(self.output_path / 'submission.csv', index=False)
        print(submission_df.head(10))
        print("Submission length: ", len(submission_df))
        print("Done")
        """

    # --------------------------------------- Utils ---------------------------------------

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
            img_df = pd.read_parquet(f)
            for idx in range(len(img_df)):
                img0 = 255 - img_df.iloc[idx, 1:].values.reshape(HEIGHT, WIDTH).astype(np.uint8)
                img = (img0 * (255.0 / img0.max())).astype(np.uint8)
                img = crop_and_resize_img(img, SIZE, WIDTH, HEIGHT)
                name = img_df.iloc[idx, 0]
                cv2.imwrite(str(self.cache_dir / f"{name}.png"), img)
            del img_df
            gc.collect()
        print("Save parquet files as png at {}".format(self.cache_dir))

    def predict_for_ensemble(self, test_dataloader):
        self.model.eval()
        names = []
        graph = None
        vowel = None
        conso = None
        with torch.no_grad():
            for imgs, paths in test_dataloader:
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
        return names, graph, vowel, conso
