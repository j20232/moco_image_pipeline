import os
import sys

import gc
import numpy as np
import pandas as pd
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
        self.model = PretrainedCNN(in_channels=1, out_dim=GRAPH + VOWEL + CONSO, **self.cfg["model"])
        self.model.load_state_dict(torch.load(str(model_weight_path), map_location=self.device))
        self.model = self.model.to(self.device)
        print("Loaded pretrained model: {}".format(model_weight_path))
        self.input_path = input_path / competition
        self.output_path = output_path
        self.cache_dir = output_path / "cache"

    def crop(self, x):
        x = (x * (255.0 / x.max())).astype(np.uint8)
        return crop_and_resize_img(x, SIZE, WIDTH, HEIGHT)

    def predict(self):
        gc.enable()
        print("Reading input parquet files...")
        test_files = [self.input_path / "test_image_data_0.parquet",
                      self.input_path / "test_image_data_1.parquet",
                      self.input_path / "test_image_data_2.parquet",
                      self.input_path / "test_image_data_3.parquet"]
        row_id = None
        target = None
        bs = self.cfg["params"]["test_batch_size"]
        for f in test_files:
            img_df = pd.read_parquet(f)
            num_rows, num_cols = img_df.shape
            imgs = []
            paths = []
            for idx in range(int(num_rows / bs) + 1):
                name = img_df.iloc[idx * bs : (idx + 1) * bs, 0]
                img0 = img_df.iloc[idx * bs : (idx + 1) * bs, 1:].values
                img0 = np.reshape(img0, (img0.shape[0], HEIGHT, WIDTH))
                img0 = 255 - img0.astype(np.uint8)
                img = [self.crop(im) for im in img0]
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                imgs.extend(img)
                paths.extend(name)
            tfms = [Normalizer(), transforms.ToTensor()]
            loader = DataLoader(SimpleDatasetNoCache(imgs, paths, transform=transforms.Compose(tfms)),
                                batch_size=bs, shuffle=False, num_workers=self.cfg["params"]["num_workers"])
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
