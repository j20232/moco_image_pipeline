from pathlib import Path

import torch
from .pipeline.models import PretrainedCNN

GRAPH = 168
VOWEL = 11
CONSO = 7
FILES =[
    "test_image_data_0.parquet.zip",
    "test_image_data_1.parquet.zip",
    "test_image_data_2.parquet.zip",
    "test_image_data_3.parquet.zip"
]


class BengaliKernel():

    def __init__(self, cfg, input_path, model_weight_path, cache_dir):
        self.cfg = cfg
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = PretrainedCNN(in_channels=3, out_dim=GRAPH + VOWEL + CONSO,
                                   is_local=True, **self.cfg["model"])
        self.model.load_state_dict(torch.load(str(model_weight_path)))
        self.model = self.model.to(self.device)
        print("Loaded pretrained model: {}".format(model_weight_path))
        self.read_input_parquets(input_path, cache_dir)
    
    def read_input_files(self, input_path, cache_dir):
        assert False

    def define_dataloader(self):
        assert False

    def predict(self):
        self.model.eval()
        with torch.no_grad():
            assert False

