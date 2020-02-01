import numpy as np
import cv2
from torch.utils.data.dataset import Dataset


class SimpleDataset(Dataset):
    def __init__(self, paths, labels=None, transform=None):
        self.paths = paths
        self.labels = labels
        self.is_train = False if self.labels is None else True
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        x = cv2.imread(str(self.paths[i]))
        # x = (255 - x).astype(np.float32) / 255.
        x = x.astype(np.float32) / 255.
        if self.transform:
            x = self.transform(x)
        return x, self.labels[i] if self.is_train else x
