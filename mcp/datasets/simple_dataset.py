import numpy as np
import cv2
from torch.utils.data.dataset import Dataset


class SimpleDataset(Dataset):
    # single channel
    def __init__(self, paths, labels=None, transform=None):
        self.paths = paths
        self.labels = labels
        self.is_train = False if self.labels is None else True
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        x = cv2.imread(str(self.paths[i]), cv2.IMREAD_GRAYSCALE)
        # x = (255 - x).astype(np.float32) / 255.
        x = x.astype(np.float64) / 255.
        if self.transform:
            x = self.transform(x)

        return (x, self.labels[i], str(self.paths[i])) if self.is_train else (x, str(self.paths[i]))


class SimpleDatasetNoCache(Dataset):

    def __init__(self, imgs, paths, transform=None):
        self.imgs = imgs
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        x = self.imgs[i].astype(np.float64) / 255.
        x = self.transform(x)
        return (x, str(self.paths[i]))
