import numpy as np
from PIL import Image
from torchvision.transforms import transforms

# Input:  PIL Image or np.ndarray, Output: np.ndarray (0 ~ 1.0)


class CenterOrRandomCrop():
    def __init__(self, size, prob=0.8):
        self.random_crop = transforms.RandomCrop(size)
        self.center_crop = transforms.CenterCrop(size)
        self.prob = np.clip(prob, 0.0, 1.0)

    def __call__(self, img):
        src = img
        if type(img) is np.ndarray:
            if img.dtype == "float32" or img.dtype == "float64":
                src = (img * 255).astype(np.uint8)
            src = Image.fromarray(src)

        if np.random.uniform() < self.prob:
            out = self.center_crop(src)
        else:
            out = self.random_crop(src)

        out = np.asarray(out)
        if type(img) is np.ndarray and (img.dtype == "float32" or img.dtype == "float64"):
            out = (out / 255).astype(np.float32)
        return out
