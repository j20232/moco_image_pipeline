import numpy as np
from torchvision.transforms import transforms


class CenterOrRandomCrop():
    def __init__(self, size, prob=0.8):
        self.random_crop = transforms.RandomCrop(size)
        self.center_crop = transforms.CenterCrop(size)
        self.prob = np.clip(prob, 0.0, 1.0)

    def __call__(self, img):
        if np.rand.uniform() < self.prob:
            img = self.center_crop(img)
        else:
            img = self.random_crop(img)
        return img


class GaussianNoise():
    def __init__(self, prob, sigma):
        self.prob = np.clip(prob, 0.0, 1.0)
        self.sigma = sigma

    def __call__(self, img):
        if np.random.uniform() < self.prob:
            img += np.random.randn(*img.shape) * self.sigma
            img = np.clip(img, 0., 1.)
        return img
