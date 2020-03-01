import numpy as np

# ndarray: H x W x C


class GaussianNoise():
    def __init__(self, prob, sigma=0.2):
        self.prob = np.clip(prob, 0.0, 1.0)
        self.sigma = sigma

    def __call__(self, img):
        if np.random.uniform() < self.prob:
            img += np.random.randn(*img.shape) * self.sigma
            img = np.clip(img, 0., 1.)
        return img
