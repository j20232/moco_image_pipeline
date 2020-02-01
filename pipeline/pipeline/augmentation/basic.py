import numpy as np
from torchvision.transforms import transforms
import albumenization as A


def apply_aug(aug, image):
    return aug(image=image)["image"]


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


class RandomBlur():
    def __init__(self, prob, median_blur_limit=5):
        self.prob = np.clip(prob, 0.0, 1.0)
        self.median_blur_limit = median_blur_limit

    def __call__(self, img):
        if np.random.uniform() < self.prob:
            r = np.random.uniform()
            if r < 0.25:
                img = apply_aug(A.Blur(p=1.0), img)
            elif r < 0.5:
                img = apply_aug(A.MedianBlur(blur_limit=self.median_blur_limit, p=1.0), img)
            elif r < 0.75:
                img = apply_aug(A.GaussianBlur(p=1.0), img)
            else:
                img = apply_aug(A.MotionBlur(p=1.0), img)
        return img


class RandomNoise():
    def __init__(self, prob, var_limit=5.):
        self.prob = np.clip(prob, 0.0, 1.0)
        self.var_limit = var_limit

    def __call__(self, img):
        if np.random.uniform() < self.prob:
            img = apply_aug(A.GaussianNoise(var_limit=self.var_limit / 255., p=1.0), img)
        else:
            img = apply_aug(A.MultiplicativeNoise(p=1.0), img)
        return img


class CoarseDropout():
    def __init__(self, prob, max_holes=8, max_height=8, max_width=8):
        self.prob = np.clip(prob, 0.0, 1.0)
        self.max_holes = max_holes
        self.max_height = max_height
        self.max_width = max_width

    def __call__(self, img):
        return apply_aug(A.CoarseDropout(p=self.prob, max_holes=self.max_holes,
                                         max_height=self.max_height, max_width=self.max_width), img)


class GridDistortion():
    def __init__(self, prob):
        self.prob = np.clip(prob, 0.0, 1.0)

    def __call__(self, img):
        return apply_aug(A.GridDistortion(p=self.prob), img)


class ElasticTransform():
    def __init__(self, prob, sigma=50, alpha=1, alpha_affine=1.0):
        self.prob = np.clip(prob, 0.0, 1.0)
        self.sigma = sigma
        self.alpha = alpha
        self.alpha_affine = alpha_affine

    def __call__(self, img):
        return apply_aug(A.ElasticTransform(p=self.prob, sigma=self.sigma,
                                            alpha=self.alpha, alpha_affine=self.alpha_affine), img)


class RandomBrightnessContrast():
    def __init__(self, prob):
        self.prob = np.clip(prob, 0.0, 1.0)

    def __call__(self, img):
        return apply_aug(A.IAAPiecewiseAffine(p=self.prob), img)


class ShiftScaleRotate():
    def __init__(self, prob, shift_limit=0.0625, scale_limit=0.1, rotate_limit=30):
        self.prob = prob
        self.shift_limit = shift_limit
        self.scale_limit = scale_limit
        self.rotate_limit = rotate_limit

    def __call__(self, img):
        return apply_aug(A.ShiftScaleRotate(p=1.0, shift_limit=self.shift_limit,
                                            scale_limit=self.scale_limit,
                                            rotate_limit=self.rotate_limit), img)
