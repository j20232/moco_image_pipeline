import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import albumentations as A

# ndarray: H x W x C


def apply_aug(aug, image):
    return aug(image=image)["image"]


# ----------------------------------- Blur -------------------------------------------
class RandomBlur():
    def __init__(self, prob, blur_limit=9):
        self.prob = np.clip(prob, 0.0, 1.0)
        self.blur_limit = blur_limit

    def __call__(self, img):
        if np.random.uniform() < self.prob:
            r = np.random.uniform()
            if r < 0.4:
                img = apply_aug(A.Blur(blur_limit=self.blur_limit, always_apply=True), img)
            elif r < 0.6:
                img = apply_aug(A.GaussianBlur(blur_limit=self.blur_limit, always_apply=True), img)
            else:
                img = apply_aug(A.MotionBlur(blur_limit=self.blur_limit, always_apply=True), img)
        return img


# ----------------------------------- Noise -------------------------------------------

class GaussNoise():
    def __init__(self, prob, var_limit=(0.0, 0.07)):
        self.prob = np.clip(prob, 0.0, 1.0)
        self.var_limit = var_limit

    def __call__(self, img):
        return apply_aug(A.GaussNoise(var_limit=self.var_limit, p=self.prob), img)


class MultiplicativeNoise():
    def __init__(self, prob, var_limit=(0.6, 1.1)):
        self.prob = np.clip(prob, 0.0, 1.0)
        self.var_limit = var_limit

    def __call__(self, img):
        return apply_aug(A.MultiplicativeNoise(multiplier=self.var_limit, p=self.prob), img)


# ---------------------------------- Distortion ---------------------------------------

class GridDistortion():
    def __init__(self, prob, num_steps=10, distort_limit=0.7):
        self.prob = np.clip(prob, 0.0, 1.0)
        self.num_steps = num_steps
        self.distort_limit = distort_limit

    def __call__(self, img):
        return apply_aug(A.GridDistortion(p=self.prob, num_steps=self.num_steps,
                                          distort_limit=self.distort_limit), img)


class ElasticTransform():
    def __init__(self, prob, sigma=40, alpha=1, alpha_affine=15):
        self.prob = np.clip(prob, 0.0, 1.0)
        self.sigma = sigma
        self.alpha = alpha
        self.alpha_affine = alpha_affine

    def __call__(self, img):
        return apply_aug(A.ElasticTransform(p=self.prob, sigma=self.sigma,
                                            alpha=self.alpha, alpha_affine=self.alpha_affine), img)


class ShiftScaleRotate():
    def __init__(self, prob, shift_limit=0.0625, scale_limit=0.2, rotate_limit=20):
        self.prob = prob
        self.shift_limit = shift_limit
        self.scale_limit = scale_limit
        self.rotate_limit = rotate_limit

    def __call__(self, img):
        return apply_aug(A.ShiftScaleRotate(p=self.prob, shift_limit=self.shift_limit,
                                            scale_limit=self.scale_limit,
                                            rotate_limit=self.rotate_limit), img)

# ----------------------------------- Histogram ----------------------------------------


class HueSaturationValue():
    def __init__(self, prob, hue_shift_limit=20, sat_shift_limit=40, val_shift_limit=100):
        self.prob = np.clip(prob, 0.0, 1.0)
        self.hue_shift_limit = hue_shift_limit
        self.sat_shift_limit = sat_shift_limit
        self.val_shift_limit = val_shift_limit

    def __call__(self, img):
        out = img if img.dtype == "uint8" else (img * 255).astype(np.uint8)

        out = apply_aug(A.HueSaturationValue(p=self.prob, hue_shift_limit=self.hue_shift_limit,
                                             sat_shift_limit=self.sat_shift_limit,
                                             val_shift_limit=self.val_shift_limit), out)
        return out if img.dtype == "uint8" else (out / 255).astype(np.float64)


class RandomBrightnessContrast():
    def __init__(self, prob, brightness_limit=2.0, contrast_limit=0.6):
        self.prob = np.clip(prob, 0.0, 1.0)
        self.brightness_limit = brightness_limit
        self.contrast_limit = contrast_limit

    def __call__(self, img):
        return apply_aug(A.RandomBrightnessContrast(p=self.prob,
                                                    brightness_limit=self.brightness_limit,
                                                    contrast_limit=self.contrast_limit,
                                                    brightness_by_max=False,
                                                    ), img)


class RandomCLAHE():
    def __init__(self, prob, clip_limit=40.0, tile_grid_size=(16, 16)):
        self.prob = np.clip(prob, 0.0, 1.0)
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img):
        out = img if img.dtype == "uint8" else (img * 255).astype(np.uint8)
        out = apply_aug(A.CLAHE(p=self.prob, clip_limit=self.clip_limit,
                                tile_grid_size=self.tile_grid_size), out)
        return out if img.dtype == "uint8" else (out / 255).astype(np.float64)


# ------------------------------------- Removal ------------------------------------------

class CoarseDropout():
    def __init__(self, prob, max_holes=10, max_height=12, max_width=12):
        self.prob = np.clip(prob, 0.0, 1.0)
        self.max_holes = max_holes
        self.max_height = max_height
        self.max_width = max_width

    def __call__(self, img):
        return apply_aug(A.CoarseDropout(p=self.prob, max_holes=self.max_holes,
                                         max_height=self.max_height, max_width=self.max_width,
                                         fill_value=np.median(img)), img)


# ------------------------------------------- Augmix -------------------------------------------
# Reference: https://www.kaggle.com/haqishen/augmix-based-on-albumentations


def int_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval .
    Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.
    Returns:
    An int that results from scaling `maxval` according to `level`.
    """
    return int(level * maxval / 10)


def float_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval.
    Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.
    Returns:
    A float that results from scaling `maxval` according to `level`.
    """
    return float(level) * maxval / 10.


def sample_level(n):
    return np.random.uniform(low=0.1, high=n)


def autocontrast(pil_img, _):
    return ImageOps.autocontrast(pil_img)


def equalize(pil_img, _):
    return ImageOps.equalize(pil_img)


def posterize(pil_img, level):
    level = int_parameter(sample_level(level), 4)
    return ImageOps.posterize(pil_img, 4 - level)


def rotate(pil_img, level):
    degrees = int_parameter(sample_level(level), 30)
    if np.random.uniform() > 0.5:
        degrees = -degrees
    return pil_img.rotate(degrees, resample=Image.BILINEAR)


def solarize(pil_img, level):
    level = int_parameter(sample_level(level), 256)
    return ImageOps.solarize(pil_img, 256 - level)


def shear_x(pil_img, level):
    level = float_parameter(sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform(pil_img.size,
                             Image.AFFINE, (1, level, 0, 0, 1, 0),
                             resample=Image.BILINEAR)


def shear_y(pil_img, level):
    level = float_parameter(sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform(pil_img.size,
                             Image.AFFINE, (1, 0, 0, level, 1, 0),
                             resample=Image.BILINEAR)


def translate_x(pil_img, level):
    level = int_parameter(sample_level(level), pil_img.size[0] / 3)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform(pil_img.size,
                             Image.AFFINE, (1, 0, level, 0, 1, 0),
                             resample=Image.BILINEAR)


def translate_y(pil_img, level):
    level = int_parameter(sample_level(level), pil_img.size[0] / 3)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform(pil_img.size,
                             Image.AFFINE, (1, 0, 0, 0, 1, level),
                             resample=Image.BILINEAR)


# operation that overlaps with ImageNet-C's test set
def color(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Color(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def contrast(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Contrast(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def brightness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Brightness(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def sharpness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Sharpness(pil_img).enhance(level)


def normalize(image):
    """Normalize input image channel-wise to zero mean and unit variance."""
    return image - 127


def apply_op(image, op, severity):
    # image = np.clip(image, 0, 255)
    pil_img = Image.fromarray(image)  # Convert to PIL.Image
    pil_img = op(pil_img, severity)
    return np.asarray(pil_img)


def augment_and_mix(image, severity=3, width=3, depth=-1, alpha=1.):
    """Perform AugMix augmentations and compute mixture.
    Args:
    image: Raw input image as float32 np.ndarray of shape (h, w, c)
    severity: Severity of underlying augmentation operators (between 1 to 10).
    width: Width of augmentation chain
    depth: Depth of augmentation chain. -1 enables stochastic depth uniformly
      from [1, 3]
    alpha: Probability coefficient for Beta and Dirichlet distributions.
    Returns:
    mixed: Augmented and mixed image.
    """

    augmentations = [
        autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
        translate_x, translate_y
    ]

    ws = np.float32(np.random.dirichlet([alpha] * width))
    m = np.float32(np.random.beta(alpha, alpha))

    mix = np.zeros_like(image).astype(np.float32)
    for i in range(width):
        image_aug = image.copy()
        depth = depth if depth > 0 else np.random.randint(1, 4)
        for _ in range(depth):
            op = np.random.choice(augmentations)
            image_aug = apply_op(image_aug, op, severity)
        # Preprocessing commutes since all coefficients are convex
        mix += ws[i] * image_aug
        # mix += ws[i] * normalize(image_aug)

    mixed = (1 - m) * image + m * mix
    # mixed = (1 - m) * normalize(image) + m * mix
    return mixed


class RandomAugMix():

    def __init__(self, prob=0.1, severity=2, width=3, depth=2, alpha=1.):
        self.prob = prob
        self.severity = severity
        self.width = width
        self.depth = depth
        self.alpha = alpha

    def __call__(self, img):
        if np.random.uniform() > self.prob:
            return img
        tmp = (img * 255).astype(np.uint8) if img.dtype != "uint8" else img
        out = augment_and_mix(tmp, self.severity, self.width, self.depth, self.alpha)
        if type(img) is np.ndarray:
            if img.dtype != "uint8":
                out = (out / 255).astype(np.float64)
        return out
