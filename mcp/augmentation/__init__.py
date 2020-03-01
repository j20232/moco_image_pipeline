from .basic import GaussianNoise
from .album import RandomBlur, RandomNoise, CoarseDropout, GridDistortion, ElasticTransform, RandomBrightnessContrast, ShiftScaleRotate, RandomAugMix
from .opencv import RandomMorphology
from .pil_aug import CenterOrRandomCrop

# PIL
pil_modules = [
    "CenterOrRandomCrop"
]


# np.ndarray
modules = [
    "GaussianNoise",
    "RandomBlur", "RandomNoise", "CoarseDropout", "GridDistortion",
    "ElasticTransform", "RandomBrightnessContrast", "ShiftScaleRotate",
    "RandomMorphology", "RandomAugMix"
]
