
from .album import RandomBlur # Blur
from .album import GaussNoise, MultiplicativeNoise # Noise
from .album import GridDistortion, ElasticTransform, ShiftScaleRotate # Distortion
from .album import HueSaturationValue, RandomBrightnessContrast, RandomCLAHE # Histogram
from .album import CoarseDropout # Removal
from .album import RandomAugMix # Augmix

from .opencv import RandomMorphology
from .pil_aug import CenterOrRandomCrop
from .mixup import Mixup

# PIL
pil_modules = [
    "CenterOrRandomCrop"
]


# np.ndarray
modules = [
    "RandomBlur",
    "GaussNoise", "MultiplicativeNoise",
    "GridDistortion", "ElasticTransform", "ShiftScaleRotate",
    "HueSaturationValue", "RandomBrightnessContrast", "RandomCLAHE",
    "CoarseDropout",
    "RandomAugMix",
    "RandomMorphology"
]
