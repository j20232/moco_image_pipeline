
from .album import RandomBlur
from .album import GaussNoise, MultiplicativeNoise
from .album import GridDistortion, ElasticTransform, ShiftScaleRotate
from .album import HueSaturationValue, RandomBrightnessContrast, RandomCLAHE
from .album import CoarseDropout
from .album import RandomAugMix

from .opencv import RandomMorphology
from .pil_aug import CenterOrRandomCrop
from .mixup import Mixup

from .kero_aug import RandomProjective, RandomPerspective, RandomRotate
from .kero_aug import RandomScale, RandomShearX, RandomShearY, RandomStretchX, RandomStretchY
from .kero_aug import RandomGridDistortion, RandomCustomGridDistortion
from .kero_aug import RandomContrast, RandomBlockFade
from .kero_aug import RandomErode, RandomDilate, RandomSpinkle, RandomNoise, RandomLine

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
    "RandomMorphology",
    "RandomProjective", "RandomPerspective", "RandomRotate",
    "RandomScale", "RandomShearX", "RandomShearY", "RandomStretchX", "RandomStretchY",
    "RandomGridDistortion", "RandomCustomGridDistortion",
    "RandomContrast", "RandomBlockFade",
    "RandomErode", "RandomDilate", "RandomSpinkle", "RandomNoise", "RandomLine"
]
