from .basic import CenterOrRandomCrop, GaussianNoise
from .album import RandomBlur, RandomNoise, CoarseDropout, GridDistortion, ElasticTransform, RandomBrightnessContrast, ShiftScaleRotate
from .opencv import RandomMorphology

modules = [
    "CenterOrRandomCrop", "GaussianNoise",
    "RandomBlur", "RandomNoise", "CoarseDropout", "GridDistortion",
    "ElasticTransform", "RandomBrightnessContrast", "ShiftScaleRotate",
    "RandomMorphology"
]
