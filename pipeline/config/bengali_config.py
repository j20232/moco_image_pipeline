from yacs.config import CfgNode as CN

_C = CN()

_C.N_GRAPHEME = 160
_C.N_VOWEL = 11
_C.N_CONSONANT = 7

_C.MODEL = "se_resnext50_32x4d"


def get_cfg():
    return _C.clone()
