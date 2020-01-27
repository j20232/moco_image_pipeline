from yacs.config import CfgNode as CN

_C = CN()

_C.N_GRAPHEME = 160
_C.N_VOWEL = 11
_C.N_CONSONANT = 7
_C.MODEL = "se_resnext50_32x4d"
_C.PRETRAINED = "null"

_C.NN = CN()
_C.NN.OPTIM_NAME = "Adam"
_C.NN.OPTIM_PARAMS = []
_C.NN.SCHEDULER_NAME = "ReduceLROnPlateau"
_C.NN.SCHEDULER_PARAMS = []


def get_cfg():
    return _C.clone()
