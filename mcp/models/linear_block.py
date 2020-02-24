import torch
from torch import nn
import torch.nn.functional as F


def residual_add(lhs, rhs):
    lhs_ch, rhs_ch = lhs.shape[1], rhs.shape[1]
    if lhs_ch < rhs_ch:
        out = lhs + rhs[:, :lhs_ch]
    elif lhs_ch > rhs_ch:
        out = torch.cat([lhs[:, :rhs_ch] + rhs, lhs[:, rhs_ch:]], dim=1)
    else:
        out = lhs + rhs
    return out


class LinearBlock(nn.Module):

    def __init__(self, in_features, out_features, bias=True,
                 use_bn=True, activation=F.relu, dropout_ratio=-1, residual=False,):
        super(LinearBlock, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.bn = nn.BatchNorm1d(out_features) if use_bn else (lambda x : x)
        self.dropout = nn.Dropout(p=dropout_ratio) if dropout_ratio > 0 else (lambda x : x)
        self.activation = activation if activation else (lambda x : x)
        self.use_bn = use_bn if use_bn else (lambda x : x)
        self.residual = residual if residual else (lambda h, x : h)

    def __call__(self, x):
        h = self.linear(x)
        h = self.bn(h)
        h = self.activation(h)
        h = residual_add(h, x)
        h = self.dropout(h)
        return h
