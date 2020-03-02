# Reference:
# - https://github.com/snakers4/MNASNet-pytorch-1/blob/master/mixup.py
# - https://github.com/snakers4/MNASNet-pytorch-1/blob/master/utils/cross_entropy.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta


def onehot(indexes, N=None, ignore_index=None):
    """
    Creates a one-representation of indexes with N possible entries
    if N is not specified, it will suit the maximum index appearing.
    indexes is a long-tensor of indexes
    ignore_index will be zero in onehot representation
    """
    if N is None:
        N = indexes.max() + 1
    sz = list(indexes.size())
    output = indexes.new().byte().resize_(*sz, N).zero_()
    output.scatter_(-1, indexes.unsqueeze(-1), 1)
    if ignore_index is not None and ignore_index >= 0:
        output.masked_fill_(indexes.eq(ignore_index).unsqueeze(-1), 0)
    return output


def mixup(x, y, num_classes, gamma=0.2, smooth_eps=0.1):
    if gamma == 0 and smooth_eps == 0:
        return x, y;
    m = Beta(torch.tensor([gamma]), torch.tensor([gamma]))
    lambdas = m.sample([x.size(0), 1, 1]).to(x)
    my = onehot(y, num_classes).to(x)
    true_class, false_class = 1. - smooth_eps * num_classes / (num_classes - 1), smooth_eps / (num_classes - 1)
    my = my * true_class + torch.ones_like(my) * false_class
    perm = torch.randperm(x.size(0))
    x2 = x[perm]
    y2 = my[perm]
    return x * (1 - lambdas) + x2 * lambdas, my * (1 - lambdas) + y2 * lambdas


class Mixup(torch.nn.Module):
    def __init__(self, num_classes=1000, gamma=0.2, smooth_eps=0.1):
        super(Mixup, self).__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.smooth_eps = smooth_eps

    def forward(self, input, target):
        return mixup(input, target, self.num_classes, self.gamma, self.smooth_eps)
