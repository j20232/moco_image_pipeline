import torch


def accuracy(y, true_label):
    pred_label = torch.argmax(y, dim=1)
    return torch.tensor((pred_label == true_label).sum().item() / len(y))
