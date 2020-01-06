import pretrainedmodels
import torch
import torch.nn as nn
import torchvision
from efficientnet_pytorch import EfficientNet


class MultiSegModels:
    def __init__(self, models):
        self.models = models

    def __call__(self, x):
        res = []
        x = x.cuda()
        with torch.no_grad():
            for m in self.models:
                res.append(m(x)[:, -8:, :, :])
        res = torch.stack(res)
        return torch.mean(res, dim=0)
