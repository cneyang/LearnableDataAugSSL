import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.distributions import Categorical

from .operations import *


class Augmenter(nn.Module):
    def __init__(self, mean, std, before_ops=None, after_ops=None):
        super(Augmenter, self).__init__()
        self.operations = nn.ModuleList(
            [
                # what more...?
                ShearX(),
                ShearY(),
                TranslateX(),
                TranslateY(),
                Rotate(),
                HorizontalFlip(),
                Invert(),
                Solarize(),
                Posterize(),
                Contrast(),
                Saturate(),
                Brightness(),
                Sharpness(),
                AutoContrast(),
                Equalize(),
            ]
        )
        mean = torch.Tensor(mean).view(1, 3, 1, 1)
        std = torch.Tensor(std).view(1, 3, 1, 1)
        self.mean = nn.parameter.Parameter(mean, requires_grad=False)
        self.std = nn.parameter.Parameter(std, requires_grad=False)
        self.before_ops = before_ops
        self.after_ops = after_ops

    def apply_operation(self, input: Tensor, mag: Tensor) -> Tensor:
        for i, op in enumerate(self.operations):
            input = op(input, mag[:, i])
        return input

    def forward(self, input: Tensor, mag: Tensor) -> Tensor:
        mag = mag.sigmoid()
        input = input * self.std + self.mean
        if self.before_ops is not None:
            input = self.before_ops(input)
        input = self.apply_operation(input, mag)
        if self.after_ops is not None:
            input = self.after_ops(input)
        input = (input - self.mean) / self.std
        return input
