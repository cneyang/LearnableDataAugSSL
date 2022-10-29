import torch
from torch import nn, Tensor
from torch.distributions import Categorical

from .operations import *


class Augmenter(nn.Module):
    def __init__(self, after_operations):
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
        self.after_operations = after_operations

    def apply_operation(self, input: Tensor, mag: Tensor) -> Tensor:
        for i, op in enumerate(self.operations):
            input = op(input, mag[:, i])
        return input

    def forward(self, input: Tensor, mag: Tensor) -> Tensor:
        mag = mag.sigmoid()
        input = self.apply_operation(input, mag)
        return self.after_operations(input)
