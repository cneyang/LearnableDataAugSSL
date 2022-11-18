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
                TranslateX(magnitude_scale=0.3),
                TranslateY(magnitude_scale=0.3),
                Rotate(),
                HorizontalFlip(),
                Invert(),
                Solarize(),
                Posterize(),
                Contrast(magnitude_range=(0.05, 0.95)),
                Saturate(),
                Brightness(magnitude_range=(0.05, 0.95)),
                Sharpness(magnitude_range=(0.05, 0.95)),
                AutoContrast(),
                Equalize(),
                Cutout(),
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
    
    # Potential Augmentations to test
    # transforms.RandomCrop(size=32, padding=[0, 2, 3, 4])
    # transforms.FiveCrop(size=32)
    # transforms.TenCrop(size=32, vertical_flip=False)
    # transforms.CenterCrop(10)
    # transforms.RandomAdjustSharpness(sharpness_factor = random.choice[0.2,0.5,0.8,1.3,1.6,2], p=0.5)
    # transforms.Grayscale(num_output_channels=3)  or transforms.RandomGrayscale(p=0.2)
    # transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)
    # transforms.RandomPerspective(distortion_scale=0.5, p=0.5)
    # transforms.RandomResizedCrop(size=32, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333)
    # transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
    # transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)
