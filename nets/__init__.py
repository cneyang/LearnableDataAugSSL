from copy import deepcopy

import torch
import torch.nn as nn
from .wideresnet import WideResNet
from .sslgan import Discriminator
from .wrnperceptualloss import *

class Encoder(nn.Module):
    def __init__(self, model_name="wresnet40_2", dropout_rate=0.0):
        super(Encoder, self).__init__()
        if model_name == "wresnet40_2":
            self.encoder = WideResNet(40, 2, dropout_rate)
        if model_name == "wresnet28_10":
            self.encoder = WideResNet(28, 10, dropout_rate)
        elif model_name == "resnet50":
            pass
        self.out_features = self.encoder.out_features

    def forward(self, x):
        return self.encoder(x)


class Classifier(nn.Module):
    def __init__(self, in_features, num_classes, num_ops):
        super(Classifier, self).__init__()
        self.classifier = nn.Linear(in_features, num_classes)
        self.discriminator = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(),
            nn.Linear(in_features, num_ops),
        )

    def forward(self, x):
        return self.classifier(x), self.discriminator(x)


class Policy(nn.Module):
    def __init__(self, encoder, num_ops):
        super(Policy, self).__init__()
        self.num_ops = num_ops
        self.encoder = deepcopy(encoder)
        self.policy = nn.Linear(encoder.out_features, num_ops)

    def forward(self, x):
        x = self.encoder(x)
        aug_params = self.policy(x)
        return aug_params
