import torch
import torch.nn as nn
from .math import *


class MobiusLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x):
        zero = torch.zeros_like(x)
        x = log_map(zero, x)
        x = self.fc(x)
        x = exp_map(zero, x)
        return x


class HyperbolicConcat(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.mfc1 = MobiusLinear(in_features[0], out_features)
        self.mfc2 = MobiusLinear(in_features[1], out_features)

    def forward(self, x1, x2):
        x1 = self.mfc1(x1)
        x2 = self.mfc2(x2)
        res = mobius_addition(x1, x2)
        return res