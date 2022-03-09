import torch.nn as nn
import torch.nn.functional as F
from .math import *
import math


class MobiusLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=False)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        if self.bias is not None:
            bound = 1 / math.sqrt(in_features) if in_features > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        zero = torch.zeros_like(x)
        x = log_map(zero, x)
        x = self.fc(x)
        x = exp_map(zero, x)

        if self.bias is not None:
            x = mobius_addition(x, self.bias)

        return x


class HyperbolicConcat(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.mfc1 = MobiusLinear(in_features[0], out_features, False)
        self.mfc2 = MobiusLinear(in_features[1], out_features, False)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        if self.bias is not None:
            in_dim = sum(in_features)
            bound = 1 / math.sqrt(in_dim) if in_dim > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x1, x2):
        x1 = self.mfc1(x1)
        x2 = self.mfc2(x2)
        x = mobius_addition(x1, x2)

        if self.bias is not None:
            x = mobius_addition(x, self.bias)

        return x


class MobiusReLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = mobius(F.relu)(x)
        return x
