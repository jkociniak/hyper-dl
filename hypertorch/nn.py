import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
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


class HyperbolicFFN(nn.Sequential):
    """
    Hyperbolic feedforward network.
    Optional skip connections (applies to all layers except last, for which user can choose).
    """

    def __init__(self,
                 input_dim,
                 hidden_dims,
                 output_dim,
                 activations=None,
                 skips=False,
                 apply_to_last_layer=False,
                 **kwargs):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.activation = activations
        self.skips = skips
        self.apply_to_last_layer = apply_to_last_layer

        super().__init__(self.get_ffn_layers())

    def get_ffn_layers(self):
        layers = OrderedDict()

        def label_l(i):
            return f'HyperbolicLinear{i}'

        def label_ls(i):
            return f'HyperbolicLinearSkip{i}'

        prev = self.input_dim  # start with input_dim
        for i, hidden_dim in enumerate(self.hidden_dims):
            if self.skips:
                layers[label_ls(i)] = HyperbolicLinearSkip(prev, hidden_dim, self.activation)
            else:
                layers[label_l(i)] = HyperbolicLinear(prev, hidden_dim, self.activation)

            prev = hidden_dim

        n = len(self.hidden_dims)
        if self.apply_to_last_layer:
            if self.skips:
                layers[label_ls(n)] = HyperbolicLinearSkip(prev, self.output_dim, self.activation)
            else:
                layers[label_l(n)] = HyperbolicLinear(prev, self.output_dim, self.activation)

        else:
            layers[label_l(n)] = HyperbolicLinear(prev, self.output_dim, self.activation)

        return layers


class HyperbolicLinear(nn.Module):
    """
    Wrapper for HyperbolicLinear with activation.
    """
    def __init__(self, input_dim, output_dim, activation):
        super().__init__()
        self.fc = MobiusLinear(input_dim, output_dim)
        if activation == 'relu':
            self.activation = MobiusReLU()
        else:
            raise NotImplementedError(f'activation {activation} in HyperbolicLinear is not implemented!')

    def forward(self, x):
        return self.activation(self.fc(x))


class HyperbolicLinearSkip(nn.Module):
    """
    Wrapper for HyperbolicLinear with skip connection after activation.
    """
    def __init__(self, input_dim, output_dim, activation):
        super().__init__()
        self.fc = MobiusLinear(input_dim, output_dim)
        if activation == 'relu':
            self.activation = MobiusReLU()
        else:
            raise NotImplementedError(f'activation {activation} in LinearSkip is not implemented!')

    def forward(self, x):
        y = self.fc(x)
        y = self.activation(y)
        return y + x


