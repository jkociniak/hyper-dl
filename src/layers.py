import torch
import torch.nn as nn
from collections import OrderedDict


def hard_clipping(r, x):
    x_norm = torch.linalg.norm(x).item()
    to_compare = torch.Tensor([1, r / x_norm])
    scale = torch.min(to_compare)
    return x * scale


class EuclideanFFN(nn.Sequential):
    """
    Ordinary feedforward network.
    Optional batch norm and skip connections (applies to all layers except last, for which user can choose).
    If parameter clipping_r is provided, then the network is applied in tangent space and clipping_r describes
    the clipping parameter from https://arxiv.org/pdf/2107.11472.pdf.
    """
    def __init__(self,
                 input_dim,
                 hidden_dims,
                 output_dim,
                 activations='relu',
                 batch_norm=False,
                 skips=False,
                 apply_to_last_layer=False,
                 **kwargs):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.activation = activations
        self.batch_norm = batch_norm
        self.skips = skips
        self.apply_to_last_layer = apply_to_last_layer

        super().__init__(self.get_ffn_layers())


    def get_ffn_layers(self):
        layers = OrderedDict()

        label_l = lambda i: f'Linear{i}'
        label_ls = lambda i: f'LinearSkip{i}'
        label_bn = lambda i: f'BatchNorm{i}'

        prev = self.input_dim  # start with input_dim
        for i, hidden_dim in enumerate(self.hidden_dims):
            if self.skips:
                layers[label_ls(i)] = LinearSkip(prev, hidden_dim, self.activation)
            else:
                layers[label_l(i)] = Linear(prev, hidden_dim, self.activation)

            if self.batch_norm:
                layers[label_bn(i)] = nn.BatchNorm1d(hidden_dim)
            prev = hidden_dim

        n = len(self.hidden_dims)
        if self.apply_to_last_layer:
            if self.skips:
                layers[label_ls(n)] = LinearSkip(prev, self.output_dim, self.activation)
            else:
                layers[label_l(n)] = Linear(prev, self.output_dim, self.activation)

            if self.batch_norm:
                layers[label_bn(n)] = nn.BatchNorm1d(self.output_dim)
        else:
            layers[label_l(n)] = nn.Linear(prev, self.output_dim)

        return layers


class Linear(nn.Module):
    """
    Wrapper for nn.Linear with activation.
    """
    def __init__(self, input_dim, output_dim, activation):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        if activation == 'relu':
            self.activation = nn.ReLU()
        else:
            raise NotImplementedError(f'activation {activation} in Linear is not implemented!')

    def forward(self, x):
        return self.activation(self.fc(x))


class LinearSkip(nn.Module):
    """
    Linear layer with skip connection after activation. (https://arxiv.org/pdf/1701.09175.pdf)
    """
    def __init__(self, input_dim, output_dim, activation):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        if activation == 'relu':
            self.activation = nn.ReLU()
        else:
            raise NotImplementedError(f'activation {activation} in LinearSkip is not implemented!')

    def forward(self, x):
        y = self.fc(x)
        y = self.activation(y)
        return y + x
