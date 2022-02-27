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
    Uses one activation function for all layers.
    Last layer is not followed by activation.
    Optional batch norm.
    """
    def __init__(self, input_dim, hidden_dims, output_dim, activations='relu', batch_norm=False, skips=True):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.activation = activations
        self.batch_norm = batch_norm
        self.skips = skips

        layers = self.build_layers()
        super().__init__(layers)


    def build_layers(self):
        layers = OrderedDict()

        label_fc = lambda i: f'fc{i}'
        label_act = lambda i: f'activation{i}'
        label_bn = lambda i: f'batch{i}'

        n_hidden = len(self.hidden_dims)
        prev = self.input_dim  # start with input_dim
        for i in range(n_hidden):
            layers[label_fc(i)] = nn.Linear(prev, self.hidden_dims[i])

            if self.activation == 'relu':
                layers[label_act(i)] = nn.ReLU()
            else:
                raise NotImplementedError(f'activation {self.activation} in EuclideanFFN is not implemented!')

            if self.batch_norm:
                layers[label_bn(i)] = nn.BatchNorm1d(num_features=self.hidden_dims[i])
            prev = self.hidden_dims[i]

        layers[label_fc(n_hidden)] = nn.Linear(prev, self.output_dim)
        return layers