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
                 skips=True,
                 apply_to_last_layer=False,
                 clipping_r=None,
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

        label_lb = lambda i: f'LinearBlock{i}'

        prev = self.input_dim  # start with input_dim
        for i, hidden_dim in enumerate(self.hidden_dims):
            layers[label_lb(i)] = LinearBlock(prev, hidden_dim, self.activation, self.batch_norm, self.skips)
            prev = hidden_dim

        n = len(self.hidden_dims)
        if self.apply_to_last_layer:
            layers[label_lb(n)] = LinearBlock(prev, self.output_dim, self.activation, self.batch_norm, self.skips)
        else:
            layers[label_lb(n)] = nn.Linear(prev, self.output_dim)

        return layers


class LinearBlock(nn.Module):
    """
    Linear block followed by optional skip connection and batch norm.
    Skip connections are applied after activation. (https://arxiv.org/pdf/1701.09175.pdf)
    Batch norm is applied after residual connection.
    """
    def __init__(self, input_dim, output_dim, activation, batch_norm, skip):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)

        if activation == 'relu':
            self.activation = nn.ReLU()
        else:
            raise NotImplementedError(f'activation {activation} in LinearBlock is not implemented!')

        if skip:
            assert input_dim == output_dim, "Skip connection in LinearBlock requires equal input and output dimensions!"
        self.skip = skip

        self.bn = nn.BatchNorm1d(num_features=output_dim) if batch_norm else None


    def forward(self, x):
        y = self.fc(x)
        y = self.activation(y)
        if self.skip:
            y = y + x
        if self.bn is not None:
            y = self.bn(y)
        return y