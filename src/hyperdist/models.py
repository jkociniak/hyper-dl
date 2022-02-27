import torch
import torch.nn as nn
from functools import partial
from hypertorch.math import exp_map, log_map
from hypertorch.nn import HyperbolicConcat
from .layers import EuclideanFFN, hard_clipping


def build_model(name, **kwargs):
    if name == 'EncoderHeadModel':
        return EncoderHeadModel(**kwargs)
    elif name == 'EuclideanFFNModel':
        return EuclideanFFNModel(**kwargs)
    else:
        raise NotImplementedError(f'the model {name} is not implemented!')


class EuclideanFFNModel(nn.Module):
    def __init__(self, input_dim, hidden_dims, activations, batch_norm=True, skips=True, **kwargs):
        super().__init__()
        self.concat_layer = nn.Linear(2 * input_dim, hidden_dims[0])
        self.ffn = EuclideanFFN(hidden_dims[0], hidden_dims[1:], 1, activations, batch_norm, skips)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        x = self.concat_layer(x)
        x = self.ffn(x)
        x = x.squeeze()
        return x


class EncoderHeadModel(nn.Module):
    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = self.build_encoder(**encoder)
        self.head = self.build_head(**head)

    def build_encoder(self, name, **kwargs):
        if name == 'EuclideanFFNEncoder':
            return EuclideanFFNEncoder(**kwargs)
        else:
            raise NotImplementedError(f'the encoder {name} is not implemented!')

    def build_head(self, name, **kwargs):
        if name == 'EuclideanFFN':
            return EuclideanFFN(**kwargs)
        else:
            raise NotImplementedError(f'the head {name} is not implemented!')

    def forward(self, x1, x2):
        embeddings = self.encoder(x1, x2)
        distance = self.head(embeddings)
        distance = distance.squeeze()
        return distance


class EuclideanFFNEncoder(nn.Module):
    def __init__(self,
                 io_dim,
                 hidden_dims,
                 clipping_r,
                 activations,
                 batch_norm,
                 start_with_log=False):
        super().__init__()
        self.start_with_log = start_with_log
        self.concat_layer = nn.Linear(2 * io_dim, hidden_dims[0])
        self.ffn = EuclideanFFN(hidden_dims[0], hidden_dims[1:], io_dim, activations, batch_norm)

        self.clipping_r = clipping_r
        self.clipping = partial(hard_clipping, clipping_r)


    def forward(self, x1, x2):
        if self.start_with_log:
            x1 = log_map(torch.zeros_like(x1), x1)
            x2 = log_map(torch.zeros_like(x2), x2)
        x = torch.concat((x1, x2), dim=1)
        x = self.concat_layer(x)
        x = self.ffn(x)
        x = self.clipping(x)
        x = exp_map(torch.zeros_like(x), x)
        return x


class HyperbolicFFNEncoder(nn.Module):
    def __init__(self, hyperbolic_dim):
        super().__init__()
        dims = [hyperbolic_dim] * 2
        self.hyperbolic_concat = HyperbolicConcat(dims, hyperbolic_dim)

    def forward(self, x1, x2):
        x = self.hyperbolic_concat(x1, x2)
        return x