import torch
import torch.nn as nn
from hypertorch.nn import HyperbolicConcat
from layers import EuclideanFFN, hard_clipping
from hypertorch.math import exp_map, log_map
from functools import partial


def build_model(name, **kwargs):
    """
    Function used to build model based on its name.
    :param name: name of the model
    :param kwargs: parameters to pass to the model constructor
    :return: object containing built model
    """
    if name == 'EncoderHeadModel':
        return EncoderHeadModel(**kwargs)
    elif name == 'EuclideanFFNModel':
        return EuclideanFFNModel(**kwargs)
    elif name == 'HyperbolicFFNModel':
        return HyperbolicFFNModel(**kwargs)
    else:
        raise NotImplementedError(f'the model {name} is not implemented!')


class EuclideanFFNModel(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dims,
                 **kwargs):
        super().__init__()
        self.concat_layer = nn.Linear(2 * input_dim, hidden_dims[0])
        self.ffn = EuclideanFFN(hidden_dims[0], hidden_dims[1:], 1, **kwargs)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        x = self.concat_layer(x)
        x = self.ffn(x)
        x = x.squeeze()
        return x


# TODO finish
class EncoderHeadModel(nn.Module):
    def __init__(self, input_dim, encoder, head, encoder_type, head_type, encoder_r):
        super().__init__()
        encoder['input_dim'] = input_dim
        encoder['output_dim'] = input_dim
        head['input_dim'] = input_dim
        head['output_dim'] = 1
        self.encoder = self.build_encoder(**encoder)
        self.head = self.build_head(**head)
        if encoder_r is not None:
            assert encoder_r > 0, "Clipping parameter should be positive!"
            self.clipping = partial(hard_clipping, encoder_r)

    def forward(self, x1, x2):
        if self.clipping_r is not None:
            x1 = log_map(torch.zeros_like(x1), x1)
            x2 = log_map(torch.zeros_like(x2), x2)

        e = self.encoder(x1, x2)

        if self.clipping_r is not None:
            e = self.clipping(e)
            e = exp_map(torch.zeros_like(e), e)

        d = self.head(e)
        d = d.squeeze()
        return d

    def build_encoder(self, name, **kwargs):
        if name == 'EuclideanFFN':
            return EuclideanFFN(**kwargs)
        else:
            raise NotImplementedError(f'the encoder {name} is not implemented!')

    def build_head(self, name, **kwargs):
        if name == 'EuclideanFFN':
            return EuclideanFFN(**kwargs)
        else:
            raise NotImplementedError(f'the head {name} is not implemented!')


class HyperbolicFFNModel(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dims,
                 **kwargs):
        super().__init__()
        dims = [input_dim] * 2
        self.hyperbolic_concat = HyperbolicConcat(dims, input_dim)

    def forward(self, x1, x2):
        x = self.hyperbolic_concat(x1, x2)
        x = self.ffn(x)
        x = x.squeeze()
        return x