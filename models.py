from hypertorch.nn import DoubleInputHyperbolicFFN, SemiRiemannianModule
from layers import DoubleInputEuclideanFFN, build_layer
from hypertorch.math import exp_map0, log_map0
import torch


def isfinite_check(var, layer):
    if not var.isfinite().all():
        raise RuntimeError(f'Invalid value (inf/-inf/nan) detected after {layer}')


class HyperbolicFFNModel(DoubleInputHyperbolicFFN):
    def __init__(self,
                 input_dim,
                 hidden_dims,
                 **kwargs):
        super().__init__(input_dim, hidden_dims, 1, **kwargs)


class EuclideanFFNModel(DoubleInputEuclideanFFN):
    def __init__(self,
                 input_dim,
                 hidden_dims,
                 **kwargs):
        super().__init__(input_dim, hidden_dims, 1, **kwargs)


class EncoderHeadModel(SemiRiemannianModule):
    def __init__(self, input_dim, inter_dim, encoder, head):
        super().__init__()
        encoder['input_dim'] = input_dim
        encoder['output_dim'] = inter_dim
        self.encoder = build_layer(**encoder)

        head['input_dim'] = inter_dim
        head['output_dim'] = 1
        self.head = build_layer(**head)

        # if encoder_r is not None:
        #     assert encoder_r > 0, "Clipping parameter should be positive!"
        #     self.clipping = partial(hard_clipping, encoder_r)

    def forward(self, x1, x2):
        # if self.clipping_r is not None:
        #     x1 = log_map0(x1)
        #     x2 = log_map0(x2)

        e = self.encoder(x1, x2)

        isfinite_check(e, 'encoder')

        e = log_map0(e, torch.tensor(1))

        isfinite_check(e, 'log map')

        # # should I go back into tangent space?
        # # We start with hyperbolic input. If
        # if isinstance(self.encoder, EuclideanFFN) and isinstance(self.head, HyperbolicFFN):
        #     e =

        # if self.clipping_r is not None:
        #     e = self.clipping(e)
        #     e = exp_map0(e)

        d = self.head(e)

        d = d.squeeze()
        return d


name2model = {
    'EncoderHeadModel': EncoderHeadModel,
    'EuclideanFFNModel': EuclideanFFNModel,
    'HyperbolicFFNModel': HyperbolicFFNModel,
}


def build_model(name, **kwargs):
    """
    Function used to build model based on its name.
    :param name: name of the model
    :param kwargs: parameters to pass to the model constructor
    :return: object containing built model
    """
    try:
        return name2model[name](**kwargs)
    except KeyError:
        raise NotImplementedError(f'the model {name} is not implemented!')