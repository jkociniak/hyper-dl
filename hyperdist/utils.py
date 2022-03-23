import torch
import numpy as np
from hypertorch.math import mobius_addition_np
import random


def reset_rngs(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def MSE(y, y_pred, reduction='mean'):
    s = (y - y_pred)**2
    if reduction == 'mean':
        return torch.mean(s)
    elif reduction == 'sum':
        return torch.sum(s)
    else:
        return s


def MAPE(y, y_pred, reduction='mean'):
    s = torch.abs((y - y_pred) / y)
    if reduction == 'mean':
        return torch.mean(s)
    elif reduction == 'sum':
        return torch.sum(s)
    else:
        return s


def hyperbolic_dist_np(x, y):
    return 2 * np.arctanh(np.linalg.norm(mobius_addition_np(-x, y)))