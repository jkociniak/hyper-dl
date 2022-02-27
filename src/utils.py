import torch
import numpy as np
from hypertorch.math import mobius_addition_np

def mape(y, y_pred):
    return torch.mean(torch.abs((y - y_pred) / y))

def hyperbolic_dist_np(x, y):
    return 2 * np.arctanh(np.linalg.norm(mobius_addition_np(-x, y)))


