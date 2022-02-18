import torch
import numpy as np

def mobius_addition(x, y):
    dot_xy = torch.sum(x * y, dim=1, keepdim=True)
    dot_xx = torch.sum(x * x, dim=1, keepdim=True)
    dot_yy = torch.sum(y * y, dim=1, keepdim=True)
    numerator = (1 + 2 * dot_xy + dot_yy) * x + (1 - dot_xx) * y
    denominator = 1 + 2 * dot_xy + dot_xx * dot_yy
    return numerator / denominator

def mobius_addition_np(x, y):
    dot_xy = np.sum(x * y)
    dot_xx = np.sum(x * x)
    dot_yy = np.sum(y * y)
    numerator = (1 + 2 * dot_xy + dot_yy) * x + (1 - dot_xx) * y
    denominator = 1 + 2 * dot_xy + dot_xx * dot_yy
    return numerator / denominator

def exp_map(x, v):
    v_norm = torch.linalg.norm(v, dim=1, keepdim=True)
    cf = conformal_factor(x)
    scalar_factor = torch.tanh(cf * v_norm / 2) / v_norm
    return mobius_addition(x, scalar_factor * v)


def log_map(x, y):
    v = mobius_addition(-x, y)
    v_norm = torch.linalg.norm(v, dim=1, keepdim=True)
    cf = conformal_factor(x)
    scalar_factor = 2 * torch.arctanh(v_norm) / (cf * v_norm)
    return scalar_factor * v


def conformal_factor(x):
    return 2 / (1 - torch.linalg.norm(x) ** 2)


def hyperbolic_dist(x):
    norm = torch.linalg.norm(x, dim=1)
    res = 2 * torch.arctanh(norm)
    return res