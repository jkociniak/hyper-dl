import torch
import numpy as np
from hypertorch.math import mobius_addition_np
from scipy.special import gamma
from scipy.interpolate import interp1d
import random
import os

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


def hyperbolic_dist_np(x, y, c):
    mobadd = mobius_addition_np(-x, y, c)
    d = np.linalg.norm(mobadd)
    max_radius = 1/c
    d = max_radius - 1e-8 if d > max_radius - 1e-8 else d
    return 2/np.sqrt(c) * np.arctanh(np.sqrt(c) * d)


def hyperbolic_volume_inverse(n, c, min_r=1e-3, max_r=5.31, n_samples=100000):
    #rs = np.logspace(np.log(min_r), np.log(max_r), num=n_samples, base=np.e)
    # rs_small = np.linspace(9e-3, 11e-3, 3*n_samples)
    # rs_med = np.linspace(11e-3, 21e-3, 3*n_samples)
    # rs_big = np.linspace(21e-3, max_r, 3*n_samples)
    # rs = np.concatenate((rs_small, rs_med, rs_big))
    rs = np.linspace(min_r, max_r, n_samples)
    max_v = hyperbolic_volume(n, c, max_r)
    min_v = hyperbolic_volume(n, c, min_r)
    vs = np.array([(hyperbolic_volume(n, c, r) - min_v)/(max_v - min_v) for r in rs])
    _, idx = np.unique(vs, return_index=True)
    return interp1d(vs[idx], rs[idx], kind='linear')


def e(n):
    num = 2 * np.pi ** ((n+1)/2)
    denom = gamma((n+1)/2)
    return num/denom


def hyperbolic_volume(n, c, r):
    # n: dimension of the hyperbolic space
    # r: radius of the sphere
    # c: curvature of the hyperbolic space
    #
    # we want to compute V_{n, c}(r) = e_{n-1} \int_0^r (sinh(t*sqrt(c))/sqrt(c))^{n-1} dt
    # where e_{n-1} = 2pi^{n/2}/gamma(n/2) is a volume of unit ball in ndim eucl space
    # so we use the formula
    #
    # sinh(t) = (e^x - e^{-x})/2
    #
    # and the binomial theorem to derive
    #
    # \int_0^r sinh(t)^n dt =
    # 1/(2sqrt(c))^{n-1} \sum_{k=0}^{n-1} (n-1 choose k) (-1)^{n-1-k} (e^{(2k-n)r sqrt(c)} - 1)/((2k-n)sqrt(c))
    c = np.sqrt(c)

    sign = 1 if (n-1) % 2 == 0 else -1
    sum = sign * np.expm1(-(n-1) * r * c) / ((-(n-1)) * c)

    coeff = 1
    for k in range(1, n):
        sign = -sign
        coeff *= (n-k)/k
        if 2*k - (n-1) != 0:
            sum += sign * coeff * np.expm1((2*k - (n-1)) * r * c) / ((2*k - (n-1)) * c)
        else:
            sum += sign * coeff * r
    return e(n-1) * sum / ((2*c)**(n-1))
