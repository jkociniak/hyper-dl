import torch
from torch.optim import Optimizer
from .math import conformal_factor, exp_map


class RiemannianSGD(Optimizer):
    """
    Implements Riemannian SGD.
    """
    def __init__(self, params, lr=1e-3, curv=1):
        assert lr > 0
        assert curv > 0
        defaults = {'lr': lr, 'curv': torch.tensor(curv)}
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            curv = group['curv']

            for p in group['params']:
                if p.grad is not None:
                    grad = p.grad
                    riemannian_grad = grad.mul(1/(conformal_factor(p, curv) ** 2))
                    new_p = exp_map(p, -lr * riemannian_grad, curv)
                    p.data.copy_(new_p)

        return loss
