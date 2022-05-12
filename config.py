import torch.nn as nn
import torch.optim as optim
import hypertorch.optim as roptim

from models import build_model
from loops import train_loop
from utils import MSE, MAPE, reset_rngs

from omegaconf import OmegaConf


def run_training(epochs,
                 seed,
                 model,
                 optimizer,
                 scheduler,
                 r_optimizer,
                 r_scheduler,
                 metrics,
                 loaders,
                 neptune_cfg=None,
                 cfg=None,
                 **kwargs):
    # reset RNGS before model training
    reset_rngs(seed)

    # build stuff
    model = build_model(**model)
    criterion, metrics = build_metrics(**metrics)

    optimizer = build_optimizer(list(model.e_parameters()), **optimizer)
    scheduler = build_scheduler(optimizer, **scheduler)

    r_optimizer = build_r_optimizer(list(model.r_parameters()), **r_optimizer)
    r_scheduler = build_r_scheduler(r_optimizer, **r_scheduler)

    if neptune_cfg is not None:
        import neptune.new as neptune
        run = neptune.init(**neptune_cfg)
        
        for i, group in enumerate(optimizer.param_groups):
            run[f'metrics/train/lr{i}'].log(group['lr'])
        
        if r_optimizer is not None:
            for i, group in enumerate(r_optimizer.param_groups):
                run[f'metrics/train/r_lr{i}'].log(group['lr'])
        
        run['config'] = OmegaConf.to_container(cfg, resolve=True)
    else:
        run = None

    results = train_loop(run,
                         epochs,
                         model,
                         criterion,
                         metrics,
                         optimizer,
                         scheduler,
                         r_optimizer,
                         r_scheduler,
                         loaders)
    return results, run


def convert_to_strings(d):
    res = {}
    for k, v in d.items():
        if isinstance(v, dict):
            res[k] = convert_to_strings(v)
        else:
            res[k] = str(v)
    return res


def build_metrics(loss, additional):
    criterion = build_loss(loss)
    metrics = {name: build_metric(name) for name in additional}
    return criterion, metrics


def build_loss(loss):
    if loss == 'MSE':
        return nn.MSELoss()
    else:
        raise NotImplementedError(f'the loss function {loss} is not implemented!')


def build_metric(name):
    if name == 'MAPE':
        return MAPE
    elif name == 'MSE':
        return MSE
    else:
        raise NotImplementedError(f'the metric {name} is not implemented!')


def build_optimizer(params, name, **kwargs):
    if not params:
        return None
    if name == 'SGD':
        return optim.SGD(params, **kwargs)
    elif name == 'Adam':
        return optim.Adam(params, **kwargs)
    else:
        raise NotImplementedError(f'the optimizer {name} is not implemented!')


def build_scheduler(optimizer, name, **kwargs):
    if optimizer is None:
        return None
    if name == 'None':
        return None
    elif name == 'ReduceLROnPlateau':
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
    else:
        raise NotImplementedError(f'the scheduler {name} is not implemented!')


def build_r_optimizer(params, name, **kwargs):
    if not params:
        return None
    if name == 'RiemannianSGD':
        return roptim.RiemannianSGD(params, **kwargs)
    # elif name == 'Adam':
    #     return optim.Adam(params, **kwargs)
    else:
        raise NotImplementedError(f'the optimizer {name} is not implemented!')


def build_r_scheduler(optimizer, name, **kwargs):
    if optimizer is None:
        return None
    if name == 'None':
        return None
    elif name == 'ReduceLROnPlateau':
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
    else:
        raise NotImplementedError(f'the scheduler {name} is not implemented!')
