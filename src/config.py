import torch.nn as nn
import torch.optim as optim
from models import build_model
from loops import train_loop
from utils import mape


def run_training(train_loader, val_loader, test_loader, config):
    model = build_model(**config.model)
    criterion, metrics = build_metrics(**config.metrics)
    optimizer = build_optimizer(model.parameters(), **config.optimizer)
    scheduler = build_scheduler(optimizer, **config.scheduler)

    if 'neptune' in config:
        import neptune.new as neptune
        run = neptune.init(**config.neptune)
        str_config = convert_to_strings(config)
        run['config'] = str_config
    else:
        run = None

    results = train_loop(run,
                         config.epochs,
                         model,
                         criterion,
                         metrics,
                         optimizer,
                         scheduler,
                         train_loader,
                         val_loader,
                         test_loader)
    if run is not None:
        run.stop()
    return results


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
        return mape
    else:
        raise NotImplementedError(f'the metric {name} is not implemented!')


def build_optimizer(params, name, **kwargs):
    if name == 'SGD':
        return optim.SGD(params, **kwargs)
    elif name == 'Adam':
        return optim.Adam(params, **kwargs)
    else:
        raise NotImplementedError(f'the optimizer {name} is not implemented!')


def build_scheduler(optimizer, name, **kwargs):
    if name == 'None':
        return None
    elif name == 'ReduceLROnPlateau':
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
    else:
        raise NotImplementedError(f'the scheduler {name} is not implemented!')






