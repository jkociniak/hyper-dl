### VERSION OF RUN.PY WITHOUT HYDRA DECORATOR
### COMPOSE API IS USED INSTEAD TO ALLOW INTERACTIVE DEBUGGING

import hydra
from hydra import initialize, compose
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from config import run_training
from data import build_loaders, build_dataset_path
from plot import setup_dirs, dump_results
import os
import pickle
import threading


def train(cfg: DictConfig) -> None:
    cwd = os.getcwd()
    print("Working directory : {}".format(cwd))

    dataset_path = build_dataset_path(**cfg['dataset_params'])
    dataset_path = os.path.join(cwd, 'datasets', dataset_path)

    with open(dataset_path, 'rb') as f:
        datasets = pickle.load(f)
    loaders = build_loaders(datasets, cfg['bs'], cfg['num_workers'])

    results, run = run_training(loaders=loaders,
                                cfg=cfg,
                                **cfg)
    # folder_names = set(datasets.keys()) | set(results.keys())
    # base_dir = os.getcwd()
    # setup_dirs(folder_names, base_dir)


if __name__ == "__main__":
    overrides = [
        'num_workers=0',
        #'~neptune_cfg',
        'epochs=50',
        'model/head=hyperbolic_dist',
        'model.encoder.hidden_dims=[32,32]',
        'model.encoder.bias=True'
    ]

    with initialize(config_path="conf"):
        cfg = compose(config_name="config.yaml", overrides=overrides)
    
    train(cfg)
