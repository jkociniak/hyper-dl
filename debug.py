### VERSION OF RUN.PY WITHOUT HYDRA DECORATOR
### COMPOSE API IS USED INSTEAD TO ALLOW INTERACTIVE DEBUGGING

import hydra
from hydra import initialize, compose
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from config import run_training
from data import build_loaders, build_dataset_path, build_datasets
from plot import setup_dirs, dump_results
import os
import pickle
import threading


def train(cfg: DictConfig) -> None:
    cwd = os.getcwd()
    print("Working directory : {}".format(cwd))

    dataset_name = build_dataset_path(**cfg['dataset_params'])
    dataset_dir = os.path.join(cwd, 'datasets')
    dataset_path = os.path.join(dataset_dir, dataset_name)

    print(f'Loading dataset {dataset_name} from dir {dataset_dir}')

    try:
        with open(dataset_path, 'rb') as f:
            datasets = pickle.load(f)
        print('Successfully loaded dataset')
    except FileNotFoundError:
        print('Did not found requested dataset... generating new one')
        build_datasets(**cfg['dataset_params'], datasets_folder=dataset_dir)
        with open(dataset_path, 'rb') as f:
            datasets = pickle.load(f)
        print('Successfully loaded dataset')

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
        'dataset_params.dim=2',
        'dataset_params.n_samples=100000',
        #'~neptune_cfg',
        'epochs=200',
        #'model.encoder.hidden_dims=[2, 2]',
        #'model.encoder.bias=False',
        'model/encoder=true_encoder',
        'model/head=true_head',
        #'model.head.hidden_dims=[128, 128]',
        'dataset_params.curv=1'
    ]

    with initialize(config_path="conf"):
        cfg = compose(config_name="config.yaml", overrides=overrides)

    train(cfg)
