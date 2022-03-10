import argparse
import yaml
import pathlib
import torch
import numpy as np
import random
from config import run_training
from data import build_loaders
from attrdict import AttrDict
from itertools import product
from copy import deepcopy
import pickle


def reset_rngs(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


parser = argparse.ArgumentParser(description='Train hyperbolic distance prediction model.')
parser.add_argument('config_file', help='path to the configuration file', type=pathlib.Path)
args = parser.parse_args()
with open(args.config_file, 'r') as f:
    default_config = AttrDict(yaml.load(f, Loader=yaml.Loader))

# experiments grid definition
epsilons = [1e-2, 1e-3, 1e-4]
dim_tdim_products = [(dim, t_dim) for dim in range(2, 5) for t_dim in range(dim, dim + 3)]
results_path = f'results/grid_transform_dims_full'
results_gdrive_path = '/content/drive/My Drive/Hyperbolic neural networks/' + results_path
model_seeds = [777, 888, 999]

# main loop
results = {}
for eps, (dim, t_dim) in product(epsilons, dim_tdim_products):
    default_config['dataset_params']['eps'] = eps
    default_config['dataset_params']['dim'] = dim
    default_config['model']['input_dim'] = dim
    default_config['dataset_params']['transform_dim'] = t_dim

    # reset RNGs before dataset generation
    reset_rngs(default_config.dataset_seed)

    loaders = build_loaders(**default_config.dataset_params)

    for seed in model_seeds:
        default_config['model_seed'] = seed

        temp_config = deepcopy(default_config)
        results[eps, dim, t_dim, seed] = {'config': temp_config}

        print(f'Running experiment with eps={eps}, dim={dim}, transform_dim={t_dim}, seed={seed}')
        # reset RNGS before model training
        reset_rngs(seed)
        results[eps, dim, t_dim, seed]['metrics'] = run_training(loaders['train'],
                                                                 loaders['val'],
                                                                 loaders['test'],
                                                                 temp_config)

        with open(results_gdrive_path, 'wb') as f:
            pickle.dump(results, f)
