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


parser = argparse.ArgumentParser(description='Train hyperbolic distance prediction model.')
parser.add_argument('config_file', help='path to the configuration file', type=pathlib.Path)
args = parser.parse_args()
with open(args.config_file, 'r') as f:
    default_config = AttrDict(yaml.load(f, Loader=yaml.Loader))

# experiments grid definition
depths = range(3, 8)
widths = [160]
dims = range(2, 14)
results_path = f'results/grid_dims2-13_noskips_d3-7_width160'

# main loop
results = {"keys": list(product(dims, depths, widths))}
try:
    for dim in dims:  # separate loop because of different datasets
        default_config['dataset_params']['dim'] = dim

        # reseed the RNGs before constructing new datasets
        torch.manual_seed(default_config.seed)
        random.seed(default_config.seed)
        np.random.seed(default_config.seed)

        loaders = build_loaders(**default_config.dataset_params)
        for depth, width in product(depths, widths):
            temp_config = deepcopy(default_config)

            temp_config['model']['hidden_depth'] = depth
            temp_config['model']['hidden_width'] = width

            hidden_dims = [width] * depth
            temp_config['model']['hidden_dims'] = hidden_dims
            temp_config['model']['input_dim'] = dim

            results[dim, depth, width] = {'config': temp_config}

            print(f'Running experiment with dim={dim}, hidden_depth={depth}, hidden_width={width}')
            results[dim, depth, width]['metrics'] = run_training(loaders['train'],
                                                                 loaders['val'],
                                                                 loaders['test'],
                                                                 temp_config)
finally:  # save partial results even if program crashes or is stopped
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
