import argparse
import yaml
import pathlib
from hyperdist.config import run_training
from attrdict import AttrDict
from itertools import product
from copy import deepcopy
import pickle

parser = argparse.ArgumentParser(description='Train hyperbolic distance prediction model.')
parser.add_argument('config_file', help='path to the configuration file', type=pathlib.Path)
args = parser.parse_args()
with open(args.config_file, 'r') as f:
    config = AttrDict(yaml.load(f, Loader=yaml.Loader))

depths = range(1, 10)
widths = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
ns = [1e5]

results = {}
for depth, width, n_samples in product(depths, widths, ns):
    temp_config = deepcopy(config)
    hidden_dims = [width] * depth
    temp_config['model']['hidden_dims'] = hidden_dims
    temp_config['dataset_params']['n_samples'] = n_samples
    print(f'Running experiment with hidden_dims={hidden_dims}, n_samples={n_samples}')
    results[depth, width, n_samples] = run_training(temp_config)

results_path = '../reports/dim2_width_depth_nsamples_grid_search_results'
with open(results_path, 'wb') as f:
    pickle.dump(results, f)

print('Finished training!')