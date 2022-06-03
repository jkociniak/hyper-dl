import seaborn as sns
import pickle
from plot import plot_pairplot
from data import build_dataset_path
from itertools import product


settings = {
    'seed': 777,
    'inverse_transform': 'hyperbolic',
    'min_r': 0.1,
    'max_r': 5.3,
    'n_samples': 100000
}

dims = range(16, 20),
curvs = [0.9, 0.99, 1, 1.01, 1.1],

for dim, curv in product(dims, curvs):
    print(f'processing dim: {dim}, curv: {curv}')
    path = build_dataset_path(dim=dim, curv=curv, **settings)
    with open(path, 'rb') as f:
        datasets = pickle.load(f)
    for name, set in datasets:
        print(f'Processing {name} set')
