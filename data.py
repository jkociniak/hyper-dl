import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.uniform import Uniform
from utils import hyperbolic_dist_np, reset_rngs
import pickle
import os


def build_datasets(seed, n_samples, dim, eps, transform_dim, curv, datasets_folder):
    reset_rngs(seed)  # reset RNGs before dataset generation

    n_train, n_val = int(0.7 * n_samples), int(0.2 * n_samples)

    sizes = {
        'train': n_train,
        'val': n_val,
        'test': int(n_samples - n_train - n_val)
    }

    datasets = {name: HyperbolicPairsDataset(size, dim, eps, curv, transform_dim)
                for name, size in sizes.items()}

    filename = f'dim={dim},n_samples={n_samples},eps={eps},transform_dim={transform_dim},curv={curv},seed={seed}.pkl'
    filepath = os.path.join(datasets_folder, filename)
    with open(filepath, 'wb') as f:
        pickle.dump(datasets, f)


def build_loaders(datasets, bs, num_workers):
    loaders = {name: build_dataloader(name, dataset, bs, num_workers)
               for name, dataset in datasets.items()}
    return loaders


def build_dataloader(name, dataset, bs, num_workers):
    if name == 'train':
        return DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=num_workers, pin_memory=True)
    else:
        return DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=num_workers, pin_memory=True)


class HyperbolicPairsDataset(Dataset):
    def __init__(self, n_samples, dim, eps, curv, transform_dim=None):
        self.n_samples = n_samples
        self.dim = dim
        self.eps = eps
        self.curv = curv
        self.transform_dim = dim
        if transform_dim is not None:
            self.transform_dim = transform_dim
        self.pairs = self.generate_hyperbolic_pairs()
        self.distances = self.compute_distances()

    def generate_hyperbolic_pairs(self):
        mean = torch.zeros(self.dim)
        cov = torch.eye(self.dim)
        distribution = MultivariateNormal(mean, cov)
        directions = distribution.sample((self.n_samples, 2))

        norms = torch.linalg.norm(directions, axis=2, keepdims=True)
        unit_directions = torch.divide(directions, norms)

        max_radius = 1/self.curv - self.eps
        distribution = Uniform(0, max_radius)
        radii = distribution.sample((self.n_samples, 2, 1))
        transformed_radii = torch.pow(radii, 1 / self.transform_dim)

        pairs = unit_directions * transformed_radii
        return pairs

    def compute_distances(self):
        # based on https://stackoverflow.com/questions/46084656/numpy-apply-along-n-spaces
        pairs_reshaped = np.reshape(self.pairs.numpy(), (self.n_samples, 2 * self.dim))  # flatten the two last axes

        def hyperbolic_dist_wrapper(row):
            row_2d = np.reshape(row, (2, self.dim))  # unflatten
            return hyperbolic_dist_np(row_2d[0], row_2d[1], self.curv)  # compute the distance

        distances = np.apply_along_axis(hyperbolic_dist_wrapper, 1, pairs_reshaped)
        distances = distances.astype(np.float32)
        return torch.from_numpy(distances)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return {'pairs': self.pairs[idx], 'dist': self.distances[idx]}
