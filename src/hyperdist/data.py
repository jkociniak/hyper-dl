import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.uniform import Uniform
from .utils import hyperbolic_dist_np


def build_loaders(n_samples, dim, eps, bs):
    n_train, n_val = int(0.7 * n_samples), int(0.2 * n_samples)

    sizes = {
        'train': n_train,
        'val': n_val,
        'test': int(n_samples - n_train - n_val)
    }

    datasets = {name: HyperbolicPairsDataset(size, dim, eps)
                for name, size in sizes.items()}

    loaders = {name: build_dataloader(name, dataset, bs)
               for name, dataset in datasets.items()}

    return loaders


def build_dataloader(name, dataset, bs):
    if name == 'test':
        return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    else:
        return DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=4)


class HyperbolicPairsDataset(Dataset):
    def __init__(self, n_samples, dim, eps):
        self.n_samples = n_samples
        self.dim = dim
        self.eps = eps
        self.pairs = self.generate_hyperbolic_pairs()
        self.distances = self.compute_distances()

    def generate_hyperbolic_pairs(self):
        mean = torch.zeros(self.dim)
        cov = torch.eye(self.dim)
        distribution = MultivariateNormal(mean, cov)
        directions = distribution.sample((self.n_samples, 2))

        norms = torch.linalg.norm(directions, axis=2, keepdims=True)
        unit_directions = torch.divide(directions, norms)

        max_radius = 1 - self.eps
        distribution = Uniform(0, max_radius)
        radii = distribution.sample((self.n_samples, 2, 1))
        transformed_radii = torch.pow(radii, 1/self.dim)

        pairs = unit_directions * transformed_radii
        return pairs

    def compute_distances(self):
        # based on https://stackoverflow.com/questions/46084656/numpy-apply-along-n-spaces
        pairs_reshaped = np.reshape(self.pairs.numpy(), (self.n_samples, 2 * self.dim))  # flatten the two last axes

        def hyperbolic_dist_wrapper(row):
            row_2d = np.reshape(row, (2, self.dim))  # unflatten
            return hyperbolic_dist_np(row_2d[0], row_2d[1])  # compute the distance

        distances = np.apply_along_axis(hyperbolic_dist_wrapper, 1, pairs_reshaped)
        distances = distances.astype(np.float32)
        return torch.from_numpy(distances)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return {'pairs': self.pairs[idx], 'dist': self.distances[idx]}