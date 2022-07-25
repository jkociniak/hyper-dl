from hydra import initialize, compose
from omegaconf import DictConfig
from data import build_datasets
import os


def generate_datasets(cfg: DictConfig) -> None:
    print("Working directory : {}".format(os.getcwd()))
    build_datasets(**cfg)


if __name__ == "__main__":
    overrides = [
        'dim=2',
        'inverse_transform=hyperbolic',
        'curv=0.25'
    ]

    with initialize(config_path="conf"):
        cfg = compose(config_name="dataset_generation.yaml", overrides=overrides)

    generate_datasets(cfg)
