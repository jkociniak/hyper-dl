import hydra
from omegaconf import DictConfig
from data import build_datasets
import os


@hydra.main(config_path='conf', config_name='dataset_generation')
def generate_datasets(cfg: DictConfig) -> None:
    print("Working directory : {}".format(os.getcwd()))
    build_datasets(**cfg)


if __name__ == "__main__":
    generate_datasets()
