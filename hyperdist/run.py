import hydra
from omegaconf import DictConfig
from config import run_training
from data import build_loaders
from plot import setup_dirs, dump_dataset_info, dump_results
import os


@hydra.main(config_path='conf', config_name='config')
def run_training(cfg: DictConfig) -> None:
    print("Working directory : {}".format(os.getcwd()))
    datasets, loaders = build_loaders(**cfg['dataset_params'])
    results = run_training(loaders=loaders,
                           cfg=cfg,
                           **cfg)

    folder_names = set(datasets.keys()) | set(results.keys())
    base_dir = os.getcwd()
    setup_dirs(folder_names, base_dir)
    dump_dataset_info(datasets, base_dir, dump_datasets=cfg.dump_datasets)
    dump_results(results, base_dir)


if __name__ == "__main__":
    run_training()
