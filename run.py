import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from config import run_training
from data import build_loaders, build_dataset_path
from plot import setup_dirs, dump_results
import os
import pickle
import threading


@hydra.main(config_path='conf', config_name='config')
def train(cfg: DictConfig) -> None:
    print("Working directory : {}".format(os.getcwd()))

    dataset_path = build_dataset_path(**cfg['dataset_params'])
    dataset_path = os.path.join(get_original_cwd(), 'datasets', dataset_path)

    with open(dataset_path, 'rb') as f:
        datasets = pickle.load(f)
    loaders = build_loaders(datasets, cfg['bs'], cfg['num_workers'])

    results, run = run_training(loaders=loaders,
                                cfg=cfg,
                                **cfg)
    folder_names = set(datasets.keys()) | set(results.keys())
    base_dir = os.getcwd()
    setup_dirs(folder_names, base_dir)

    dump_thread = threading.Thread(target=dump_results, args=(datasets, results, base_dir, run, cfg['plots']))
    dump_thread.start()


if __name__ == "__main__":
    train()
