import os
import pandas as pd

multirun_dir = 'experiments/multi/transform_dim_big_batch/2022-03-24_03:44:38'


def get_experiment_folders(multirun_dir):
    for base_entry in os.scandir(multirun_dir):
        if base_entry.is_dir():  # subfolder with name param1=val1,param2=val2 etc
            params = base_entry.name.split(sep=',', maxsplit=1)
            params = [p.split(sep='=') for p in params]
            params = {p[0]: p[1] for p in params}
            for seed_entry in os.scandir(base_entry.path):
                if seed_entry.is_dir():  # subfolder with name seed=val
                    params['seed'] = seed_entry.name.split('=')[1]
                    for set_entry in os.scandir(seed_entry.path):
                        if set_entry.is_dir() and set_entry.name[0] != '.':  # subfolder with name train/val/test
                            params['set'] = set_entry.name
                            yield set_entry.path, dict(params)


results_full = []
for i, (folder, params) in enumerate(get_experiment_folders(multirun_dir)):
    print(i)
    results_path = os.path.join(folder, 'results.csv')
    results = pd.read_csv(results_path)
    result_row = params
    for col in results.columns:
        if col not in ['Unnamed: 0', 'dist', 'pred']:
            result_row[col] = results[col].mean()
    results_full.append(result_row)

results_full = pd.DataFrame(results_full)
results_path = os.path.join(multirun_dir, 'full_results.csv')
results_full.to_csv(results_path)

