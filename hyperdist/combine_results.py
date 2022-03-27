import os
import pandas as pd
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Combine results of a multirun into one dataframe.')
    parser.add_argument('multirun_dir')
    args = parser.parse_args()
    return args.multirun_dir


# https://stackoverflow.com/questions/4664850/how-to-find-all-occurrences-of-a-substring
def find_all(a_str, sub):
    start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1: return
        yield start
        start += len(sub) # use start += 1 to find overlapping matches


def handle_list(s):
    # we assume that the string is of form xxxx[xxx]xxxx[xxxx]x[xxxx]xxxxx
    left_idx = find_all(s, '[')
    right_idx = find_all(s, ']')
    par_idx = zip(left_idx, right_idx)
    s = list(s)
    for lp_idx, rp_idx in par_idx:
        substr = ''.join(s[lp_idx:rp_idx+1])
        s[lp_idx:rp_idx+1] = list(substr.replace(',', '|'))
    return ''.join(s)


def get_experiment_folders(multirun_dir):
    for base_entry in os.scandir(multirun_dir):
        if base_entry.is_dir():  # subfolder with name param1=val1,param2=val2 etc
            params = base_entry.name
            params = handle_list(params)
            params = params.split(sep=',')
            params = [p.split(sep='=') for p in params]
            params = {p[0]: p[1] for p in params}
            for seed_entry in os.scandir(base_entry.path):
                if seed_entry.is_dir():  # subfolder with name seed=val
                    params['seed'] = seed_entry.name.split('=')[1]
                    for set_entry in os.scandir(seed_entry.path):
                        if set_entry.is_dir() and set_entry.name[0] != '.':  # subfolder with name train/val/test
                            params['set'] = set_entry.name
                            yield set_entry.path, dict(params)


def get_full_results(multirun_dir):
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
    return results_full


if __name__ == "__main__":
    multirun_dir = parse_args()
    results_full = get_full_results(multirun_dir)
    results_full = pd.DataFrame(results_full)
    results_path = os.path.join(multirun_dir, 'full_results.csv')
    results_full.to_csv(results_path)
