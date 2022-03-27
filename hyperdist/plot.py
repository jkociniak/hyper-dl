import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle


def setup_dirs(names, dir_path):
    for name in names:
        path = os.path.join(dir_path, name)
        os.mkdir(path)


def dataset_pairs(pairs, path):
    fig, ax = plt.subplots()
    ax.plot(pairs[:, 0, 0], pairs[:, 0, 1], 'o', color='black', markersize=2)  # first point in pair
    ax.plot(pairs[:, 1, 0], pairs[:, 1, 1], 'o', color='blue', markersize=2)  # second point in pair
    circle = plt.Circle((0, 0), 1, color='r', fill=False)

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.add_patch(circle)

    fig.savefig(path)


def plot_distributions(res, path):
    n_plots = len(res.columns)
    fig, ax = plt.subplots(1, n_plots, figsize=(40, 10))
    for i in range(n_plots):
        sns.histplot(res, x=res.columns[i], ax=ax[i])
    fig.savefig(path)


def plot_scatterplots(res, path):
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.scatterplot(data=res, x='dist', y='pred', ax=ax)
    fig.savefig(path)


def plot_pairplot(res, path):
    sns_plot = sns.pairplot(data=res, height=3, aspect=1)
    sns_plot.fig.savefig(path)


def dump_dataset_info(datasets, dir_path, dump_datasets=False):
    for name, dataset in datasets.items():
        plot_name = 'pairs_2d_visual.png'
        plot_path = os.path.join(dir_path, name, plot_name)

        if dataset.dim == 2:
            dataset_pairs(dataset.pairs, plot_path)

        if dump_datasets:
            dataset_name = 'dataset.pickle'
            dataset_path = os.path.join(dir_path, name, dataset_name)
            with open(dataset_path, 'wb') as f:
                pickle.dump(dataset.pairs, f)


def dump_results(results, dir_path):
    for set_name, res in results.items():
        results_name = 'results.csv'
        path = os.path.join(dir_path, set_name, results_name)
        res.to_csv(path, index=False)

        # plot_name = 'distributions.png'
        # plot_path = os.path.join(dir_path, set_name, plot_name)
        # plot_distributions(res, plot_path)
        #
        # plot_name = 'scatterplots.png'
        # plot_path = os.path.join(dir_path, set_name, plot_name)
        # plot_scatterplots(res, plot_path)

        plot_name = 'pairplot.png'
        plot_path = os.path.join(dir_path, set_name, plot_name)
        plot_pairplot(res, plot_path)




