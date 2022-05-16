import os
from neptune.new.types import File
# NO IMPORT OF MATPLOTLIB HERE TO AVOID BUGS


def setup_dirs(names, dir_path):
    for name in names:
        path = os.path.join(dir_path, name)
        os.mkdir(path)


def dump_results(datasets, results, dir_path, run, plot):
    import matplotlib.pyplot as plt
    import seaborn as sns

    def plot_pairplot(res, path):
        sns_plot = sns.pairplot(data=res, height=3, aspect=1)
        sns_plot.fig.savefig(path)
        return sns_plot.fig

    set_names = ['test']
    for set_name in set_names:
        res = results[set_name]
        results_name = 'results.csv'
        path = os.path.join(dir_path, set_name, results_name)
        res.to_csv(path, index=False)
        if run is not None:
            run[f'results/{set_name}/results'].upload(path)

        #data = datasets[set_name]

        # plot_name = 'distributions.png'
        # plot_path = os.path.join(dir_path, set_name, plot_name)
        # plot_distributions(res, plot_path)
        #
        # plot_name = 'scatterplots.png'
        # plot_path = os.path.join(dir_path, set_name, plot_name)
        # plot_scatterplots(res, plot_path)

        if plot:
            plot_name = 'pairplot.png'
            plot_path = os.path.join(dir_path, set_name, plot_name)
            fig = plot_pairplot(res, plot_path)
            if run is not None:
                run[f'results/{set_name}/pairplot'] = File.as_html(fig)
    if run is not None:
        run.stop()


# def dataset_pairs(pairs, path):
#     fig, ax = plt.subplots()
#     ax.plot(pairs[:, 0, 0], pairs[:, 0, 1], 'o', color='black', markersize=2)  # first point in pair
#     ax.plot(pairs[:, 1, 0], pairs[:, 1, 1], 'o', color='blue', markersize=2)  # second point in pair
#     circle = plt.Circle((0, 0), 1, color='r', fill=False)
#
#     ax.set_xlim(-1.2, 1.2)
#     ax.set_ylim(-1.2, 1.2)
#     ax.set_aspect('equal')
#     ax.add_patch(circle)
#
#     fig.savefig(path)
#
#
# def plot_distributions(res, path):
#     n_plots = len(res.columns)
#     fig, ax = plt.subplots(1, n_plots, figsize=(40, 10))
#     for i in range(n_plots):
#         sns.histplot(res, x=res.columns[i], ax=ax[i])
#     fig.savefig(path)
#
#
# def plot_scatterplots(res, path):
#     fig, ax = plt.subplots(figsize=(10, 10))
#     sns.scatterplot(data=res, x='dist', y='pred', ax=ax)
#     fig.savefig(path)


