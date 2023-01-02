import pickle, glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
from utilities.multifilereader import *

# TODO: Generalize


field_names = ['millview']
methods = ["NSGA2", "SPEA2"]
iterations = []
alg_stats = dict()
for field_name in field_names:
    mf = MultiFileReader(field_name)
    experiment_filenames = mf.path_to_files
    alg_stats[field_name] = dict()
    for method in methods:
        print(method)
        alg_stats[field_name][method] = dict()
        experiments = [x for x in experiment_filenames if method + '_' in x and field_name in x.lower()]
        if experiments:
            for experiment in experiments:
                feamoo = pickle.load(open(experiment, 'rb'))
                print(feamoo.iteration_stats[-1])
                print(feamoo.nondom_archive[0].fitness)
                try:
                    alg_stats[field_name][method]['iterations'] = [stats['FEA_run'] for i, stats in
                                                                enumerate(feamoo.iteration_stats) if i != 0]
                except KeyError:
                    alg_stats[field_name][method]['iterations'] = [stats['GA_run'] for i, stats in
                                                                enumerate(feamoo.iteration_stats) if i != 0]
                alg_stats[field_name][method]['diversity'] = [stats['diversity'] for i, stats in
                                                            enumerate(feamoo.iteration_stats) if i != 0]
                alg_stats[field_name][method]['hv'] = [stats['hypervolume'] for i, stats in enumerate(feamoo.iteration_stats) if
                                                    i != 0]
                alg_stats[field_name][method]['nd'] = [stats['ND_size'] for i, stats in enumerate(feamoo.iteration_stats) if
                                                    i != 0]
        else:
            break

stat_names = ['hv', 'diversity', 'nd']

stat_name = 'hv'
# ccea = list(alg_stats['henrys']['CCEAMOO'][stat_name])
# nsga = list(alg_stats['henrys']['NSGA2'][stat_name])
# fea = list(alg_stats['henrys']['FEAMOO'][stat_name])
# fig, axs = plt.subplots(1, 1, figsize=(15, 4.5), sharey=True)

# axs[0].plot(alg_stats['henrys']['CCEAMOO']['iterations'], ccea, color='tab:blue', marker='.')
# axs[0].plot(alg_stats['henrys']['NSGA2']['iterations'], nsga, color='#2ca02c', marker="^")
# axs[0].plot(alg_stats['henrys']['FEAMOO']['iterations'], fea, color='tab:red', marker="*")
# axs[0].set_title("Henrys")

# ccea = list(alg_stats['sec35middle']['CCEAMOO'][stat_name])
# nsga = list(alg_stats['sec35middle']['NSGA2'][stat_name])
# fea = list(alg_stats['sec35middle']['FEAMOO'][stat_name])

# axs[0].plot(alg_stats['sec35middle']['CCEAMOO']['iterations'], ccea, color='tab:blue', marker='.')
# axs[0].plot(alg_stats['sec35middle']['NSGA2']['iterations'], nsga, color='#2ca02c', marker="^")
# axs[0].plot(alg_stats['sec35middle']['FEAMOO']['iterations'], fea, color='tab:red', marker="*")
# axs[0].set_title("Sec35Mid")
# axs[0].set_xlabel("Generations")

# ccea = list(alg_stats['sec35west']['CCEAMOO'][stat_name])
# nsga = list(alg_stats['sec35west']['NSGA2'][stat_name])
# fea = list(alg_stats['sec35west']['FEAMOO'][stat_name])

# axs[2].plot(alg_stats['sec35west']['CCEAMOO']['iterations'], ccea, color='tab:blue',marker='.')
# axs[2].plot(alg_stats['sec35west']['NSGA2']['iterations'], nsga, color='#2ca02c', marker="^")
# axs[2].plot(alg_stats['sec35west']['FEAMOO']['iterations'], fea, color='tab:red', marker="*")
# axs[2].set_title("Sec35West")
# fig.suptitle('')

# ccea = mlines.Line2D([], [], color='#1f77b4', marker='.',
#                      markersize=12, label='CC-NSGA-II')
# nsga = mlines.Line2D([], [], color='#2ca02c', marker='^',
#                      markersize=12, label='NSGA-II')
# fea = mlines.Line2D([], [], color='tab:red', marker='*',
#                     markersize=12, label='F-NSGA-II')
# axs[1].legend(handles=[nsga, ccea, fea], bbox_to_anchor=(0, 1.15, 1., .105), loc='center',
#               ncol=3, mode="expand", borderaxespad=0.)
# # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
# plt.suptitle('Hypervolume', size='18')
# fig.tight_layout(pad=1)
# plt.show()

# stat_name = 'diversity'
# ccea = list(alg_stats['henrys']['CCEAMOO'][stat_name])
# nsga = list(alg_stats['henrys']['NSGA2'][stat_name])
# fea = list(alg_stats['henrys']['FEAMOO'][stat_name])
# data = [nsga, ccea, fea]
# fig, axs = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
# axs[0].set_ylabel('')
# axs[0].boxplot(data)
# axs[0].set_title("Henrys")
# axs[0].set_xticklabels(['NSGA-II', 'CC-NSGA-II', 'F-NSGA-II'])
#
# ccea = list(alg_stats['sec35middle']['CCEAMOO'][stat_name])
# nsga = list(alg_stats['sec35middle']['NSGA2'][stat_name])
# fea = list(alg_stats['sec35middle']['FEAMOO'][stat_name])
# data = [nsga, ccea, fea]
# axs[1].boxplot(data)
# axs[1].set_title("Sec35Mid")
# axs[1].set_xticklabels(['NSGA-II', 'CC-NSGA-II', 'F-NSGA-II'])
#
# ccea = list(alg_stats['sec35west']['CCEAMOO'][stat_name])
# nsga = list(alg_stats['sec35west']['NSGA2'][stat_name])
# fea = list(alg_stats['sec35west']['FEAMOO'][stat_name])
# data = [nsga, ccea, fea]
# axs[2].boxplot(data)
# axs[2].set_title("Sec35West")
# axs[2].set_xticklabels(['NSGA-II', 'CC-NSGA-II', 'F-NSGA-II'])
#
# fig.suptitle('')
# plt.suptitle('Spread Indicator', size='18')
# fig.tight_layout(pad=1)
# plt.show()
