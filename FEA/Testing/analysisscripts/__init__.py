# import pickle
# from utilities.util import PopulationMember
# from MOO.paretofront import ParetoOptimization
# import numpy as np
# from pymoo.util.nds.non_dominated_sorting import find_non_dominated
# import matplotlib.pyplot as plt
# import matplotlib.lines as mlines
# from itertools import combinations
# from scipy.stats import ttest_ind

# experiments = ["/media/amy/WD Drive/Prescriptions/optimal/rf/NSGA_Sec35Mid_strip_trial_3_objectives_ga_runs_500_population_500_1904081644.pickle",
# "/media/amy/WD Drive/Prescriptions/optimal/rf/NSGA_Sec35Mid_strip_trial_3_objectives_ga_runs_500_population_500_2104073638",
# "/media/amy/WD Drive/Prescriptions/optimal/rf/FEAMOO_Sec35Mid_strip_trial_3_objectives_ga_runs_20_population_25_2.pickle",
# "/media/amy/WD Drive/Prescriptions/optimal/rf/FEAMOO_Sec35Mid_strip_trial_3_objectives_ga_runs_20_population_25_1.pickle",
# "/media/amy/WD Drive/Prescriptions/optimal/rf/FEAMOO_Sec35Mid_strip_trial_3_objectives_ga_runs_20_population_25_3.pickle",
# "/media/amy/WD Drive/Prescriptions/optimal/rf/CCEAMOO_Sec35Mid_strip_trial_3_objectives_ga_runs_20_population_50_1.pickle",
# "/media/amy/WD Drive/Prescriptions/optimal/rf/CCEAMOO_Sec35Mid_strip_trial_3_objectives_ga_runs_20_population_50_2.pickle",
# "/media/amy/WD Drive/Prescriptions/optimal/rf/CCEAMOO_Sec35Mid_strip_trial_3_objectives_ga_runs_20_population_50_3.pickle"
# ]

# fig, ax = plt.subplots()
# colors = [ "#2ca02c", "tab:red", "#1f77b4", "black"]
# symbols = [ "^", "*", ".", "s"]

# objs = ['jumps', 'fert', 'NR', 'centr']
# ttest_objs = {'jumps':[], 'fert':[], 'NR':[], 'centr':[]}
# np.set_printoptions(suppress=True)
# total_nondom = []
# for j,experiment in enumerate(experiments):
#     NR = []
#     moo_model = pickle.load(open(experiment, "rb"))
#     find_center_obj = []
#     nondom_fitness = np.array([np.array(x.fitness) for x in moo_model.nondom_archive])
#     total_nondom.extend(nondom_fitness)
#     for i in range(3):
#         nondom_fitness = nondom_fitness[nondom_fitness[:,i].argsort()]
#         prescription = nondom_fitness[0]
#         ttest_objs[objs[i]].append(prescription[-1])
#         NR.append(-1*prescription[-1])
#         find_center_obj.append(np.array(prescription))
#     find_center_obj = np.array(find_center_obj)
#     length = find_center_obj.shape[0]
#     sum_x = np.sum(find_center_obj[:, 0])
#     sum_y = np.sum(find_center_obj[:, 1])
#     sum_z = np.sum(find_center_obj[:, 2])
#     point = np.array([sum_x / length, sum_y / length, sum_z / length])
#     dist = np.sum((nondom_fitness - point) ** 2, axis=1)
#     idx = np.argmin(dist)
#     prescription = nondom_fitness[idx]
#     ttest_objs[objs[-1]].append(prescription[-1])
#     NR.append(-1*prescription[-1])

#     # ax.scatter(objs, NR, color=colors[j], marker=symbols[j])
#     # ax.plot(objs, NR, color=colors[j])

# param_perm = combinations(objs, 2)
# for perm in param_perm:
#     print(perm)
#     print(ttest_objs[perm[0]])
#     print(ttest_ind(ttest_objs[perm[0]], ttest_objs[perm[1]]))
#     print(ttest_ind(ttest_objs[perm[0]], ttest_objs[perm[1]]))

# # NR = []
# # find_center_obj = []
# # idxs = find_non_dominated(np.array(total_nondom))
# # total_nondom = np.array([np.array(total_nondom[i])for i in idxs])
# # for i in range(3):
# #     print(objs[i])
# #     total_nondom = total_nondom[total_nondom[:,i].argsort()]
# #     prescription = total_nondom[0]
# #     NR.append(-1*prescription[-1])
# #     find_center_obj.append(np.array(prescription))

# # find_center_obj = np.array(find_center_obj)
# # length = find_center_obj.shape[0]
# # sum_x = np.sum(find_center_obj[:, 0])
# # sum_y = np.sum(find_center_obj[:, 1])
# # sum_z = np.sum(find_center_obj[:, 2])
# # point = np.array([sum_x / length, sum_y / length, sum_z / length])
# # dist = np.sum((total_nondom - point) ** 2, axis=1)
# # idx = np.argmin(dist)
# # prescription = total_nondom[idx]
# # NR.append(-1*prescription[-1])

# # ax.scatter(objs, NR, color=colors[-1], marker=symbols[-1])
# # ax.plot(objs, NR, color=colors[-1])

# # ccea = mlines.Line2D([], [], color='#1f77b4', marker='.',
# #                           markersize=12, label='CC-NSGA-II')
# # nsga = mlines.Line2D([], [], color='#2ca02c', marker='^',
# #                           markersize=12, label='NSGA-II')
# # fea = mlines.Line2D([], [], color='tab:red', marker='*',
# #                           markersize=12, label='F-NSGA-II')
# # comb = mlines.Line2D([], [], color='black', marker='s',
# #                           markersize=10, label='Union')
# # box = ax.get_position()
# # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# # plt.legend(handles=[nsga, ccea, fea, comb], bbox_to_anchor=(1,.5)) #, bbox_to_anchor=(0, 1.15, 1., .105), loc='center',ncol=4, mode="expand", borderaxespad=0.)
# # #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
# # plt.suptitle('Net Return', size='18')

# # plt.show()