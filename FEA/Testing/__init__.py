# import pickle
# from pstats import Stats

# from sklearn.metrics import adjusted_rand_score
# from utilities.util import PopulationMember
# from MOO.paretofront import ParetoOptimization
# import numpy as np
# from pymoo.util.nds.non_dominated_sorting import find_non_dominated
# import matplotlib.pyplot as plt
# import matplotlib.lines as mlines

# experiments = [
# "/media/amy/WD Drive/Prescriptions/optimal/NSGA_Henrys_strip_trial_3_objectives_ga_runs_500_population_500_2204063149.pickle"
# ]

# np.set_printoptions(suppress=True)
# for j,experiment in enumerate(experiments):
#     file = open(experiment, "rb")
#     #moo_model = pickle.Unpickler(file).load()
#     moo_model = pickle.load(file)
#     print(moo_model.iteration_stats[-1])
#     print(moo_model.nondom_archive[-1].fitness)
# #     po = ParetoOptimization()
# #     stats = po.evaluate_solution(moo_model.nondom_archive, [1,1,1])
# #     print(stats)
# #     moo_model.iteration_stats[-1]["hypervolume"] = stats["hypervolume"]
# #     moo_model.iteration_stats[-1]["diversity"] = stats["diversity"]
# #     #pickle.dump(moo_model, open("/media/amy/WD Drive/Prescriptions/optimal/FEAMOO_Sec35Mid_strip_trial_3_objectives_ga_runs_20_population_25_3.pickle", "wb"))
