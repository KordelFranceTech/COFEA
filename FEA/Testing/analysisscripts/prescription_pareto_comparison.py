import pickle
import numpy as np
import itertools, random
#from pygmo.core import hypervolume
from pymoo.util.nds.non_dominated_sorting import find_non_dominated
from MOO.paretofrontevaluation import ParetoOptimization

# TODO: generalize

experiment_filenames = ["/media/amy/WD Drive/Prescriptions/optimal/rf/NSGA_Sec35Mid_strip_trial_3_objectives_ga_runs_500_population_500_1904081644.pickle",
"/media/amy/WD Drive/Prescriptions/optimal/rf/NSGA_Sec35Mid_strip_trial_3_objectives_ga_runs_500_population_500_2104073638.pickle",
"/media/amy/WD Drive/Prescriptions/optimal/rf/NSGA_Sec35Mid_strip_trial_3_objectives_ga_runs_500_population_500_2304064012.pickle",
"/media/amy/WD Drive/Prescriptions/optimal/rf/FEAMOO_Sec35Mid_strip_trial_3_objectives_ga_runs_20_population_25_1.pickle",
"/media/amy/WD Drive/Prescriptions/optimal/rf/FEAMOO_Sec35Mid_strip_trial_3_objectives_ga_runs_20_population_25_2.pickle",
"/media/amy/WD Drive/Prescriptions/optimal/rf/FEAMOO_Sec35Mid_strip_trial_3_objectives_ga_runs_20_population_25_3.pickle",
 "/media/amy/WD Drive/Prescriptions/optimal/rf/CCEAMOO_Sec35Mid_strip_trial_3_objectives_ga_runs_20_population_50_1.pickle",
"/media/amy/WD Drive/Prescriptions/optimal/rf/CCEAMOO_Sec35Mid_strip_trial_3_objectives_ga_runs_20_population_50_2.pickle",
 "/media/amy/WD Drive/Prescriptions/optimal/rf/CCEAMOO_Sec35Mid_strip_trial_3_objectives_ga_runs_20_population_50_3.pickle"
]

methods = ["CCEAMOO", "NSGA", "FEAMOO"]
fea_sum = 0
nsga_sum = 0
for i in range(5):
    nondom_solutions = dict()
    full_solutions = []
    for method in methods:
        experiment = [x for x in experiment_filenames if method in x.upper()]
        if experiment:
            experiment = experiment[random.randint(0,2)]
        else:
            break
        feamoo = pickle.load(open(experiment, 'rb'))
        nondom_solutions[method] = np.array([np.array(x.fitness) for x in feamoo.nondom_archive])
        full_solutions.extend([x for x in feamoo.nondom_archive])
    ccea_len = len(nondom_solutions['CCEAMOO'])
    nsga_len = len(nondom_solutions['NSGA'])
    fea_len = len(nondom_solutions['FEAMOO'])

    total_front = np.vstack((nondom_solutions['CCEAMOO'], nondom_solutions['FEAMOO'], nondom_solutions['NSGA']))
    indeces = find_non_dominated(total_front)
    global_solutions = [total_front[i] for i in indeces]
    all_solutions = [full_solutions[i] for i in indeces]
    po = ParetoOptimization()
    print('hv and diversity: ', po.evaluate_solution(all_solutions, [1,1,1]))
    print('CCEA value', len([x for x in indeces if x < ccea_len]) / len(indeces))
    fea_coverage = len([x for x in indeces if  ccea_len <= x < ccea_len+fea_len])/len(indeces)
    fea_sum += fea_coverage
    print('FEA value', fea_coverage)
    nsga_coverage = len([x for x in indeces if ccea_len+fea_len <= x <= ccea_len + fea_len + nsga_len]) / len(indeces)
    nsga_sum += nsga_coverage
    print('NSGA value', nsga_coverage)

print("averages:", fea_sum/5, nsga_sum/5 )

# print('CCEAMOO vs FEAMOO for field ', field_name)
# to_compare = np.vstack((nondom_solutions[field_name]['CCEAMOO'], nondom_solutions[field_name]['FEAMOO']))
# indeces = find_non_dominated(to_compare)
# print('CCEA value', len([x for x in indeces if x < ccea_len])/ccea_len)
# print('FEA value', len([x for x in indeces if  ccea_len <= x <= ccea_len+fea_len])/fea_len)
#
# print('CCEAMOO vs NSGA2 for field ', field_name)
# to_compare = np.vstack((nondom_solutions[field_name]['CCEAMOO'], nondom_solutions[field_name]['NSGA2']))
# indeces = find_non_dominated(to_compare)
# print('CCEA value', len([x for x in indeces if x < ccea_len])/ccea_len)
# print('NSGA value', len([x for x in indeces if  ccea_len <= x <= ccea_len+nsga_len])/nsga_len)
#
# print('NSGA2 vs FEAMOO for field ', field_name)
# to_compare = np.vstack((nondom_solutions[field_name]['NSGA2'], nondom_solutions[field_name]['FEAMOO']))
# indeces = find_non_dominated(to_compare)
# print('NSGA value', len([x for x in indeces if x < nsga_len])/nsga_len)
# print('FEA value', len([x for x in indeces if  nsga_len <= x <= nsga_len+fea_len])/fea_len)