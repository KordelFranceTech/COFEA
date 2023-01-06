"""
Running Knapsack experiments.
"""

import os, re
import pickle
from datetime import timedelta
import time

from optimizationproblems.knapsack import *
from MOO.MOFEA import MOFEA
from MOO.MOEA import NSGA2, SPEA2
from FEA.factorarchitecture import FactorArchitecture
from utilities.util import *

nr_items = 1000
sizes = [100, 100]  # , 200, 100]
overlaps = [80, 100]  # 100, 160, 80]  # , 10, 20]
fea_runs = [20]
ga_run = 100
population = 500
ks_type = 'multi'
nr_obj = 3

FA = FactorArchitecture(nr_items)

ga = SPEA2
ks = Knapsack(number_of_items=nr_items, max_nr_items=nr_items, nr_objectives=nr_obj, nr_constraints=1,
              knapsack_type=ks_type)

current_working_dir = os.getcwd()
path = re.search(r'^(.*?[\\/]FEA)', current_working_dir)
path = path.group()


@add_method(SPEA2)
def calc_fitness(variables, gs=None, factor=None):
    if gs is not None and factor is not None:
        full_solution = [x for x in gs.variables]
        for i, x in zip(factor, variables):
            full_solution[i] = x
    else:
        full_solution = variables
    if ks_type == 'single':
        ks.set_fitness_single_knapsack(full_solution)
    elif ks_type == 'multi':
        ks.set_fitness_multi_knapsack(full_solution)
    return ks.objective_values


for s, o in zip(sizes, overlaps):
    if s == o:
        name = 'CCSPEA2'
    else:
        name = 'FSPEA2'
    FA.linear_grouping(s, o)
    FA.get_factor_topology_elements()
    for i in range(5):
        print('##############################################\n', i)
        for fea_run in fea_runs:
            start = time.time()
            filename = path + '/results/Knapsack/' + name + '/' + name + '_' + ks_type + '_knapsack_' + str(nr_obj) + \
                       '_objectives_fea_runs_' + str(fea_run) + '_grouping_' + str(s) + '_' + \
                       str(o) + time.strftime('_%d%m%H%M%S') + '.pickle'
            feamoo = MOFEA(fea_run, alg_iterations=ga_run, pop_size=population, factor_architecture=FA, base_alg=ga, dimensions=nr_items,
                           combinatorial_options=[0, 1], ref_point=ks.ref_point)
            feamoo.run()
            end = time.time()
            file = open(filename, "wb")
            pickle.dump(feamoo, file)
            elapsed = end - start
            print(
                "FEA with ga runs %d and population %d %d took %s" % (fea_run, s, o, str(timedelta(seconds=elapsed))))
