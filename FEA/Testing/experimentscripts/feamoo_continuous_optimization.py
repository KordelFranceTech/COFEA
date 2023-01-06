from datetime import timedelta

from pymoo.decomposition.tchebicheff import Tchebicheff
from pymoo.factory import get_problem, get_reference_directions

from MOO.MOEA import NSGA2, SPEA2, MOEAD, MOEA
from FEA.factorarchitecture import FactorArchitecture
from MOO.MOFEA import MOFEA
from utilities.util import *

import os, re, time, pickle
from pymoo.problems.many.dtlz import DTLZ1

dimensions = 1000
sizes = [100]  # , 200, 100]
overlaps = [100]  # 100, 160, 80]  # , 10, 20]
fea_runs = [20]
ga_run = 20
population = 500
nr_objs = [3,5,10]
problem = 'WFG3'

# pf = get_problem(problem, n_var=dimensions, n_obj=nr_obj).pareto_front(ref_dirs)
# reference_point = np.max(f, axis=0)

current_working_dir = os.getcwd()
path = re.search(r'^(.*?[\\/]FEA)', current_working_dir)
path = path.group()

for nr_obj in nr_objs:
    if nr_obj > 3:
        ref_dirs = get_reference_directions("das-dennis", nr_obj, n_partitions=4)
    else:
        ref_dirs = get_reference_directions("das-dennis", nr_obj, n_partitions=12)

    moea1 = partial(SPEA2, population_size=population, ea_runs=ga_run)
    moea2 = partial(NSGA2, population_size=population, ea_runs=ga_run)
    moea3 = partial(MOEAD, ea_runs=ga_run, weight_vector=ref_dirs, n_neighbors=10, problem_decomposition=Tchebicheff())

    partial_methods = [ moea2, moea3]
    names=[ 'NSGA2', 'MOEAD']  # 'SPEA2', 'NSGA2',

    @add_method(MOEA)
    def calc_fitness(variables, gs=None, factor=None):
        if gs is not None and factor is not None:
            full_solution = [x for x in gs.variables]
            for i, x in zip(factor, variables):
                full_solution[i] = x
        else:
            full_solution = variables
        dtlz = get_problem(problem, n_var=dimensions, n_obj=nr_obj)
        objective_values = dtlz.evaluate(full_solution)
        return tuple(objective_values)

    FA = FactorArchitecture(dimensions)
    # FA.load_architecture(path_to_load=path+"/FEA/factor_architecture_files/MEET_MOO/MEET2_DG_random_DTLZ1_5")
    # FA.method = 'MEET2'

    #FA.linear_grouping(1, 1)
    #FA.get_factor_topology_elements()

    #FA.linear_grouping(s, o)
    FA.classic_random_grouping(100)
    FA.get_factor_topology_elements()
    #FA.method = "linear_" + str(s) + '_' + str(o)
    iter = 5
    for i in range(iter):
        for j,alg in enumerate(partial_methods):
            # if s == o:
            #     name = 'CC' + names[j]
            # else:
            #     name = 'F' + names[j]
            name = 'CC'+names[j]
            print(name, FA.method, problem, str(nr_obj))
            print('##############################################\n', i)
            for fea_run in fea_runs:
                start = time.time()
                filename = path + '/results/'+problem+'/' + name + '/' + name + '_'+problem+'_' + str(nr_obj) + \
                        '_objectives_fea_runs_' + str(fea_run) + '_grouping_' + FA.method + time.strftime('_%d%m%H%M%S') + '.pickle'
                feamoo = MOFEA(fea_run, factor_architecture=FA, base_alg=alg, dimensions=dimensions,
                            value_range=[0.0, 1.0], ref_point=np.ones(nr_obj))
                feamoo.run()
                end = time.time()
                try:
                    file = open(filename, "wb")
                    pickle.dump(feamoo, file)
                except OSError:
                    if not os.path.isdir(path + '/results/' + problem.upper() + '/' + name + '/'):
                        try:
                            os.mkdir(path + '/results/' + problem.upper() + '/' + name + '/')
                            file = open(filename, "wb")
                            pickle.dump(feamoo, file)
                        except OSError:
                            if not os.path.isdir(path + '/results/' + problem.upper() + '/'):
                                os.mkdir(path + '/results/' + problem.upper() + '/')
                                os.mkdir(path + '/results/' + problem.upper() + '/' + name + '/')
                            file = open(filename, "wb")
                            pickle.dump(feamoo, file)
                elapsed = end - start
                print(
                    "FEA with ga runs %d and population %s took %s" % (fea_run, FA.method, str(timedelta(seconds=elapsed))))
