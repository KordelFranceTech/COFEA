from datetime import timedelta

from pymoo.decomposition.tchebicheff import Tchebicheff
from pymoo.factory import get_problem, get_reference_directions

from MOO.MOEA import MOEA, NSGA2, SPEA2, MOEAD
from utilities.util import *

import os, re, time, pickle
from pymoo.problems.many.dtlz import DTLZ1

dimensions = 1000
ga_run = 200
population = 500
nr_objs = [3, 5, 10]

current_working_dir = os.getcwd()
path = re.search(r'^(.*?[\\/]FEA)', current_working_dir)
path = path.group()

problems = ['WFG1', 'WFG2', 'WFG3']

# moea1 = partial(SPEA2, population_size=population, ea_runs=ga_run)
# moea2 = partial(NSGA2, population_size=population, ea_runs=ga_run)
# moea3 = partial(MOEAD, ea_runs=ga_run, weight_vector=ref_dirs, n_neighbors=10, problem_decomposition=Tchebicheff())

names = ['NSGA2', 'SPEA2', 'MOEAD'] #'SPEA2',
problems = ["dtlz1", "dtlz2", "dtlz3", "dtlz4", "dtlz5"]

for problem in problems:
    print(problem)
    for nr_obj in nr_objs:
        print(nr_obj)
        @add_method(MOEA)
        def calc_fitness(variables, gs=None, factor=None):
            dtlz = get_problem(problem, n_var=dimensions, n_obj=nr_obj)
            objective_values = dtlz.evaluate(variables)
            return tuple(objective_values)

        for name in names:
            print(name)
            for i in range(5):
                print('##############################################\n', i)
                start = time.time()
                filename = path + '/results/' + problem.upper() + '/' + name + '/' + name + '_' + problem.upper() + '_' + str(dimensions) + '_dimensions_' + str(nr_obj) + \
                    '_objectives_ea_runs_' + str(ga_run) + '_population_' + str(population) + time.strftime(
                    '_%d%m%H%M%S') + '.pickle'
                if name == 'SPEA2':
                    moo = SPEA2(dimensions=dimensions, value_range=[0.0, 1.0], reference_point=list(np.ones(nr_obj)),
                               ea_runs=ga_run)
                elif name == 'MOEAD':
                    if nr_obj > 3:
                        n_part = 4
                    else:
                        n_part = 12
                    ref_dirs = get_reference_directions("das-dennis", nr_obj, n_partitions=n_part)
                    # pf = get_problem("dtlz1", n_var=dimensions, n_obj=nr_obj).pareto_front(ref_dirs)
                    # reference_point = np.max(pf, axis=0)
                    moo = MOEAD(dimensions=dimensions, value_range=[0.0, 1.0], reference_point=list(np.ones(nr_obj)),
                                ea_runs=ga_run, weight_vector=ref_dirs, n_neighbors=10, problem_decomposition=Tchebicheff())
                else:
                    moo = NSGA2(dimensions=dimensions, value_range=[0.0, 1.0], reference_point=list(np.ones(nr_obj)),
                                ea_runs=ga_run)
                moo.run()
                end = time.time()
                try:
                    file = open(filename, "wb")
                except FileNotFoundError:
                    if not os.path.isdir(path + '/results/' + problem.upper() + '/' + name + '/'):
                        try:
                            os.mkdir(path + '/results/' + problem.upper() + '/' + name + '/')
                        except FileNotFoundError:
                            if not os.path.isdir(path + '/results/' + problem.upper() + '/'):
                                os.mkdir(path + '/results/' + problem.upper() + '/')
                                os.mkdir(path + '/results/' + problem.upper() + '/' + name + '/')
                            file = open(filename, "wb")

                    file = open(filename, "wb")
                pickle.dump(moo, file)
                elapsed = end - start
                print(
                    "Alg with %d runs and population %d took %s" % (ga_run, population, str(timedelta(seconds=elapsed))))
