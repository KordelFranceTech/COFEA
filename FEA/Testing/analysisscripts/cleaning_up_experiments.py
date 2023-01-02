from utilities.multifilereader import MultiFileReader
import pickle, re
from MOO.paretofrontevaluation import ParetoOptimization
from utilities.util import PopulationMember
import numpy as np
from itertools import combinations
from scipy.stats import ttest_ind
from pymoo.util.nds.non_dominated_sorting import find_non_dominated

problems = ['DTLZ1']
algorithms = ['FMOEAD', 'CCMOEAD']
nr_objs = [3]

po = ParetoOptimization()
file_set = set()

for problem in problems:
    print('************************************************')
    print(problem)
    for n_obj in nr_objs:
        print(n_obj)
        # file_regex = r'_1000_dimensions_3_objectives_ea_runs_[0-9]*_population_500_'

        t_test_pop = dict()
        total_nondom_pop = []

        for alg in algorithms:
            print('************************************************')
            print(alg)
            print('************************************************')
            file_regex = re.escape(alg) + r'_' + re.escape(problem) + r'_3_objectives_fea_runs_20_grouping'
            stored_files = MultiFileReader(file_regex=file_regex,
                                           dir="D:/" + problem + "/" + alg + "/")
            file_list = stored_files.path_to_files
            print(len(file_list))
            for file in file_list:
                obj = pickle.load(open(file, 'rb'))
                #new_div = po.calculate_diversity(obj.nondom_archive)
                for sol in obj.nondom_archive:
                    for f in sol.fitness:
                        if f < 0:
                            file_set.add(file)
[print(f +'\n') for f in file_set]