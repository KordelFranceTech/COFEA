from utilities.multifilereader import MultiFileReader
import pickle, re, os
from MOO.paretofrontevaluation import ParetoOptimization
from utilities.util import PopulationMember
import numpy as np
from itertools import combinations
from scipy.stats import ttest_ind
from pymoo.util.nds.non_dominated_sorting import find_non_dominated
from pymoo.factory import get_performance_indicator
from pymoo.factory import get_problem

#TODO: make reusable and general

problems = ['DTLZ1', 'DTLZ2', 'DTLZ3', 'DTLZ4', 'DTLZ5', 'DTLZ6', 'DTLZ7']
algorithms = ['SPEA2', 'NSGA2', 'MOEAD']
nr_objs = [5, 10]
decompositions = ['100_100', '100_80', 'classic_random_100'] #'MEET2', 'diff_grouping', '100_80', '200_160', 'linear'

po = ParetoOptimization()

import csv

# field names
fields = ['problem', 'n_obj', 'algorithm', 'decomp', 'hv', 'spread', 'nd_size', 'n_experiments']

# data rows of csv file
for n_obj in nr_objs:
    rows = []
    if n_obj < 10:
        max_ref = 10000
    else:
        max_ref = 50
    for problem in problems:
        for decomp in decompositions:
            print(problem)
            print('************************************************')
            print(decomp)
            #file_regex = r'_1000_dimensions_3_objectives_ea_runs_[0-9]*_population_500_'

            t_test_pop = dict()
            total_nondom_pop = []

            for alg in algorithms:
                if decomp == "100_100" or decomp == "classic_random_100":
                    name = "CC"+alg
                else:
                    name = "F"+ alg
                file_regex = name + r'_' + problem + r'_' + re.escape(str(n_obj)) + r'_objectives_fea_runs_20_grouping_(.*)' + decomp
                print('************************************************')
                print(name)
                print('************************************************')
                stored_files = MultiFileReader(file_regex=file_regex, dir = "/media/amy/WD Drive/"+problem+"/"+name+"/")
                file_list = stored_files.path_to_files
                t_test_pop[name] = dict()
                if decomp != 'linear':
                    t_test_pop[name][decomp] = dict()
                    t_test_pop[name][decomp]["HV"] = []
                    t_test_pop[name][decomp]["spread"] = []
                ND_size = 0
                amount = 0
                if len(file_list) != 0:
                    for file in file_list:
                        amount += 1
                        try:
                            obj = pickle.load(open(file, 'rb'))
                        except EOFError:
                            print("error in file: ", file)
                        if decomp == 'linear':
                            factor_length = len(obj.factor_architecture.factors[0])
                            if factor_length == 200:
                                decomp = '200_160'
                            else:
                                decomp = '100_80'
                            t_test_pop[name][decomp] = dict()
                            t_test_pop[name][decomp]["HV"] = []
                            t_test_pop[name][decomp]["spread"] = []
                        new_div = po.calculate_diversity(obj.nondom_archive)
                        hv = get_performance_indicator("hv",
                                                    ref_point=np.ones(n_obj)*max_ref)  # hypervolume(np.array(updated_pareto_set))
                        # [print(sol.fitness) for sol in obj.nondom_archive]
                        print('hv')
                        new_hv = hv.calc(np.array([np.array(sol.fitness) for sol in obj.nondom_archive]))
                        print('hv: ', new_hv)
                        t_test_pop[name][decomp]["spread"].append(new_div) #obj.iteration_stats[-1]['diversity'])
                        t_test_pop[name][decomp]["HV"].append(new_hv)
                        ND_size = ND_size + obj.iteration_stats[-1]['ND_size']
                        #total_nondom_pop.extend(obj.nondom_archive)
                    HV = np.sum(t_test_pop[name][decomp]["HV"])/amount
                    ND_size = ND_size/amount
                    spread = np.sum(t_test_pop[name][decomp]["spread"])/amount
                    rows.append([problem, str(n_obj), name, decomp, str(HV), str(spread), str(ND_size), str(amount)])
                    # #print(file_regex)
                    print('\n------------------------------------------------\n')
                    print('hv: ', HV, 'nd: ', ND_size, 'spread: ', spread)
            # nondom_fitnesses = []
            # for x in total_nondom_pop:
            #     try:
            #         nondom_fitnesses.append(PopulationMember(x.variables,x.fitness))
            #     except:
            #         nondom_fitnesses.append(PopulationMember([c.nitrogen for c in x.variables],x.objective_values))
            # nondom_indeces = find_non_dominated(np.array([np.array(x.fitness) for x in nondom_fitnesses]))
            #     #np.array([np.array(x.fitness) for x in total_nondom_pop]))
            # nondom_pop = [nondom_fitnesses[i] for i in nondom_indeces]
            # dict_ = po.calculate_diversity(nondom_pop) #po.evaluate_solution(nondom_pop, [1,1,1])
            # print("union div: ", dict_)
            print('**************************************************\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n')

            # param_perm = combinations(decompositions, 2)
            # for perm in param_perm:
            #     print(perm)
            #     print(ttest_ind(t_test_pop[perm[0]]["spread"], t_test_pop[perm[1]]["spread"]))
            #     print(ttest_ind(t_test_pop[perm[0]]["HV"], t_test_pop[perm[1]]["HV"]))
            #     print('\n\n')

    # name of csv file
    filename = str(n_obj) + "_objectives_statistics.csv"

    # writing to csv file
    if os.path.isfile(filename):
        with open(filename, 'a') as csvfile:
            csvwriter = csv.writer(csvfile)
            #csvwriter.writerow(fields)
            csvwriter.writerows(rows)
    
    else:
        with open(filename, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(fields)
            csvwriter.writerows(rows)


