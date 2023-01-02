from utilities.multifilereader import MultiFileReader
import pickle, random
import numpy as np
from pymoo.util.nds.non_dominated_sorting import find_non_dominated
from itertools import combinations

#TODO: make reusable and general

algorithms = ['MOEAD', 'SPEA2', 'NSGA2']
objs = [10]
problems = ['DTLZ1', 'DTLZ2', 'DTLZ3', 'DTLZ4', 'DTLZ5', 'DTLZ6', 'DTLZ7', 'WFG1', 'WFG2', 'WFG3', 'WFG4', 'WFG5', 'WFG7']
comparing = ['population_500']#, 'grouping_linear_100_100', 'grouping_linear_100_80', 'classic_random_100'] # 'grouping_200_200', 'grouping_200_160', 'grouping_MEET2', 'grouping_diff_grouping']

for problem in problems:
    print(problem)
    for obj in objs:
        alg_list = []
        file_regex = r'_'+problem+r'_(.*)'+str(obj)+r'_objectives_'
        stored_files = MultiFileReader(file_regex, dir = "/media/amy/WD Drive/"+problem+"/")
        experiment_filenames = stored_files.path_to_files
        total_front = []
        nondom_solutions = dict()
        lengths = dict()
        for alg in algorithms:
            # print('************************************************')
            # print('************************************************')
            for compare in comparing:
                full_compare = compare+'_'+alg
                alg_list.append(full_compare)
                experiment = [x for x in experiment_filenames if compare+'_' in x and alg+'_' in x]
                if experiment:
                    rand_int = random.randint(0, len(experiment)-1)
                    experiment = experiment[rand_int]
                else:
                    break
                try:
                    feamoo = pickle.load(open(experiment, 'rb'))
                except EOFError:
                    print('issues with file: ', experiment)
                solutions = np.array([np.array(x.fitness) for x in feamoo.nondom_archive])
                total_front.extend(solutions)
                nondom_solutions[full_compare] = solutions
                lengths[full_compare] = len(solutions)
                #print(compare, ' len of front', len(solutions))

        #print('total front length ', len(total_front))
        # total_front = np.vstack((nondom_solutions['CCEA'], nondom_solutions['FEA'], nondom_solutions['NSGA']))
        indeces = find_non_dominated(np.array(total_front))

        lb = 0
        ub = 0
        for i, compare in enumerate(alg_list):
            try:
                ub += lengths[compare]
            except KeyError:
                print("no results for: ", compare)
            if i != 0:
                try:
                    lb += lengths[alg_list[i-1]]
                except KeyError:
                    print("no results")
            #print('upper and lower: ', ub, lb)
            print(len([x for x in indeces if lb <= x < ub]) / len(indeces))

            # pair_compare = [comb for comb in combinations(comparing, 2)]
            # for pair in pair_compare:
            #     print(pair)
            #     to_compare = np.vstack((nondom_solutions[pair[0]], nondom_solutions[pair[1]]))
            #     indeces = find_non_dominated(to_compare)
            #     print(pair[0], lengths[pair[0]], len([x for x in indeces if x < lengths[pair[0]]])/lengths[pair[0]])
            #     print(pair[1], lengths[pair[1]], len([x for x in indeces if lengths[pair[0]] <= x <= lengths[pair[0]]+lengths[pair[1]]])/lengths[pair[1]])

        # print('\n------------------------------------------------\n')