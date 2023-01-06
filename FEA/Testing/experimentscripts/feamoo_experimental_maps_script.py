import os, re
import pickle

from optimizationproblems.prescription import Prescription
from MOO.MOEA import *
from utilities.util import *

field_names = ['Henrys', 'Sec35Mid', 'Sec35West']
current_working_dir = os.getcwd()
path = re.search(r'^(.*?\\FEA)',current_working_dir)
path = path.group()

field_1 = pickle.load(open(path + '/utilities/saved_fields/Henrys.pickle', 'rb')) # /home/alinck/FEA
field_2 = pickle.load(open(path + '/utilities/saved_fields/sec35mid.pickle', 'rb'))
field_3 = pickle.load(open(path + '/utilities/saved_fields/sec35west.pickle', 'rb'))
fields_to_test = [field_1, field_2, field_3]

fea_runs = 20
ga_runs = [100]
population_sizes = [500]


def create_strip_groups(field):
    cell_indeces = field.create_strip_trial()
    factors = []
    single_cells = []
    sum = 0
    for j, strip in enumerate(cell_indeces):
        if len(strip.original_index) == 1:
            single_cells.append(sum)
        else:
            factors.append([i+sum for i, og in enumerate(strip.original_index)])
        sum = sum + len(strip.original_index)
    if single_cells:
        factors.append(single_cells)
    return factors


for i, field in enumerate(fields_to_test):
    print(field_names[i], '-- NSGA')
    # #FA = FactorArchitecture(len(field.cell_list))
    # factors = create_strip_groups(field)
    # print(factors)
    # factor_size = np.mean([len(f) for f in factors])
    # print('avg size of strips: ', np.round(factor_size))
    # FA.linear_grouping(int(np.round(factor_size)), int(np.round(factor_size/2)))
    # #FA.factors = create_strip_groups(field)
    # FA.get_factor_topology_elements()
    #nsga = NSGA2

    @add_method(NSGA2)
    def calc_fitness(variables, gs=None, factor=None):
        pres = Prescription(variables=variables, field=field, factor=factor)
        if gs is not None:
            global_solution = Prescription(variables = gs.variables, field = field)
            pres.set_fitness(global_solution=global_solution)
        else:
            pres.set_fitness()
        return pres.objective_values

    for j in range(5):
        for population in population_sizes:
            for ga_run in ga_runs:
                start = time.time()
                filename = path + '/results/prescriptions/NSGA2_' + field_names[i] + '_strip_trial_3_objectives_ga_runs_' + str(ga_run) + '_population_' + str(population) + time.strftime('_%d%m%H%M%S') + '.pickle'
                #feamoo = MOFEA(fea_runs, ga_run, population, FA, nsga, dimensions=len(field.cell_list), combinatorial_options=field.nitrogen_list)
                #feamoo.run()
                nsga = NSGA2(population_size=population, ea_runs=ga_run, dimensions=len(field.cell_list),
                             combinatorial_values=field.nitrogen_list, ref_point=[1,1,1])
                nsga.run()
                end = time.time()
                file = open(filename, "wb")
                pickle.dump(nsga, file)
                elapsed = end-start
                print("NSGA with ga runs %d and population %d took %s"%(ga_run, population, str(timedelta(seconds=elapsed))))
