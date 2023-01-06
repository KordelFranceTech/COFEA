from optimizationproblems.continuous_functions import Function
from FEA.varinteraction import MEE, RandomTree
from FEA.factorevolution import FEA
from FEA.factorarchitecture import FactorArchitecture
from basealgorithms.pso import PSO
import random
import time

import functools
import sys


def output_callback(output_file, fea, fea_run):
    # print(f'WRITING:  , , {fea_run}, {fea.global_fitness}')
    output_file.write(f" , , {fea_run}, {fea.global_fitness}\n")
    output_file.flush()

def average_factors(fa: FactorArchitecture):
    print(len(fa.factors))
    size = 0
    for f in fa.factors:
        size += len(f)
    print(size / len(fa.factors))
    return len(fa.factors), size / len(fa.factors)

if __name__ == '__main__':

    # arguments are used to pass through which function to use, the relevant shift data file, and the matrix data file if present
    if len(sys.argv) == 4:
        outputcsv = open(f'./MeetRandom/Experiment_{dim}_f{int(sys.argv[1])}.csv', 'a')
        f = Function(int(sys.argv[1]), shift_data_file=sys.argv[2], matrix_data_file=sys.argv[3])
    elif len(sys.argv) == 3:
        outputcsv = open(f'./MeetRandom/Experiment_{dim}_f{int(sys.argv[1])}.csv', 'a')
        f = Function(int(sys.argv[1]), shift_data_file=sys.argv[2])
    else:
        # Option to manually adjust file names if you don't want to use cmdline args, currently looking at CEC2018 Function 20
        f = Function(20, shift_data_file="f20_o.txt")
        outputcsv = open('MeetRandom/Experiment_1000_f20_thing.csv', 'a')
    print(f.function_to_call)

    dg_file = open(f'MeetRandom/Graph4_{f.function_to_call}_thing.csv', 'a')

    dim = 1000
    dg_epsilon = 0.001

    k = "epsilon, pop_size, Thing, Thing Fac, Thing Runs, Thing PSO"
    outputcsv.write(k + '\n')
    s = time.time()
    print(dg_epsilon)

    # Load factorarchitecture from text file and save object
    f.evals = 0
    thing = FactorArchitecture(dim=dim)
    thing.load_txt_architecture('arch', 1000)
    thing.save_architecture(f'MeetRandom/{f.function_to_call}_thing')


    """
    DIFFERENT ARCHITECTURES
    """
    # f.evals = 0
    # odg = FactorArchitecture(dim=dim)
    # odg.overlapping_diff_grouping(f, dg_epsilon)
    # odg.save_architecture(f'MeetRandom/{f.function_to_call}_odg')

    # dg = FactorArchitecture(dim=dim)
    # print('starting dg')
    # f.evals = 0
    # dg.diff_grouping(f, dg_epsilon)
    # dg.save_architecture(f'MeetRandom/{f.function_to_call}_dg')

    # ccea = FactorArchitecture(dim=dim)
    # print('starting single')
    # f.evals = 0
    # ccea.single_grouping()
    # ccea.save_architecture(f'MeetRandom/{f.function_to_call}_single')

    # im = RandomTree(f, dim, 3000, dg_epsilon, 0.000001)

    # print('starting random')
    # T = im.run(5)
    # print("finished Random ")
    # meet = FactorArchitecture(dim=dim)
    # meet.MEET(T)
    # meet.save_architecture(f"MeetRandom/{f.function_to_call}_rand")

    # meet2 = FactorArchitecture(dim=dim)
    # meet2.MEET2(T)
    # meet2.save_architecture(f"MeetRandom/{f.function_to_call}_2_rand")

    for pop_size in [10]:
        for trial in range(1):
            outputcsv.write(f'{dg_epsilon},{pop_size},')
            outputcsv.flush()

            fa = FactorArchitecture()
            # load object from file
            fa.load_architecture(f"MeetRandom/{f.function_to_call}_thing")
            print(f"DG {len(fa.factors)}")

            f.evals = 0
            # run algorithm
            fea = FEA(f, 50, 15, pop_size, fa, PSO, seed=trial, log_file=dg_file)
            fea_run, pso_runs = fea.run()
            print(f"DG, \t\t{fea.global_fitness}\n")

            outputcsv.write(f'{fea.global_fitness},{len(fa.factors)},{fea_run},{sum(pso_runs) / len(pso_runs)}\n')
            outputcsv.flush()

        """
        Regular PSO
        """
        # print("PSO")
        # pso = PSO(generations=3000, population_size=pop_size, function=f, dim=dim)  # generations=3000
        # gbest = pso.run()
        # summary['PSO'] = pso.gbest.fitness
        #
        # print(summary)
        #
        # keys = k.split(',')
        # if all(elem in keys for elem in summary.keys()):
        #     line_out = ','.join([str(summary[key]) for key in keys])
        #     outputcsv.write(line_out + '\n')
        #     outputcsv.flush()
        # else:
        #     print(f'{summary.keys()} != {keys}')
        # outputfile.close()