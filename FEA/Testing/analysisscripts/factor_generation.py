from pymoo.factory import get_problem
from FEA.factorarchitecture import *
from FEA.varinteraction import *
import re

current_working_dir = os.getcwd()
path = re.search(r'^(.*?[\\/]FEA)', current_working_dir)
path = path.group()

n_objs = [3,5]
dimensions = 1000

for n_obj in n_objs:
    dtlz = get_problem("dtlz1", n_var=dimensions, n_obj=n_obj)
    complete_fa = FactorArchitecture(dim=dimensions)
    for i in range(n_obj):
        dg = DGInteraction(dtlz, dimensions, epsilon=5, n_obj=i, moo=True)
        #dg = Entropic(dtlz, dimensions, samples=5, de_thresh=0.001, delta=0.001, moo=True, n_obj=i)
        #mee = MEE(dtlz, dimensions, 10, mic_thresh= 0.1, de_thresh= 0.0001, delta= 0.000001, measure=dg)
        #im = mee.get_IM()
        rt = RandomTree(dimensions, dg)
        im = rt.run(trials=100)
        fa = FactorArchitecture(dim=dimensions)
        fa.MEET2(im, number_of_factors=25)
        fa.save_architecture(path_to_save=path+"/FEA/factor_architecture_files/MEET_MOO/MEET2_DG_random_DTLZ1_"+str(n_obj)+"_obj_"+str(i))
        for f in fa.factors:
            print(len(f))
            complete_fa.factors.append(f)
    print(len(complete_fa.factors))
    print(complete_fa.factors)
    complete_fa.get_factor_topology_elements()
    complete_fa.save_architecture(path_to_save=path+"/FEA/factor_architecture_files/MEET_MOO/MEET2_DG_random_DTLZ1_"+str(n_obj))

# moofa = MooFactorArchitecture(dim=1000, problem=dtlz, n_obj=3)
# factors = moofa.create_objective_factors()
# print(factors.factors)
# factors.save_architecture()