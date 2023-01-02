from predictionalgorithms.CNN_yieldpredictor.Predictor.YieldMapPredictor import *
from predictionalgorithms.yieldprediction import *
from optimizationproblems.prescription import Prescription
from FEA.factorarchitecture import FactorArchitecture
from utilities.util import *
from MOO.MOEA import *
from MOO.MOFEA import MOFEA
import pandas as pd
import numpy as np
from datetime import timedelta
import pickle, random, re, os, time, utm

np.set_printoptions(suppress=True)

fea_runs = 10
ga_runs = [20]
population_sizes = [25]
upper_bound = 150

field_names = ['sec35middle', 'Sec35West'] #'Henrys',
current_working_dir = os.getcwd()
path_ = re.search(r'^(.*?[\\/]FEA)',current_working_dir)
path_ = path_.group()
print('path: ', path_)

#field_1 = pickle.load(open(path + '/utilities/saved_fields/Henrys.pickle', 'rb')) # /home/alinck/FEA
field_2 = pickle.load(open(path_ + '/utilities/saved_fields/sec35mid.pickle', 'rb'))
#field_3 = pickle.load(open(path + '/utilities/saved_fields/sec35west.pickle', 'rb'))
fields_to_test = [field_2] #[field_1, field_2, field_3]

for i, field in enumerate(fields_to_test):
    field.field_name = field_names[i]
    field.angle = 0
    field.cell_length_min = field.cell_height
    field.cell_length_max = field.cell_height
    agg_files = ["/home/amy/Documents/Work/OFPE/Data/Sec35Mid/broyles_sec35mid_2016_yl_aggreg_20181112.csv"]
    reduced_agg_files = ["/home/amy/Documents/Work/OFPE/Data/Sec35Mid/reduced_broyles_sec35mid_cnn_random.csv"]
    df = pd.read_csv(agg_files[i])
    y_labels = df['yl_2016'] #yl18_bu_ac
    data_to_use = ['x', 'y', 'n_lbs_ac', 'elev_m', 'slope_deg', 'ndvi_2012', 'ndvi_2014', 'ndvi_2015', 'yl14_nn_bu_ac', 'n15_lbs_ac', 'n14_lbs_ac']
    #HENRYS ['x', 'y', 'n_lbs_ac', 'elev_m', 'slope_deg', 'ndvi15', 'ndvi16', 'ndvi17', 'yl16_nn_bu_ac','n16_lbs_ac']
    #SEC35MID ['x', 'y', 'n_lbs_ac', 'elev_m', 'slope_deg', 'ndvi_2012', 'ndvi_2014', 'ndvi_2015', 'yl14_nn_bu_ac', 'n15_lbs_ac', 'n14_lbs_ac']
    cnn = YieldMapPredictor(filename='/home/amy/Documents/Work/OFPE/Data/broyles_10m_yldDat_with_sentinel.csv', field=field.field_name, pred_year=2018, training_years=[2016] )
    # Load prediction data (it will be saved in cnn.data)
    cnn.load_pred_data(objective='yld')
    # Load model weights
    cnn.model = cnn.init_model(modelType='Hyper3DNet')
    path_weights = '/home/amy/projects/OFPETool-master/static/uploads/Model-Hyper3DNet-sec35middle--Objective-yld/Hyper3DNet' + "-" + field.field_name+ "--Objective-yld"
    cnn.model.loadModel(path=path_weights)
    cnn.patches, centers = cnn.extract2DPatches()

    #adjust patches and centers for use in MOO
    lat_lon = [utm.to_latlon(cnn.coords[x, y][0], cnn.coords[x, y][1], 12, northern=True) for (x, y) in centers]
    # add cell ids to centers dataframe
    cnn.centers = create_indexed_dataframe(dps=pd.DataFrame(lat_lon), field=field)
    indeces = [center[2] for center in np.array(cnn.centers)]
    points_in_cells_indeces = np.array(indeces).astype(int)
    for idx, patch in enumerate(cnn.patches):
        if idx not in points_in_cells_indeces:
            nr = cnn.patches[idx, 0, 0, :]
            nr[nr != 0] = 120#field.base_rate
            cnn.patches[idx, 0, 0, :] = nr

    field.fixed_costs = 1000
    random_global_variables = [random.randrange(0, upper_bound) for x in range(len(field.cell_list))]
    pr = Prescription(variables=random_global_variables, field=field)
    yp = YieldPredictor(prescription=pr, field=field, agg_data_file=reduced_agg_files[i], trained_model=cnn, data_headers=data_to_use, cnn_bool=True)

    #yp.calculate_yield(cnn=True)
    FA = FactorArchitecture(len(field.cell_list))
    FA.factors = field.create_strip_groups(overlap=True)
    FA.get_factor_topology_elements()
    nsga = NSGA2

    @add_method(NSGA2)
    def calc_fitness(variables, gs=None, factor=None):
        pres = Prescription(variables=variables, field=field, factor=factor, optimized=True, yield_predictor=yp)
        if gs:
            #global_solution = Prescription(variables=gs.variables, field=field)
            pres.set_fitness(global_solution=gs.variables, cont_bool=True)
        else:
            pres.set_fitness(cont_bool=True)
        return pres.objective_values
    
    for j in range(5):
        for population in population_sizes:
            for ga_run in ga_runs:
                start = time.time()
                filename = path_ + '/results/prescriptions/CNN_optimized/FEAMOO_' + field_names[i] + '_strip_trial_3_objectives_ga_runs_' + str(ga_run) + '_population_' + str(population) + time.strftime('_%d%m%H%M%S') + '.pickle'
                feamoo = MOFEA(fea_runs, ga_run, population, factor_architecture=FA, base_alg=nsga, dimensions=len(field.cell_list),
                               value_range=[0,upper_bound], ref_point=[1, 1, 1]) #, combinatorial_options=field.nitrogen_list)
                feamoo.run()
                # nsga = NSGA2(population_size=population, ea_runs=ga_run, dimensions=len(field.cell_list),
                #              upper_value_limit=150, ref_point=[1, 1, 1])
                # nsga.run()
                end = time.time()
                pickle.dump(nsga, open(filename, "wb"))
                elapsed = end-start
                print("NSGA with ga runs %d and population %d took %s"%(ga_run, population, str(timedelta(seconds=elapsed))))



