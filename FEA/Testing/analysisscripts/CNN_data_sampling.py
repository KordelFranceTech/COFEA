from predictionalgorithms.CNN_yieldpredictor.Predictor.YieldMapPredictor import *
from predictionalgorithms.yieldprediction import *
from optimizationproblems.prescription import Prescription
from utilities.util import *
import pandas as pd
import numpy as np
import pickle, random, utm, re

current_working_dir = os.getcwd()
path = re.search(r'^(.*?[\\/]FEA)', current_working_dir)
path_ = path.group()

np.set_printoptions(suppress=True)
field = pickle.load(open(path_ + '/utilities/saved_fields/sec35mid.pickle', 'rb'))
field.field_name = 'sec35middle'

agg_file = "home/amy/Documents/Work/OFPE/Data/Sec35Mid/broyles_sec35mid_2016_yl_aggreg_20181112.csv"
reduced_agg_files = ["/home/amy/Documents/Work/OFPE/Data/Sec35Mid/broyles_sec35mid_2016_yl_aggreg_20181112.csv" ,#I:/FieldData/sec35mid/broyles_sec35mid_2016_yl_aggreg_20181112.csv",
                     "/home/amy/Documents/Work/OFPE/Data/Sec35Mid/reduced_broyles_sec35mid_cnn_random.csv",
                     "/home/amy/Documents/Work/OFPE/Data/Sec35Mid/reduced_broyles_sec35mid_cnn_spatial.csv",
                     "/home/amy/Documents/Work/OFPE/Data/Sec35Mid/reduced_broyles_sec35mid_cnn_aggregate.csv"]
type_of_file = ['full', 'random', 'spatial', 'aggregate']
# df = pd.read_csv(agg_file)
# y_labels = df['yl_2016']  # yl18_bu_ac
data_to_use = ['x', 'y', 'n_lbs_ac', 'elev_m', 'slope_deg', 'ndvi_2012', 'ndvi_2014', 'ndvi_2015', 'yl14_nn_bu_ac',
               'n15_lbs_ac', 'n14_lbs_ac']
# HENRYS ['x', 'y', 'n_lbs_ac', 'elev_m', 'slope_deg', 'ndvi15', 'ndvi16', 'ndvi17', 'yl16_nn_bu_ac','n16_lbs_ac']
# SEC35MID ['x', 'y', 'n_lbs_ac', 'elev_m', 'slope_deg', 'ndvi_2012', 'ndvi_2014', 'ndvi_2015', 'yl14_nn_bu_ac', 'n15_lbs_ac', 'n14_lbs_ac']

"""
Initialize CNN model
"""
cnn = YieldMapPredictor(filename='/home/amy/Documents/Work/OFPE/Data/broyles_10m_yldDat_with_sentinel.csv',
                        field='sec35middle', pred_year=2020, training_years=[2016, 2018])
# Load prediction data (it will be saved in cnn.data)
cnn.load_pred_data(objective='yld')
# Load model weights
cnn.model = cnn.init_model(modelType='Hyper3DNet')
path_weights = '/home/amy/projects/OFPETool-master/static/uploads/Model-Hyper3DNet-sec35middle--Objective-yld/Hyper3DNet' + "-sec35middle--Objective-yld"
cnn.model.loadModel(path=path_weights)
cnn.patches, centers = cnn.extract2DPatches()

"""
Adjust patches and centers for use in MOO
"""
# get lat lon coordinates for the center points of CNN data
lat_lon = [utm.to_latlon(cnn.coords[x, y][0], cnn.coords[x, y][1], 12, northern=True) for (x, y) in centers]
# assign cell indeces to each of the centers
cnn.centers = create_indexed_dataframe(dps=pd.DataFrame(lat_lon), field=field)
# get indeces of points that fall within the cells of the grid
indeces = [center[2] for center in np.array(cnn.centers)]
points_in_cells_indeces = np.array(indeces).astype(int)
# for each of the patches, check if the patch falls within one of the cells, otherwise assign base nitrogen rate
for idx, patch in enumerate(cnn.patches):
    if idx not in points_in_cells_indeces:
        # get array of NR to assign base rate to
        nr = cnn.patches[idx, 0, 0, :]
        nr[nr != 0] = 120  # assign field.base_rate
        cnn.patches[idx, 0, 0, :] = nr

"""
Run experiments with reduced data sets
"""
field.fixed_costs = 1000
#cell_predictions = np.zeros((10, 4, len(field.cell_list)))
from itertools import combinations
combs = combinations(type_of_file, 2)

for k in range(10):
    cell_predictions = dict()
    total_predictions = dict()
    random_global_variables = [random.randrange(0, 150) for x in range(len(field.cell_list))]
    pr = Prescription(variables=random_global_variables, field=field)
    for i, reduced_agg_file in enumerate(reduced_agg_files):
        yp = YieldPredictor(prescription=pr, field=field, agg_data_file=reduced_agg_file, trained_model=cnn,
                            data_headers=data_to_use, cnn_bool=True)

        yp.adjust_nitrogen_data(prescription=pr, cnn=True)
        total_yield = yp.calculate_yield(cnn=True)
        cell_predictions[type_of_file[i]] = yp.cell_predictions
        total_predictions[type_of_file[i]] = total_yield

        print('##################')
        print(type_of_file[i])
        print('##################')
        print('yield for whole field: ', total_yield)
        #print('cell specific predictions: ', yp.cell_predictions)
    combs = combinations(type_of_file, 2)
    for comb in combs:
        print(comb)
        diff = np.array(cell_predictions[comb[0]]) - np.array(cell_predictions[comb[1]])
        print('avg difference: ', np.mean(diff))
        print('total difference: ', total_predictions[comb[0]] - total_predictions[comb[1]])
        
    print('\n')